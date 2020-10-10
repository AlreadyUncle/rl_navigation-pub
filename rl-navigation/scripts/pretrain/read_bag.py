import numpy as np
from os import listdir
from os.path import isfile, join
import rosbag
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist, Pose, Pose2D
from nav_msgs.msg import Odometry
import sys 
sys.path.append('/home/shiqing/rl_navigation_ws/src/rl-navigation/scripts')
from utils.common import *
from utils.robot import *
from tf.transformations import euler_from_quaternion, quaternion_from_euler

def get_states_and_labels():
    bag_paths = ['/home/shiqing/rl_navigation_ws/src/rl-navigation/recorded_bags/one_obs/pretrain_map2_one_dyn_obs.bag']
    # bag_paths = [join(bag_base_path, f) for f in listdir(bag_base_path) if isfile(join(bag_base_path, f))]

    # read bags
    times = []
    tmp_goal_list = []
    goal_time_list = []
    scan_list = []
    scan_time_list = []
    twist_list = []
    pose_list = []
    odom_time_list = []
    action_list = []

    twist_list_obs = []
    pose_list_obs = []
    odom_time_list_obs = []

    for bag_path in bag_paths:
        bag = rosbag.Bag(bag_path)
        for topic, msg, t in bag.read_messages():
            if topic == '/robot_0/cmd_vel':
                action_list.append([msg.linear.x, msg.angular.z])
                times.append(t.to_sec())
            elif topic == '/robot_0/move_base_simple/goal':
                tmp_goal_list.append(msg.pose)
                goal_time_list.append(t.to_sec())
        
        i = 1
        for topic, msg, t in bag.read_messages():
            if abs(t.to_sec()-times[i-1])<=0.12 :
                if topic == '/robot_0/scan' and len(scan_time_list)<i:
                    scan_list.append(msg.ranges)
                    scan_time_list.append(t.to_sec())
                elif topic == '/robot_0/base_pose_ground_truth' and len(odom_time_list)<i:
                    pose_list.append(msg.pose.pose)
                    twist_list.append(msg.twist.twist)
                    odom_time_list.append(t.to_sec())
                elif topic == '/robot_1/base_pose_ground_truth' and len(odom_time_list_obs)<i:
                    pose_list_obs.append(msg.pose.pose)
                    twist_list_obs.append(msg.twist.twist)
                    odom_time_list_obs.append(t.to_sec())
            if len(scan_time_list) == i and len(odom_time_list)==i and len(odom_time_list_obs)==i:
                i += 1
            if i > len(times):
                break
        
        # replicate goal list
        i = 1
        goal_list = []
        for time in times:
            if i>=len(goal_time_list):
                goal_list.append(tmp_goal_list[i-1])
            elif time < goal_time_list[i]:
                goal_list.append(tmp_goal_list[i-1])
            elif time >= goal_time_list[i]:
                i += 1
                goal_list.append(tmp_goal_list[i-1])

    #print(len(times))
    #print(len(goal_list))
    #print(len(odom_time_list))
    #print(len(scan_list))
    #print(len(odom_time_list_obs))

    # post-processing and build state
    n_data = len(times)
    total_laser_samples = len(scan_list[0])
    fov = 360.0
    laser_slice = int((180.0*total_laser_samples)/(fov*36.0))
    laser_slice_offset = int((total_laser_samples*(fov-180))/(2*fov))
    #laser_slice = int((total_laser_samples - 2*laser_slice_offset)/36.0)
    max_clip = 10.0
    laser_sensor_offset = 0.0
    inverse_distance_states = True
    v_lims = [0.0, 1.0]
    w_lims = [-1.0, 1.0]
    collision_radius  = 0.178

    states = []
    labels = []

    for i in range(n_data):
        ranges = scan_list[i]
        laser_data_states = do_linear_transform(np.array( [min(ranges[current: current + laser_slice]) for current in xrange(laser_slice_offset, len(ranges) - laser_slice_offset, laser_slice)]) - laser_sensor_offset, max_clip, inverse_distance_states)
        # print(xrange(laser_slice_offset, len(ranges) - laser_slice_offset, laser_slice))
        state = list(laser_data_states[0:36])

        position_data_state = do_linear_transform(get_distance(pose_list[i].position, goal_list[i].position), max_clip, inverse_distance_states) 
        orientation_to_goal_data_state = get_relative_angle_to_goal(pose_list[i].position, pose_list[i].orientation, goal_list[i].position)/np.pi
        state += [orientation_to_goal_data_state] 
        state += [position_data_state]

        orientation = pose_list[i].orientation
        angle = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])[2]
        if(angle/np.pi < -1):
            angle = angle + 2*np.pi
        elif(angle/np.pi > 1):
            angle =  angle - 2*np.pi
        state += [angle/np.pi]

        vx = twist_list[i].linear.x * np.cos(angle)
        vy = twist_list[i].linear.x * np.sin(angle)
        vx_state = (2*vx - (v_lims[0] + v_lims[1]))/(v_lims[1] - v_lims[0])
        vy_state = (2*vy - (v_lims[0] + v_lims[1]))/(v_lims[1] - v_lims[0])
        state += [vx_state, vy_state]

        collision_radius_state = do_linear_transform(collision_radius, max_clip,inverse_distance_states)
        state += [collision_radius_state] 

        # mutual distance
        mutual_distance_state = do_linear_transform(get_distance(pose_list[i].position, pose_list_obs[i].position), max_clip, inverse_distance_states) 
        state += [mutual_distance_state]
        # orientation
        orientation_obs = pose_list_obs[i].orientation
        angle_obs = euler_from_quaternion([orientation_obs.x, orientation_obs.y, orientation_obs.z, orientation_obs.w])[2]
        if(angle_obs/np.pi < -1):
            angle_obs += 2*np.pi
        elif(angle_obs/np.pi > 1):
            angle_obs -= 2*np.pi
        # velocity
        vx_obs = twist_list_obs[i].linear.x * np.cos(angle_obs)
        vy_obs = twist_list_obs[i].linear.x * np.sin(angle_obs)
        vx_state_obs = (2*vx_obs - (v_lims[0] + v_lims[1]))/(v_lims[1] - v_lims[0])
        vy_state_obs = (2*vy_obs - (v_lims[0] + v_lims[1]))/(v_lims[1] - v_lims[0])
        
        # distances in the frame of robot_0
        delta_x = pose_list_obs[i].position.x - pose_list[i].position.x
        delta_y = pose_list_obs[i].position.y - pose_list[i].position.y
        transformed_delta_x = np.cos(angle)*delta_x + np.sin(angle)*delta_y
        transformed_delta_y = np.cos(angle)*delta_y - np.sin(angle)*delta_x
        transformed_delta_x_state = do_linear_transform(transformed_delta_x, max_clip, inverse_distance_states)
        transformed_delta_y_state = do_linear_transform(transformed_delta_y, max_clip, inverse_distance_states)
        state += [transformed_delta_x_state, transformed_delta_y_state]
        state += [vx_state_obs, vy_state_obs]
        state += [angle_obs/np.pi]

        # collision distance
        state += [collision_radius_state]
        states.append(state)
    #print(len(state))
    labels = []
    for i in range(n_data):
        v = action_list[i][0]
        v_label = (2*v - (v_lims[0] + v_lims[1]))/(v_lims[1] - v_lims[0])
        w = action_list[i][1]
        w_label = (2*w - (w_lims[0] + w_lims[1]))/(w_lims[1] - w_lims[0])
        labels.append([v_label, w_label])
    
    return (np.array(states, dtype=float), np.array(labels, dtype=float))