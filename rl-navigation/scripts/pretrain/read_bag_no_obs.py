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
    bag_base_path = '/home/shiqing/rl_navigation_ws/src/rl-navigation/recorded_bags'
    bag_paths = [join(bag_base_path, f) for f in listdir(bag_base_path) if isfile(join(bag_base_path, f))]

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

    for bag_path in bag_paths:
        bag = rosbag.Bag(bag_path)
        for topic, msg, t in bag.read_messages():
            if topic == '/cmd_vel':
                action_list.append([msg.linear.x, msg.angular.z])
                times.append(t.to_sec())
            elif topic == '/move_base_simple/goal':
                tmp_goal_list.append(msg.pose)
                goal_time_list.append(t.to_sec())
        
        i = 1
        for topic, msg, t in bag.read_messages():
            if abs(t.to_sec()-times[i-1])<=0.11 :
                if topic == '/scan' and len(scan_time_list)<i:
                    scan_list.append(msg.ranges)
                    scan_time_list.append(t.to_sec())
                elif topic == '/base_pose_ground_truth' and len(odom_time_list)<i:
                    pose_list.append(msg.pose.pose)
                    twist_list.append(msg.twist.twist)
                    odom_time_list.append(t.to_sec())
            if len(scan_time_list) == i and len(odom_time_list)==i:
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

    # post-processing and build state
    n_data = len(times)
    total_laser_samples = len(scan_list[0])
    laser_slice = int((180.0*total_laser_samples)/(270*36.0))
    laser_slice_offset = int((total_laser_samples*(270-180))/(2*270.0))
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
        state = list(laser_data_states)

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
        state += [-1.0]
        # velocities
        state += [-1.0, -1.0]
        # distances
        state += [-1.0, -1.0]
        # orientation
        state += [0.0]
        # collision distance
        state += [collision_radius_state]
        states.append(state)

    labels = []
    for i in range(n_data):
        v = action_list[i][0]
        v_label = (2*v - (v_lims[0] + v_lims[1]))/(v_lims[1] - v_lims[0])
        w = action_list[i][1]
        w_label = (2*w - (w_lims[0] + w_lims[1]))/(w_lims[1] - w_lims[0])
        labels.append([v_label, w_label])
    
    return (np.array(states, dtype=float), np.array(labels, dtype=float))