import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist, Pose, Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int8
from move_base_msgs.msg import MoveBaseActionGoal
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from utils.common import *
from utils.robot import *


class Robot(object):
    def __init__(self, args, id, n_states):
        rospy.loginfo("Initializing robot_{:d}...".format(id))
        self.id = id
        self.action_count = 0
        self.passing_count = 0
        self.overtaking_count = 0
        self.crossing_count = 0
        self.action_duration = args.action_duration
        self.map_size = args.map_size
        self.v_lims = [args.trans_vel_low, args.trans_vel_high]
        self.w_lims = [args.rot_vel_low, args.rot_vel_high]
        self.use_safety_cost = args.use_safety_cost
        self.crash_reward = args.crash_reward
        self.is_running = False
        self.goal = Pose()
        self.goal_distance_tolerance = args.goal_distance_tolerance
        self.goal_reward = args.goal_reward
        self.distance_reward_scaling = args.distance_reward_scaling
        self.goal_reached = False
        self.pose_data = Pose()
        self.position = None
        self.max_clip = args.max_clip
        self.inverse_distance_states = True
        self.twist_data = Twist()
        self.network_laser_inputs = n_states - 2

        self.use_twist_data_states = False
        self.use_orientation_state = True
        self.use_velocity_states = True
        self.use_collision_radius_state = True
        self.use_interaction_with_the_nearest_robot_states = True
        self.use_path_distance_reward = args.use_path_distance_reward
        self.use_euclidean_distance_reward = args.use_euclidean_distance_reward
        self.use_social_awareness_reward = args.use_social_awareness_reward
        self.social_awareness_reward = args.social_awareness_reward
        self.fov = args.fov

        if (self.use_twist_data_states):
            self.network_laser_inputs -= 2
        if (self.use_orientation_state):
            self.network_laser_inputs -= 1
        if (self.use_velocity_states):
            self.network_laser_inputs -= 2
        if (self.use_collision_radius_state):
            self.network_laser_inputs -= 1
        if (self.use_interaction_with_the_nearest_robot_states):
            self.network_laser_inputs -= 7

        self.use_min_laser_pooling = args.use_min_laser_pooling
        self.laser_sensor_offset = args.laser_sensor_offset
        self.motion_command = Twist()
        self.stalled = False
        self.max_action_count = args.max_action_count
        self.orientation = 0
        self.vx = 0
        self.vy = 0
        self.collision_radius  = 0.178  # because the size of the robot is 0.356*0.356*0.109
        self.nearest_robot = None

        rospy.loginfo("Examining Laser Sensor...")
        ##Added for single laser(stage)
        self.laser_data = LaserScan()
        msg = rospy.wait_for_message("/robot_{:d}/base_scan".format(self.id), LaserScan, 10)
        self.total_laser_samples = len(msg.ranges)
        self.laser_slice = int((self.fov*self.total_laser_samples)/(270*self.network_laser_inputs))
        self.laser_slice_offset = int((self.total_laser_samples*(270-self.fov))/(2*270))

        rospy.loginfo("Setting Publishers...")
        self.velocity_publisher = rospy.Publisher('/robot_{:d}/cmd_vel'.format(self.id), Twist, queue_size = 1) ##Set queue size
        rospy.loginfo("Publisher Created: " + '/robot_{:d}/cmd_vel'.format(self.id))

        self.pose_publisher = rospy.Publisher('/robot_{:d}/cmd_pos'.format(self.id), Pose2D, queue_size = 1)
        rospy.loginfo("Publisher Created: " + '/robot_{:d}/cmd_pos'.format(self.id))

        ##Added for single laser(stage)
        rospy.loginfo("Setting Subscribers...")
        rospy.Subscriber("/robot_{:d}/base_scan".format(self.id), LaserScan, self.update_robot_laser_data)
        rospy.loginfo("Subscribed to /robot_{:d}/base_scan".format(self.id))

        rospy.Subscriber("/robot_{:d}/move_base_simple/goal".format(self.id), PoseStamped, self.update_robot_goal_data )

        rospy.Subscriber("/robot_{:d}/stalled".format(self.id), Int8, self.update_robot_stall_data)
        rospy.loginfo("Subscribed to /robot_{:d}/stalled".format(self.id))
        rospy.Subscriber("/robot_{:d}/base_pose_ground_truth".format(self.id), Odometry, self.update_robot_state_data)
        rospy.loginfo("Subscribed to /robot_{:d}/base_pose_ground_truth".format(self.id))
        rospy.sleep(2)

    def reset(self):
        self.is_running = True
        self.action_count = 0
        self.passing_count = 0
        self.overtaking_count = 0
        self.crossing_count = 0
        return self.get_network_state()

    def set_robot_pose(self, position, orientation):
        robot_pose_data = Pose2D()
        robot_pose_data.x = position[0]
        robot_pose_data.y = position[1]
        robot_pose_data.theta = orientation[2]
        self.pose_publisher.publish(robot_pose_data)
        #print("Pose is set")
    
    def set_goal(self, goal):
        goal_position = goal[0]
        goal_orientation = goal[1]
        goal_orientation = quaternion_from_euler(goal_orientation[0], goal_orientation[1], goal_orientation[2])
        self.goal.position = Point(goal_position[0], goal_position[1], goal_position[2])
        self.goal.orientation = Quaternion(goal_orientation[0], goal_orientation[1], goal_orientation[2], goal_orientation[3])

        self.goal_reached = False
        self.crashed = False
    
    def set_distance_map(self, distance_map):
        self.distance_map = distance_map

    def set_obstacles_map(self, obstacles_map, resolution):
        self.obstacles_map = obstacles_map
        self.resolution = resolution
    
    def set_nearest_robot(self, robot):
        self.nearest_robot = robot

    def update_robot_laser_data(self, msg):
        self.laser_data = msg

    def update_robot_stall_data(self, msg):
        self.stalled = msg.data
    
    def update_robot_goal_data(self, msg):
        self.goal = msg.pose

    def update_robot_state_data(self, msg):
        all_data = msg
        self.pose_data = all_data.pose.pose
        self.orientation = euler_from_quaternion([self.pose_data.orientation.x, self.pose_data.orientation.y, self.pose_data.orientation.z, self.pose_data.orientation.w])[2]
        self.twist_data = all_data.twist.twist
        self.vx = self.twist_data.linear.x * np.cos(self.orientation)
        self.vy = self.twist_data.linear.x * np.sin(self.orientation)
        self.position = self.pose_data.position

    def get_pose(self):
        return self.pose_data

    def get_position(self):
        return self.pose_data.position

    def get_orientation(self):
        return self.pose_data.orientation

    def get_goal_pose(self):
        return self.goal

    def get_goal_position(self):
        return self.goal.position

    def get_linear_speed(self):
        linear_velocity = self.twist_data.linear
        return np.sqrt(linear_velocity.x**2 + linear_velocity.y**2)

    def get_angular_speed(self):
        return np.absolute(self.twist_data.angular.z)

    def get_network_state(self):
        laser_data_states = self.get_laser_data_states()
        pose_data_states = self.get_pose_data_states()
        states = laser_data_states + pose_data_states
        if (self.use_orientation_state):
            orientation_state = self.get_orientation_state()
            states += orientation_state
        if (self.use_velocity_states):
            velocity_states = self.get_velocity_states()
            states += velocity_states
        if (self.use_collision_radius_state):
            collision_radius_state = self.get_collision_radius_state()
            states += collision_radius_state
        if (self.use_interaction_with_the_nearest_robot_states):
            interaction_with_the_nearest_robot_states = self.get_interaction_with_nearest_robot_states()
            states += interaction_with_the_nearest_robot_states
        if (self.use_twist_data_states):
            twist_data_states = self.get_twist_data_states()
            states += twist_data_states
        
        return np.asarray(states)
    
    def get_laser_data_states(self):
        if(self.use_min_laser_pooling):
            laser_data_states = do_linear_transform(np.array( [min(self.laser_data.ranges[current: current+self.laser_slice]) for current in xrange(self.laser_slice_offset, len(self.laser_data.ranges) - self.laser_slice_offset, self.laser_slice)]) - self.laser_sensor_offset, self.max_clip, self.inverse_distance_states)
        else:
            laser_data_states = do_linear_transform(np.array(self.laser_data.ranges[self.laser_slice_offset+int(self.laser_slice/2):self.total_laser_samples - self.laser_slice_offset:self.laser_slice]) - self.laser_sensor_offset, self.max_clip, self.inverse_distance_states)
        return list(laser_data_states)

    def create_position_subscriber(self):
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.update_robot_state_data)
        rospy.loginfo("Topic Subscribed: /base_pose_ground_truth")

    def get_pose_data_states(self):
        position_data_state = do_linear_transform(get_distance(self.pose_data.position, self.goal.position), self.max_clip, self.inverse_distance_states)
        orientation_to_goal_data_state = get_relative_angle_to_goal(self.pose_data.position, self.pose_data.orientation, self.goal.position)/np.pi
        orientation_with_goal_data_state = get_relative_orientation_with_goal(self.pose_data.orientation, self.goal.orientation)/np.pi
        return [orientation_to_goal_data_state] + [position_data_state]

    def get_twist_data_states(self):
        trans_vel_state = (2*self.twist_data.linear.x - (self.v_lims[0] + self.v_lims[1]))/(self.v_lims[1] - self.v_lims[0])
        rot_vel_state = (2*self.twist_data.angular.z - (self.w_lims[0] + self.w_lims[1]))/(self.w_lims[1] - self.w_lims[0])
        return [trans_vel_state, rot_vel_state]

    def get_velocity_states(self):
        vx_state = (2*self.vx - (self.v_lims[0] + self.v_lims[1]))/(self.v_lims[1] - self.v_lims[0])
        vy_state = (2*self.vy - (self.v_lims[0] + self.v_lims[1]))/(self.v_lims[1] - self.v_lims[0])
        return [vx_state, vy_state]
        
    def get_orientation_state(self):
        angle = self.orientation
        if(angle/np.pi < -1):
            angle = angle + 2*np.pi
        elif(angle/np.pi > 1):
            angle =  angle - 2*np.pi
        return [angle/np.pi]

    def get_collision_radius_state(self):
        state = do_linear_transform(self.collision_radius,self.max_clip, self.inverse_distance_states)
        return [state]
    
    def get_interaction_with_nearest_robot_states(self):
        interaction_states = []
        nearest_robot_position = self.nearest_robot.position
        relative_distance_state = do_linear_transform(get_distance(self.pose_data.position, nearest_robot_position), self.max_clip, self.inverse_distance_states)
        interaction_states += [relative_distance_state]

        delta_x = nearest_robot_position.x - self.nearest_robot.position.x
        delta_y = nearest_robot_position.y - self.nearest_robot.position.y
        # transform into the local coordinates of the robot (not the nearest robot)
        transformed_delta_x = np.cos(self.orientation)*delta_x + np.sin(self.orientation)*delta_y
        transformed_delta_y = np.cos(self.orientation)*delta_y - np.sin(self.orientation)*delta_x
        transformed_delta_x_state = do_linear_transform(transformed_delta_x, self.max_clip, self.inverse_distance_states)
        transformed_delta_y_state = do_linear_transform(transformed_delta_y, self.max_clip, self.inverse_distance_states)
        interaction_states += [transformed_delta_x_state, transformed_delta_y_state]

        interaction_states += self.nearest_robot.get_velocity_states()
        interaction_states += self.nearest_robot.get_orientation_state()
        interaction_states += self.nearest_robot.get_collision_radius_state()
        return interaction_states
    
    def check_goal_reached(self):
        distance_to_goal = get_distance(self.pose_data.position, self.goal.position)
        if(distance_to_goal < self.goal_distance_tolerance):
            self.goal_reached = True
            motion_command = Twist()
            motion_command.linear.x = 0
            motion_command.angular.z = 0
            self.velocity_publisher.publish(motion_command)

    def execute_action(self, action):

        self.action_count += 1
        self.motion_command = Twist()

        self.motion_command.linear.x = action[0]
        self.motion_command.angular.z = action[1]

        action_start_time = rospy.Time.now()
        flag = False
        reward = 0
        safety_cost = 0
        crashed = False

        while(rospy.Time.now() - action_start_time < rospy.Duration(self.action_duration)):
            if(rospy.Time.now() - action_start_time < rospy.Duration(0)): #Stage simulator crashes and restarts sometimes, the flag gets activated in that case to re-run the episode.
                #print("Simulator crashed!!!!")
                flag = True
                self.is_running = False
                break
            self.velocity_publisher.publish(self.motion_command)
            if(self.stalled):
                crashed = True

        if(crashed):
            self.crashed = True
            if (self.use_safety_cost):
                safety_cost += 1
            else:
                reward += self.crash_reward

        next_state = self.get_network_state()

        if(self.action_count >= self.max_action_count):
            self.is_running = False

        next_euclidean_distance_to_goal = get_distance(self.pose_data.position, self.goal.position)

        if(self.use_euclidean_distance_reward):
            reward += -self.distance_reward_scaling*(next_euclidean_distance_to_goal - self.euclidean_distance_to_goal)

        if(self.use_path_distance_reward):
            position_x_shifted_scaled = int(np.around((self.pose_data.position.x + self.map_size)/self.resolution))
            position_y_shifted_scaled = int(np.around((self.pose_data.position.y + self.map_size)/self.resolution))
            next_path_distance_to_goal = self.distance_map[position_x_shifted_scaled, position_y_shifted_scaled]
            if(next_path_distance_to_goal != np.inf and self.path_distance_to_goal != np.inf):
                reward += -self.distance_reward_scaling*(next_path_distance_to_goal - self.path_distance_to_goal)
            self.path_distance_to_goal = next_path_distance_to_goal

        if(next_euclidean_distance_to_goal < self.goal_distance_tolerance):
            self.goal_reached = True
            self.is_running = False
            reward += self.goal_reward

        self.euclidean_distance_to_goal = next_euclidean_distance_to_goal

        # calculate rewards for social awareness
        if (self.use_social_awareness_reward):
            nearest_robot_position = self.nearest_robot.position
            relative_distance_to_nearest_robot = get_distance(self.pose_data.position, nearest_robot_position)
            relative_distance_to_goal = get_distance(self.pose_data.position, self.goal.position)
            delta_x = nearest_robot_position.x - self.nearest_robot.position.x
            delta_y = nearest_robot_position.y - self.nearest_robot.position.y
            # transform into the local coordinates of the robot (not the nearest robot)
            transformed_delta_x = np.cos(self.orientation)*delta_x + np.sin(self.orientation)*delta_y
            transformed_delta_y = np.cos(self.orientation)*delta_y - np.sin(self.orientation)*delta_x
            nearest_robot_orientation = self.nearest_robot.orientation
            delta_theta = nearest_robot_orientation - self.orientation
            if(delta_theta/np.pi < -1):
                delta_theta += 2*np.pi
            elif(delta_theta/np.pi > 1):
                delta_theta -= 2*np.pi
            nearest_robot_linear_velocity = self.nearest_robot.twist_data.linear.x
            relative_rotation_angle = np.arctan((self.nearest_robot.vx - self.vx)/(self.nearest_robot.vy - self.vy))

            # passing
            if (relative_distance_to_goal > 3) and (transformed_delta_x > 1) and (transformed_delta_x < 4) and (transformed_delta_y < 0) and (transformed_delta_y > -2) and (np.abs(delta_theta)>0.75*np.pi):
                reward += self.social_awareness_reward
                # print("passing satisfied")
                self.passing_count += 1
        

            # overtaking
            if (relative_distance_to_goal > 3) and (transformed_delta_x > 0) and (transformed_delta_x < 3) and (transformed_delta_y > 0) and (transformed_delta_y < 1) and (np.abs(delta_theta)<0.25*np.pi) and (nearest_robot_linear_velocity< self.twist_data.linear.x):
                reward += self.social_awareness_reward
                # print("overtaking satisfied")
                self.overtaking_count += 1
        

            # crossing
            if (relative_distance_to_goal > 3) and (relative_distance_to_nearest_robot <2) and (delta_theta > 0.25*np.pi) and (delta_theta < 0.75*np.pi) and (relative_rotation_angle < 0):
                reward += self.social_awareness_reward
                # print("crossing satisfied")
                self.crossing_count += 1

        if(self.use_safety_cost):
            return next_state, reward, safety_cost, flag, self.is_running

        else:
            return next_state, reward, safety_cost, flag, self.is_running # Here, safety_cost is zero


class Environment(object):
    
    def reset(self):
        self.is_running = True
        self.total_action_count = 0
        state_list = []
        if (self.n_robots==2):
            self.robot_list[0].set_nearest_robot(self.robot_list[1])
            self.robot_list[1].set_nearest_robot(self.robot_list[0])
        elif (self.n_robots==3):
            self.robot_list[0].set_nearest_robot(self.robot_list[1])
            self.robot_list[1].set_nearest_robot(self.robot_list[2])
            self.robot_list[2].set_nearest_robot(self.robot_list[0])
        elif(self.n_robots==4):
            self.robot_list[0].set_nearest_robot(self.robot_list[1])
            self.robot_list[1].set_nearest_robot(self.robot_list[0])
            self.robot_list[2].set_nearest_robot(self.robot_list[3])
            self.robot_list[3].set_nearest_robot(self.robot_list[2])
        elif(self.n_robots==6):
            self.robot_list[0].set_nearest_robot(self.robot_list[1])
            self.robot_list[1].set_nearest_robot(self.robot_list[0])
            self.robot_list[2].set_nearest_robot(self.robot_list[3])
            self.robot_list[3].set_nearest_robot(self.robot_list[2])
            self.robot_list[4].set_nearest_robot(self.robot_list[5])
            self.robot_list[5].set_nearest_robot(self.robot_list[4])
        else:
            print('need to allocate the nearest robot') # TO DO
            assert False

        for robot in self.robot_list:
            state_list.append(robot.reset())
        
        return state_list



   
    def __init__(self, args, n_states, n_robots):

        rospy.loginfo("Initializing Environment...")
        self.n_robots = n_robots
        self.robot_list = []
        self.total_action_count = 0
        self.use_safety_cost = args.use_safety_cost
        self.is_running = False
        
        rospy.init_node('ros_node')
        rospy.loginfo("Node Created")
        for i in range(self.n_robots):
            robot = Robot(args, i, n_states)
            self.robot_list.append(robot)
        rospy.sleep(5)

    def execute_action(self, action_list, last_is_running_list):
        next_state_list = []
        reward_list = []
        safety_cost_list = []
        simulator_flag_list = []
        is_running_list = []
        simulator_flag = False

        for i in range(self.n_robots):
            robot = self.robot_list[i]
            if last_is_running_list[i] == True :
                action = action_list[i]
                next_state, reward, safety_cost, flag, is_running  = robot.execute_action(action)
                next_state_list.append(next_state)
                reward_list.append(reward)
                safety_cost_list.append(safety_cost)
                simulator_flag_list.append(flag)
                is_running_list.append(is_running)
                self.total_action_count +=1
            else :
                next_state_list.append(None)
                reward_list.append(None)
                safety_cost_list.append(None)
                simulator_flag_list.append(None)
                is_running_list.append(None)

        if not(True in is_running_list):
            self.is_running = False
        if True in simulator_flag_list:
            simulator_flag = True
        

        if(self.use_safety_cost):
            return next_state_list, reward_list, safety_cost_list, simulator_flag, is_running_list

        else:
            return next_state_list, reward_list, safety_cost_list, simulator_flag, is_running_list

