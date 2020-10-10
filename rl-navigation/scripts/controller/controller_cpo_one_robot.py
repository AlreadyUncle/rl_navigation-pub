#!/usr/bin/env python

import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')

import cPickle as pickle

import os.path

import time

import tensorflow as tf
import numpy as np
import rospy
import rospkg

from datetime import datetime

from environments.base_environment import Environment
from utils.memory import ExperienceBuffer
from algo.cpo import Actor, Critic, SafetyBaseline
from utils.common import *
from utils.map import *
from utils.robot import *
from options import Options

from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist, Pose, Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from move_base_msgs.msg import MoveBaseActionGoal



args = Options().parse()

tf.reset_default_graph()
sess = tf.Session()


actor_filename = "/home/shiqing/rl_navigation_ws/src/rl-navigation/logs/2019-12-14_18-04-56_asl_architecture/weights/weights_actor1000.p"
arch = 'social'

if(arch == 'asl'):
    n_states = 38
elif(arch == 'tai'):
    n_states = 12
elif(arch == 'social'):
    n_states = 49

n_robots = 2
action_dim = 2 #Translational and rotational velocity
trans_vel_limits = [args.trans_vel_low, args.trans_vel_high]
rot_vel_limits = [args.rot_vel_low, args.rot_vel_high]
std_trans_init = args.std_trans_init
std_rot_init = args.std_rot_init
actor_desired_kl = args.actor_desired_kl
critic_desired_kl = args.critic_desired_kl
safety_baseline_desired_kl = args.safety_baseline_desired_kl




class CPO_Controller(object):
    def __init__(self):
        self.args = Options().parse()

        #self.action_count = 0
        #self.action_duration = self.args.action_duration
        #self.map_size = self.args.map_size
        self.v_lims = [self.args.trans_vel_low, self.args.trans_vel_high]
        self.w_lims = [self.args.rot_vel_low, self.args.rot_vel_high]
        #self.use_safety_cost = self.args.use_safety_cost
        #self.crash_reward = self.args.crash_reward
        #self.goal_distance_tolerance = self.args.goal_distance_tolerance

        #self.distance_reward_scaling = self.args.distance_reward_scaling
        #self.goal_reached = False
        self.max_clip = args.max_clip
        self.inverse_distance_states = True
        self.network_laser_inputs = n_states - 2
        self.use_twist_data_states = False

        self.fov = self.args.fov
        if(self.use_twist_data_states):
            self.network_laser_inputs -= 2
        self.use_min_laser_pooling = self.args.use_min_laser_pooling
        self.laser_sensor_offset = self.args.laser_sensor_offset
        #self.stalled = False


        ##Added for single laser(stage)
        self.laser_data = LaserScan()
        # msg = rospy.wait_for_message("/scan", LaserScan, 10)
        # self.total_laser_samples = len(msg.ranges)
        self.total_laser_samples = 1080
        self.laser_slice = int((self.fov*self.total_laser_samples)/(270*self.network_laser_inputs))
        self.laser_slice_offset = int((self.total_laser_samples*(270-self.fov))/(2*270))

        rospy.init_node('CPO_controller', anonymous=True)
        self.sub_ls = rospy.Subscriber("/scan", LaserScan, self.update_robot_laser_data)
        self.sub_pose = rospy.Subscriber("/base_pose_ground_truth", Odometry, self.update_robot_state_data)
        self.sub_goal = rospy.Subscriber("/move_base/goal",MoveBaseActionGoal, self.update_goal_data )
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)

        self.scan = LaserScan()
        self.pose_data = Pose()
        self.goal = Pose()
        self.twist_data = Twist()
        self.motion_command = Twist()

        self.policy_estimator = Actor(n_states, action_dim, [trans_vel_limits, rot_vel_limits],
                                       [np.log(std_trans_init), np.log(std_rot_init)], actor_desired_kl, sess, arch, actor_filename)
        print('policy_estimator created')
        print('CPO_controller created')


    def update_robot_laser_data(self, msg):
        # print(len(msg.ranges))
        self.laser_data = msg

    def update_robot_state_data(self, msg):
        all_data = msg
        self.pose_data = all_data.pose.pose
        self.twist_data = all_data.twist.twist
    
    def update_goal_data(self, msg):
        self.goal = msg.goal.target_pose.pose

    def get_pose_data_states(self):
        position_data_state = do_linear_transform(get_distance(self.pose_data.position, self.goal.position), self.max_clip, self.inverse_distance_states)
        orientation_to_goal_data_state = get_relative_angle_to_goal(self.pose_data.position, self.pose_data.orientation, self.goal.position)/np.pi
        orientation_with_goal_data_state = get_relative_orientation_with_goal(self.pose_data.orientation, self.goal.orientation)/np.pi
        return [orientation_to_goal_data_state] + [position_data_state]
    
    def get_laser_data_states(self):
        if(self.use_min_laser_pooling):
            laser_data_states = do_linear_transform(np.array( [min(self.laser_data.ranges[current: current+self.laser_slice]) for current in xrange(self.laser_slice_offset, len(self.laser_data.ranges) - self.laser_slice_offset, self.laser_slice)]) - self.laser_sensor_offset, self.max_clip, self.inverse_distance_states)
        else:
            laser_data_states = do_linear_transform(np.array(self.laser_data.ranges[self.laser_slice_offset+int(self.laser_slice/2):self.total_laser_samples - self.laser_slice_offset:self.laser_slice]) - self.laser_sensor_offset, self.max_clip, self.inverse_distance_states)
        return list(laser_data_states)

    def get_twist_data_states(self):
        trans_vel_state = (2*self.twist_data.linear.x - (self.v_lims[0] + self.v_lims[1]))/(self.v_lims[1] - self.v_lims[0])
        rot_vel_state = (2*self.twist_data.angular.z - (self.w_lims[0] + self.w_lims[1]))/(self.w_lims[1] - self.w_lims[0])
        return [trans_vel_state, rot_vel_state]

    def get_network_state(self):
        laser_data_states = self.get_laser_data_states()
        pose_data_states = self.get_pose_data_states()
        #print(pose_data_states)
        states = laser_data_states + pose_data_states
        if(self.use_twist_data_states):
            twist_data_states = self.get_twist_data_states()
            states += twist_data_states
        return np.asarray(states)

    def execute_action(self):
        state = self.get_network_state()
        #print(np.reshape(state, (-1, n_states)))
        action = self.policy_estimator.predict_action(np.reshape(state, (-1, n_states)))
        self.motion_command = Twist()
        self.motion_command.linear.x = max(action[0],0)
        self.motion_command.angular.z = action[1]
        print([self.motion_command.linear.x, self.motion_command.angular.z])
        self.velocity_publisher.publish(self.motion_command)

    def vel_cmder(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            # Use the pre-trained RL to predict the output velocity
            if len(self.laser_data.ranges) > 0:
               self.execute_action()
            else:
                print('No laserscan received')

            rate.sleep()


if __name__ == '__main__':
    try:
        CPO_controller = CPO_Controller()
        #assert False
        CPO_controller.vel_cmder()        
    except rospy.ROSInterruptException:
        pass