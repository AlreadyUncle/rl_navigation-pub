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

from controller_environment import Environment
from controller_environment import Robot

# from utils.memory import ExperienceBuffer
from algo.cpo import Actor, Critic, SafetyBaseline
# from utils.common import *
# from utils.map import *
# from utils.robot import *
from options import Options

from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist, Pose, Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from move_base_msgs.msg import MoveBaseActionGoal



args = Options().parse()

tf.reset_default_graph()
sess = tf.Session()


actor_filename = "/home/shiqing/rl_navigation_ws/src/rl-navigation/logs/2020-05-11_02-09-49_cpo_pretrained_with_poc_fixed/weights/weights_actor1000.p"
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
        self.environment = Environment(args, n_states, n_robots)
        _ = self.environment.reset()
        self.robots_list = self.environment.robot_list
        self.state_one_robot = []
        self.goals = []
        self.stall_timestamps = []
        self.start_timestamps = []
        self.one_loop_time = []
        self.till_stall_time = []

        self.v_lims = [self.args.trans_vel_low, self.args.trans_vel_high]
        self.w_lims = [self.args.rot_vel_low, self.args.rot_vel_high]
        
        self.policy_estimator = Actor(n_states, action_dim, [trans_vel_limits, rot_vel_limits],
                                       [np.log(std_trans_init), np.log(std_rot_init)], actor_desired_kl, sess, arch, actor_filename)
        print('Policy estimator created')
        print('CPO controller created')

    def execute_action_one_robot(self):
        self.state_one_robot = self.robots_list[0].get_network_state()
        #print(np.reshape(state, (-1, n_states)))
        action = self.policy_estimator.predict_action(np.reshape(self.state_one_robot, (-1, n_states)))
        self.motion_command = Twist()
        self.motion_command.linear.x = max(action[0],0)
        self.motion_command.angular.z = action[1]
        print([self.motion_command.linear.x, self.motion_command.angular.z])
        self.robots_list[0].velocity_publisher.publish(self.motion_command)

    def vel_cmd_one_robot(self, n_goals):
        i = 0
        robot = robots_list[0]
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if robot.goal_reached:
                i = i+1
                if i == n_goals: 
                    self.one_loop_time.append(rospy.Time.now().to_sec() - self.start_timestamps[-1])
                    break
                robot.set_goal(self.goals[i])
                print("Next goal set! ")
            if robot.stalled :
                t = rospy.Time.now().to_sec()
                self.stall_timestamps.append(t)
                self.till_stall_time.append(t - self.start_timestamps[-1])
                robot.stalled = False
                print("Stalled time: {:.2f}".format(t))
                break
            if rospy.Time.now().to_sec() - self.start_timestamps[-1]> 300:
                break
            self.execute_action(robot)        
            rate.sleep()

    def execute_action(self, robot):
        if not(robot.goal_reached):
            state = robot.get_network_state()
            action = self.policy_estimator.predict_action(np.reshape(state, (-1, n_states)))
            motion_command = Twist()
            # for overtaking
            #if robot.id == 1:
            #    motion_command.linear.x = 0.3* max(action[0],0)
            #    motion_command.angular.z = 0
            #else:
            #    motion_command.linear.x = max(action[0],0)
            #    motion_command.angular.z = action[1]
            motion_command.linear.x = max(action[0],0)
            motion_command.angular.z = action[1]
            #print([motion_command.linear.x, motion_command.angular.z])
            robot.velocity_publisher.publish(motion_command)
            robot.check_goal_reached()

    def vel_cmd(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            # Use the pre-trained RL to predict the output velocity
            for robot in self.robots_list:
                self.execute_action(robot)        
            rate.sleep()


if __name__ == '__main__':
    try:
        position_list = []
        goal_list = []
        # create controller
        controller = CPO_Controller()
        robots_list = controller.robots_list
        ######### for exps in empty simulation #########
        """
        position_list.append([[1.0, 2.0 ,0.0 ], [0 ,0 ,-0.25*np.pi]])
        position_list.append([[9.0, 2.0 ,0.0 ], [0 ,0 ,0.75*np.pi]])
        position_list.append([[1.0, 8.0 ,0.0 ], [0 ,0 ,-0.25*np.pi]])
        position_list.append([[9.0, 8.0 ,0.0 ], [0 ,0 ,-np.pi]])

        goal_list.append([[9.0, 2.0 ,0.0 ], [0 ,0 ,0]])
        goal_list.append([[1.0, 2.0 ,0.0 ], [0 ,0 ,0.75*np.pi]])
        goal_list.append([[9.0, 8.0 ,0.0 ], [0 ,0 ,0]])
        goal_list.append([[1.0, 8.0 ,0.0 ], [0 ,0 ,-np.pi]])
        """

        ######### for exps in recorded map #########
        
        position_list.append([[1.0, -13.0 ,0.0 ], [0 ,0 ,0.5*np.pi]])
        position_list.append([[1.0, -14.0 ,0.0 ], [0 ,0 ,0.5*np.pi]])  
        goal_list.append([[-3, -2 ,0.0 ], [0 ,0 ,0.5*np.pi]])      
        goal_list.append([[1.0, -13.0 ,0.0 ], [0 ,0 ,-0.5*np.pi]])
        n_goals = 3
        controller.goals.append([[-3, -2 ,0.0 ], [0 ,0 ,0.5*np.pi]])
        controller.goals.append([[4, 1 ,0.0 ], [0 ,0 ,0.5*np.pi]])
        controller.goals.append([[-1, 12 ,0.0 ], [0 ,0 ,0.5*np.pi]])
        

        
        for i in range(n_robots):
            robots_list[i].set_robot_pose(position_list[i][0],position_list[i][1])
            robots_list[i].set_goal(goal_list[i])


        #assert False
        content = raw_input("Press enter to start")
        #controller.vel_cmd() 
        
        for n in range(20): 
            print("Already: {:d} loops".format(n))
            t = rospy.Time.now().to_sec()
            controller.start_timestamps.append(t)
            print("Start time: {:.2f}".format(t)) 
            controller.vel_cmd_one_robot(n_goals)
            for i in range(n_robots):
                robots_list[i].set_robot_pose(position_list[i][0],position_list[i][1])
                robots_list[i].set_goal(goal_list[i])    
        print("Loop times:")
        print(controller.one_loop_time)
        print("Till stall times:")
        print(controller.till_stall_time)
    except rospy.ROSInterruptException:
        pass