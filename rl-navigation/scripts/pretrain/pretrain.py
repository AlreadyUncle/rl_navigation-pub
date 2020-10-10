from read_bag import get_states_and_labels
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
from environments.base_environment import Robot
from utils.memory import ExperienceBuffer
from algo.cpo import Actor, Critic, SafetyBaseline
from utils.common import *
from utils.map import *
from options import Options
import matplotlib.pyplot as plt
from dataset import Dataset
from tqdm import tqdm
import cPickle as pickle

args = Options().parse()
trans_vel_limits = [args.trans_vel_low, args.trans_vel_high]
rot_vel_limits = [args.rot_vel_low, args.rot_vel_high]
action_limits = [trans_vel_limits, rot_vel_limits]
std_trans_init = args.std_trans_init
std_rot_init = args.std_rot_init
action_log_stddevs = [np.log(std_trans_init), np.log(std_rot_init)]
desired_kl = args.actor_desired_kl
arch = 'social'
summary = True


tf.reset_default_graph()

base_path = rospkg.RosPack().get_path("rl_navigation")
summary_filename = 'pretrain2'

# Set up folder structure
date_str = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
storage_path = os.path.join(base_path, "initial_weights", date_str + "_" + summary_filename)
weights_path = os.path.join(storage_path, "weights")
tensorboard_path = os.path.join(storage_path, "tensorboard")
diagnostics_path = os.path.join(storage_path, "diagnostics")
os.mkdir(storage_path)
os.mkdir(weights_path)
os.mkdir(tensorboard_path)
os.mkdir(diagnostics_path)

# get training data
(inputs, labels) = get_states_and_labels()

my_dataset = Dataset(inputs[0:2000], labels[0:2000], randomize=True )

n_states = 49
action_dim = 2

with tf.variable_scope("Actor"):

    with tf.name_scope("EpisodeData"):
        state = tf.placeholder(tf.float32, [None, n_states], name = 'States')
        action = tf.placeholder(tf.float32, [None, action_dim], name = 'Actions')
    
    with tf.name_scope('Model'):
        n_input = n_states

        n_hidden1 = 1000
        n_hidden2 = 300
        n_hidden3 = 100

        action_dim = action_dim
        n_output = action_dim

        #Initialize Weights
        with tf.name_scope('Weights'):
            weights = {'h1' : tf.Variable(tf.random_normal([n_input, n_hidden1], stddev = tf.sqrt(2./n_input)), name = 'h1'), 'h2' : tf.Variable(tf.random_normal([n_hidden1, n_hidden2], stddev = tf.sqrt(2./n_hidden1)), name = 'h2'), 'h3' : tf.Variable(tf.random_normal([n_hidden2, n_hidden3], stddev = tf.sqrt(2./n_hidden2)), name = 'h3'), 'out' : tf.Variable(tf.random_normal([n_hidden3, n_output], stddev = tf.sqrt(2./n_hidden3)), name = 'out')}
        with tf.name_scope('Biases'):
            biases = {'b1' : tf.Variable(tf.zeros([n_hidden1]), name = 'b1'), 'b2' : tf.Variable(tf.zeros([n_hidden2]), name = 'b2'), 'b3' : tf.Variable(tf.zeros([n_hidden3]), name = 'b3'), 'out' : tf.Variable(tf.zeros([n_output]), name = 'out')}


        with tf.name_scope('StandardDeviations'):
            if(action_log_stddevs is None):
                action_log_stddevs = tf.Variable(tf.zeros([n_output]), name = 'stddev')
            else:
                action_log_stddevs = tf.Variable(action_log_stddevs*tf.ones([n_output]), name = 'stddev')

        hidden_layer1 = tf.nn.tanh(tf.add(tf.matmul(state,  weights['h1']), biases['b1']))
        hidden_layer2 = tf.nn.tanh(tf.add(tf.matmul(hidden_layer1, weights['h2']), biases['b2']))
        hidden_layer3 = tf.nn.tanh(tf.add(tf.matmul(hidden_layer2, weights['h3']), biases['b3']))
        output_layer = tf.nn.tanh(tf.add(tf.matmul(hidden_layer3, weights['out']), biases['out']))
        

#define loss function
loss = tf.reduce_mean( tf.square(action - output_layer) )
total_epoches = 5000000
# total_epoches = 100000

# total_epoches = 200

learning_rate = 1e-6
batch_size = 100

#define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# writer = tf.summary.FileWriter('./log', tf.get_default_graph())
# writer.close()

tf.summary.scalar('training_loss', loss)
merged = tf.summary.merge_all()
loss_observations = []

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.initialize_all_variables())
    tb_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)

    for epoch in tqdm(range(total_epoches)):
        (inputs, labels) = my_dataset.get_next_batch(batch_size)
        
        _, merged_val, training_loss_val, output_val = sess.run([optimizer, merged, loss, output_layer], feed_dict = {
            state : inputs ,
            action : labels
        })
        loss_observations.append(training_loss_val)

        if epoch % int(total_epoches / 100) == 0:
            print('epoch: {}'.format(epoch))
            print( 'real action: {}'.format(labels[0:20,:]) )
            print( 'predicted action: {}'.format(output_val[0:20,:]) )
            print( 'average loss: {}'.format(training_loss_val) )
            tb_writer.add_summary(merged_val, epoch)
    
    pickle.dump(sess.run([weights, biases]), open(os.path.join(weights_path, "pretrained_weights"+'.p'), 'wb'))


    saver = tf.train.Saver()
    save_path = saver.save(sess, weights_path )
    print('='*30)
    print('train successfully... save_path:{}'.format(save_path) )
    
   

