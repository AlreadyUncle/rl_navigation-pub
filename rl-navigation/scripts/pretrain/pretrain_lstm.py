#!/usr/bin/env python

import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')

import os.path
import time
import tensorflow as tf
import numpy as np
import rospy
import rospkg
from datetime import datetime
from options_lstm import Options
from dataset import Dataset
from tqdm import tqdm
from read_bag_lstm import get_states_and_labels
from algo.cpo_lstm import Config

args = Options().parse()
trans_vel_limits = [args.trans_vel_low, args.trans_vel_high]
rot_vel_limits = [args.rot_vel_low, args.rot_vel_high]
action_limits = [trans_vel_limits, rot_vel_limits]
std_trans_init = args.std_trans_init
std_rot_init = args.std_rot_init
action_log_stddevs = [np.log(std_trans_init), np.log(std_rot_init)]
desired_kl = args.actor_desired_kl
summary = True


base_path = rospkg.RosPack().get_path("rl_navigation")
summary_filename = 'pretrain_lstm'

# get training data
(inputs, labels) = get_states_and_labels()
#assert False
my_dataset = Dataset(inputs[0:1800], labels[0:1800], randomize=True)


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


tf.reset_default_graph()

n_states = Config.NN_INPUT_SIZE
action_dim = 2
with tf.variable_scope("Actor"):
    with tf.name_scope("EpisodeData"):
        state = tf.placeholder(tf.float32, [None, n_states], name = 'States')
        action = tf.placeholder(tf.float32, [None, action_dim], name = 'Actions')

    with tf.name_scope('Model'):
        train_iteration = 0
        n_input = n_states
        action_dim = action_dim
        n_output = action_dim

        with tf.name_scope('StandardDeviations'):
            action_log_stddevs = tf.Variable(action_log_stddevs*tf.ones([n_output]), name = 'stddev')

        num_hidden = Config.HIDDEN_NUM
        max_length = Config.MAX_NUM_OTHER_AGENTS_OBSERVED
        num_other_agents = state[:,0]
        host_agent_vec = state[:,Config.FIRST_STATE_INDEX:Config.HOST_AGENT_STATE_SIZE+Config.FIRST_STATE_INDEX:]
        other_agent_vec = state[:,Config.HOST_AGENT_STATE_SIZE+Config.FIRST_STATE_INDEX:]
        other_agent_seq = tf.reshape(other_agent_vec, [-1, max_length, Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH])
        other_agent_seq = tf.unstack(other_agent_seq, axis = 1)
        rnn_outputs, rnn_state = tf.nn.static_rnn(tf.contrib.rnn.LSTMCell(num_hidden), other_agent_seq, dtype=tf.float32)
        rnn_output = rnn_outputs[-1]
        layer1_input = tf.concat([host_agent_vec, rnn_output],1, name='layer1_input')
        layer1 = tf.layers.dense(inputs=layer1_input, units=1000, activation=tf.nn.tanh, name = 'layer1')
        layer2 = tf.layers.dense(inputs=layer1, units=300, activation=tf.nn.tanh, name = 'layer2')
        layer3 = tf.layers.dense(inputs=layer2, units=100, activation=tf.nn.tanh, name = 'layer3')
        output_layer = tf.layers.dense(inputs=layer3, units=2, activation=tf.nn.tanh, name = 'output_layer')

vars = tf.global_variables()
saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)




#define loss function
loss = tf.reduce_mean(tf.square(action - output_layer))
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

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8)

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
    
    
    save_path = saver.save(sess, weights_path)
    print('='*30)
    print('train successfully... save_path:{}'.format(save_path) )
    
   

