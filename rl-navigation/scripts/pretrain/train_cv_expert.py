# This file is to train robot following only by images
# in a rather complex enviroment
    # Some of details refer to
    # End-to-End Deep Learning for Robotic Following
    # John M. Pierre


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm
#from tensorflow.contrib.layers.python.layers import batch_norm
from my_data_set import ImageDataset
from os import listdir
from os.path import isfile, join



import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print('tensorflow version is:{}'.format(tf.__version__))
# assert False
# assert False
#define global stuffs

ph_x_image_ = tf.placeholder( tf.float32, [None, 48, 64, 3] , name='ph_x_image')
ph_control_ = tf.placeholder(tf.float32, [None, 2])      #Control is a 2-dimensional vector
ph_dropout_ = tf.placeholder(tf.float32)

base_paths = ['/home/shiqing/my_baselines/baselines/gail/cv/human_follower_bags/']

#base_paths = ['/home/shiqing/my_baselines/baselines/gail/cv/human_follow2/']


bag_paths = []


for base_path in base_paths:
    bag_paths = bag_paths + [base_path+f for f in listdir(base_path) if isfile(join(base_path, f))]


train_fraction = 0.7
randomize = True
im_Dataset = ImageDataset(bag_paths, train_fraction, randomize)
(im , lab) = im_Dataset.get_next_batch(50)

# print(np.shape(lab))
# assert False

print ("data read finish")

# Reduce some unused params, make a more compact function
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    return tf.Variable(tf.zeros(shape))


# 1st Convolutional Layer with stride
W_conv1 = weight_variable([5,5,3,29])
b_conv1 = bias_variable([29])
h_conv1 = tf.nn.relu(conv2d(ph_x_image_, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# print('h_pool1:{}'.format(h_pool1) )
# assert False

#

# 2nd Convolutional Layer with stride
W_conv2 = weight_variable([3,3,29,48])
b_conv2 = bias_variable([48])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# h_pool2 12*16*48

# 3rd Convolutional Layer with no stride
W_conv3 = weight_variable([3,3,48,64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)



# 1st Full-connected Layer + Flatten
W_fc1 = weight_variable([8*12*64,1000])
b_fc1 = bias_variable([1000])
h_pool2_flat = tf.reshape(h_conv3, [-1,8*12*64])
fc_out1 = tf.contrib.layers.layer_norm((tf.matmul(h_pool2_flat, W_fc1) + b_fc1))
fc_out1 = tf.nn.relu(fc_out1)

# 2nd Output layers
W_fc2 = weight_variable( [1000, 100] )
b_fc2 = bias_variable([100])
fc_out2 = (tf.matmul(fc_out1, W_fc2) + b_fc2)
fc_out2 = tf.nn.relu(fc_out2)



# 3rd Output layers for vel
W1_fc3 = weight_variable( [100, 50] )
b1_fc3 = bias_variable([50])
fc1_out3 = tf.contrib.layers.layer_norm(tf.matmul(fc_out2, W1_fc3) + b1_fc3)
fc1_out3 = tf.nn.relu(fc1_out3)  


# 4th Output layers for vel
W1_fc4 = weight_variable( [50, 10] )
b1_fc4 = bias_variable([10])
fc1_out4 = tf.nn.relu(tf.matmul(fc1_out3, W1_fc4) + b1_fc4)

# 5th Output layers for vel
W1_fc5 = weight_variable( [10, 1] )
b1_fc5 = bias_variable([1])
control_hat_1 = tf.matmul(fc1_out4, W1_fc5) + b1_fc5


# 3rd Output layers for omega
W2_fc3 = weight_variable( [100, 50] )
b2_fc3 = bias_variable([50])
fc2_out3 = tf.contrib.layers.layer_norm(tf.matmul(fc_out2, W2_fc3) + b2_fc3)
fc2_out3 = tf.nn.relu(fc2_out3)  


# 4th Output layers for omega
W2_fc4 = weight_variable( [50, 10] )
b2_fc4 = bias_variable([10])
fc2_out4 = tf.nn.relu(tf.matmul(fc2_out3, W2_fc4) + b2_fc4)

# 5th Output layers for omega
W2_fc5 = weight_variable( [10, 1] )
b2_fc5 = bias_variable([1])
control_hat_2 = tf.matmul(fc2_out4, W2_fc5) + b2_fc5

# control_hat = [control_hat_1, control_hat_2]
control_hat = tf.concat((control_hat_1, control_hat_2), axis=1)
tf.identity(control_hat, name='control_predicted')


# print(control_hat)
# assert False
# print()


#define loss function
loss = tf.reduce_mean( tf.square(control_hat - ph_control_) )


total_epoches = 200000
# total_epoches = 100000

# total_epoches = 200

learning_rate = 1e-6
batch_size = 128

#define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

writer = tf.summary.FileWriter('./log', tf.get_default_graph())
writer.close()


# assert False
# Validate our model
# with tf.device('/gpu:0'):

tf.summary.scalar('training_loss', loss)
merged = tf.summary.merge_all()

loss_observations = []



def early_stop(loss_observations, min_iters=10000, min_loss=1e-3, horizon = 50, relative_loss = 1e-4):
    len_obs = len(loss_observations)
    
    if len_obs < min_iters:
        return False
    
    if min(loss_observations) > min_loss:
        return False

    sub_arr = loss_observations[max([1,len_obs-horizon]):len_obs-1]
    
    if max(sub_arr) - min(sub_arr) > relative_loss:
        return False
    
    return True

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.initialize_all_variables())
    tb_writer = tf.summary.FileWriter("log/", sess.graph)

    for epoch in tqdm(range(total_epoches)):
        (ims, labels) = im_Dataset.get_next_batch(batch_size)
        
        _, merged_val, training_loss_val, control_hat_val, fc_out2_val = sess.run([optimizer, merged, loss, control_hat, fc_out2], feed_dict = {
            ph_x_image_ : ims ,
            ph_control_ : labels
            # ph_dropout_ : 0.5
        })
        loss_observations.append(training_loss_val)

        if epoch % int(total_epoches / 100) == 0:
            print('epoch:{}'.format(epoch))
            print('fc1_out3_val:{}'.format(fc_out2_val))
            print( 'control_real_:{}'.format(labels[0:20,:]) )
            print( 'control_hat_:{}'.format(control_hat_val[0:20,:]) )
            print( 'average loss:{}'.format(training_loss_val) )
            tb_writer.add_summary(merged_val, epoch)

        # if eary_stop(loss_observations):
        #     break
    
    saver = tf.train.Saver()
    save_path = saver.save(sess, "cv_param/param")
    print('='*30)
    print('train successfully... save_path:{}'.format(save_path) )









