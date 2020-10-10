import tensorflow as tf
import numpy as np
import cPickle as pickle
from utils.common import *

class Config:
    #########################################################################
    # GENERAL PARAMETERS
    NORMALIZE_INPUT     = True
    USE_DROPOUT         = False
    USE_REGULARIZATION  = True
    ROBOT_MODE          = True
    EVALUATE_MODE       = True

    SENSING_HORIZON     = 8.0

    MIN_POLICY = 1e-4

    DEVICE                        = '/cpu:0' # Device

    HOST_AGENT_OBSERVATION_LENGTH = 42
    OTHER_AGENT_OBSERVATION_LENGTH = 7 # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_LENGTH = 1 # num other agents
    AGENT_ID_LENGTH = 1 # id
    IS_ON_LENGTH = 1 # 0/1 binary flag

    MAX_NUM_OTHER_AGENTS_OBSERVED = 7
    ACTUAL_NUM_ROBOTS = 4 # MAX = 8 + 1 = 9
    MAX_NUM_AGENTS_IN_ENVIRONMENT = MAX_NUM_OTHER_AGENTS_OBSERVED + 1
    HIDDEN_NUM = 7

    OTHER_AGENT_FULL_OBSERVATION_LENGTH = OTHER_AGENT_OBSERVATION_LENGTH
    HOST_AGENT_STATE_SIZE = HOST_AGENT_OBSERVATION_LENGTH
    FULL_STATE_LENGTH = RNN_HELPER_LENGTH + HOST_AGENT_OBSERVATION_LENGTH + MAX_NUM_OTHER_AGENTS_OBSERVED * OTHER_AGENT_FULL_OBSERVATION_LENGTH
    FIRST_STATE_INDEX = 1

    NN_INPUT_SIZE = FULL_STATE_LENGTH

class Actor():
    def get_fisher_product_op(self):
        directional_gradients = tf.reduce_sum(self.kl_flat_gradients_op*self.vec)
        return get_flat_gradients(directional_gradients, self.trainable_variables)

    def get_fisher_product(self, vec, damping = 1e-3):
        self.feed_dict[self.vec] = vec
        return self.sess.run(self.fisher_product_op, self.feed_dict) + damping*vec

    def __init__(self, action_limits, action_log_stddevs, desired_kl, filename = None, summary = True):
        
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            self._create_graph(action_limits, action_log_stddevs, desired_kl, filename, summary)

            self.sess = tf.Session(
                graph=self.graph,
                config=tf.ConfigProto(
                    gpu_options=tf.GPUOptions(allow_growth=True, 
                    per_process_gpu_memory_fraction=0.8)))
            self.sess.run(tf.global_variables_initializer())

            vars = tf.global_variables()
            self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
            if(summary):
                self._create_summaries()
            if filename:
                self.simple_load(filename)
            

    def simple_load(self, filename=None):
        if filename is None:
            print("[network.py] Didn't define simple_load filename")
        self.saver.restore(self.sess, filename)

    def save_weights(self, path):
        self.saver.save(self.sess, path)
        print("Model saved in file: %s" % path)

    def _create_graph(self, action_limits, action_log_stddevs, desired_kl, filename = None, summary = True):
        n_states = Config.NN_INPUT_SIZE
        action_dim = 2
        with tf.variable_scope("Actor"):
            with tf.name_scope("EpisodeData"):
                self.state = tf.placeholder(tf.float32, [None, n_states], name = 'States')
                self.action = tf.placeholder(tf.float32, [None, action_dim], name = 'Actions')
                self.advantage = tf.placeholder(tf.float32, [None], name = 'Advantages')
                self.safety_advantage = tf.placeholder(tf.float32, [None], name = 'SafetyAdvantages')
                self.safety_constraint = tf.placeholder(dtype = tf.float32, shape = [], name = 'SafetyConstraint')
                self.n_episodes = tf.placeholder(dtype = tf.float32, shape = [], name = 'NumberEpisodes')

            with tf.name_scope('Model'):
                self.train_iteration = 0
                self.n_input = n_states
                self.action_dim = action_dim
                self.n_output = action_dim

                with tf.name_scope('StandardDeviations'):
                    if(action_log_stddevs is None):
                        self.action_log_stddevs = tf.Variable(tf.zeros([self.n_output]), name = 'stddev')
                    else:
                        self.action_log_stddevs = tf.Variable(action_log_stddevs*tf.ones([self.n_output]), name = 'stddev')

                num_hidden = Config.HIDDEN_NUM
                max_length = Config.MAX_NUM_OTHER_AGENTS_OBSERVED
                self.num_other_agents = self.state[:,0]
                self.host_agent_vec = self.state[:,Config.FIRST_STATE_INDEX:Config.HOST_AGENT_STATE_SIZE+Config.FIRST_STATE_INDEX:]
                self.other_agent_vec = self.state[:,Config.HOST_AGENT_STATE_SIZE+Config.FIRST_STATE_INDEX:]
                self.other_agent_seq = tf.reshape(self.other_agent_vec, [-1, max_length, Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH])
                self.other_agent_seq = tf.unstack(self.other_agent_seq, axis = 1)
                self.rnn_outputs, self.rnn_state = tf.nn.static_rnn(tf.contrib.rnn.LSTMCell(num_hidden), self.other_agent_seq, dtype=tf.float32)
                self.rnn_output = self.rnn_outputs[-1]
                self.layer1_input = tf.concat([self.host_agent_vec, self.rnn_output],1, name='layer1_input')
                self.layer1 = tf.layers.dense(inputs=self.layer1_input, units=1000, activation=tf.nn.tanh, name = 'layer1')
                self.layer2 = tf.layers.dense(inputs=self.layer1, units=300, activation=tf.nn.tanh, name = 'layer2')
                self.layer3 = tf.layers.dense(inputs=self.layer2, units=100, activation=tf.nn.tanh, name = 'layer3')
                self.output_layer = tf.layers.dense(inputs=self.layer3, units=2, activation=tf.nn.tanh, name = 'output_layer')
                
                action_list = []
                for i in range(self.action_dim):
                    a = action_limits[i][0]
                    b = action_limits[i][1]
                    action_list.append(((b-a)*self.output_layer[:,i]/2 + (a+b)/2)) #denomalization

                self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
                self.trainable_variables_shapes = [var.get_shape().as_list() for var in self.trainable_variables]

                self.action_means = tf.stack(action_list, axis = 1)
                self.action_stddevs = tf.exp(self.action_log_stddevs) + 1e-8

                self.action_dist = tf.contrib.distributions.Normal(self.action_means, self.action_stddevs)

                self.sample_action = tf.squeeze(self.action_dist.sample())

                self.each_entropy_loss = tf.reduce_sum(self.action_dist.entropy(), axis = 1)
                self.average_entropy_loss = tf.reduce_mean(self.each_entropy_loss)

                self.action_log_probs = tf.reduce_sum(self.action_dist.log_prob(self.action), axis = 1)

                self.each_experience_loss = -self.action_log_probs * self.advantage
                self.average_experience_loss = tf.reduce_mean(self.each_experience_loss)

                self.chosen_action_log_probs = tf.reduce_sum(self.action_dist.log_prob(self.action), axis = 1)
                self.old_chosen_action_log_probs = tf.stop_gradient(tf.placeholder(tf.float32, [None]))
                self.each_safety_loss = tf.exp(self.chosen_action_log_probs - self.old_chosen_action_log_probs) * self.safety_advantage
                self.average_safety_loss = tf.reduce_sum(self.each_safety_loss)/self.n_episodes

                #May want to add regularization
                self.loss = self.average_experience_loss

                ##For Diagnostics
                self.old_action_means = tf.stop_gradient(tf.placeholder(tf.float32, [None, self.action_dim]))
                self.old_action_stddevs = tf.stop_gradient(tf.placeholder(tf.float32, [self.action_dim]))
                self.old_action_dist = tf.contrib.distributions.Normal(self.old_action_means, self.old_action_stddevs)

                self.each_kl_divergence = tf.reduce_sum(tf.contrib.distributions.kl_divergence(self.action_dist, self.old_action_dist), axis = 1)
                self.average_kl_divergence = tf.reduce_mean(self.each_kl_divergence)
                self.kl_gradients = tf.gradients(self.average_kl_divergence, self.trainable_variables)

                self.desired_kl = desired_kl
                self.metrics = [self.loss, self.average_kl_divergence, self.average_safety_loss]

                self.flat_params_op = get_flat_params(self.trainable_variables)
                self.loss_flat_gradients_op = get_flat_gradients(self.loss, self.trainable_variables)
                self.kl_flat_gradients_op = get_flat_gradients(self.average_kl_divergence, self.trainable_variables)
                self.constraint_flat_gradients_op = get_flat_gradients(self.average_safety_loss, self.trainable_variables)

                self.vec = tf.placeholder(tf.float32, [None])
                self.fisher_product_op = self.get_fisher_product_op()

                self.new_params = tf.placeholder(tf.float32, [None])
                self.params_assign_op = assign_network_params_op(self.new_params, self.trainable_variables, self.trainable_variables_shapes)
            
    def _create_summaries(self):
        with tf.name_scope('Summaries'):
            self.summary = list()
            with tf.name_scope('Loss'):
                self.summary.append(tf.summary.histogram('each_experience_loss', self.each_experience_loss))
                self.summary.append(tf.summary.scalar('average_experience_loss', self.average_experience_loss))
                self.summary.append(tf.summary.histogram('each_entropy_loss', self.each_entropy_loss))
                self.summary.append(tf.summary.scalar('average_entropy_loss', self.average_entropy_loss))
                self.summary.append(tf.summary.histogram('each_kl_divergence', self.each_kl_divergence))
                self.summary.append(tf.summary.scalar('average_kl_divergence', self.average_kl_divergence))
            with tf.name_scope('Outputs'):
                self.summary.append(tf.summary.histogram('layer1', self.layer1))
                self.summary.append(tf.summary.histogram('layer2', self.layer2))
                self.summary.append(tf.summary.histogram('layer3', self.layer3))
                self.summary.append(tf.summary.histogram('output_layer', self.output_layer))
                #self.summary.append(tf.summary.histogram('output_layer_std', self.output_layer_std))
                for i in range(self.action_dim):
                    self.summary.append(tf.summary.histogram('action_means_'+str(i+1), self.action_means[:,i]))
                    self.summary.append(tf.summary.scalar('standard_deviation_'+str(i+1), self.action_stddevs[i]))
                    #self.summary.append(tf.summary.histogram('standard_deviation_'+str(i+1), self.action_stddevs[:,i]))
            self.summary_op = tf.summary.merge(self.summary)

    def update_weights(self, states, actions, advantages, safety_advantages, safety_constraint, n_episodes):
        self.train_iteration += 1
        self.feed_dict = { self.state: states, self.action: actions, self.advantage: advantages, self.safety_advantage: safety_advantages, self.n_episodes: n_episodes}
        chosen_action_log_probs =  self.sess.run(self.chosen_action_log_probs, self.feed_dict)
        self.feed_dict[self.old_chosen_action_log_probs] = chosen_action_log_probs
        g, b, old_action_means, old_action_stddevs, old_params, old_safety_loss = self.sess.run([self.loss_flat_gradients_op, self.constraint_flat_gradients_op, self.action_means, self.action_stddevs, self.flat_params_op, self.average_safety_loss], self.feed_dict)
        self.feed_dict[self.old_action_means] = old_action_means
        self.feed_dict[self.old_action_stddevs] = old_action_stddevs
        v = do_conjugate_gradient(self.get_fisher_product, g)
        #H_b = doConjugateGradient(self.getFisherProduct, b)
        approx_g = self.get_fisher_product(v)
        #b = self.getFisherProduct(H_b)
        linear_constraint_threshold = np.maximum(0, safety_constraint) + old_safety_loss
        eps = 1e-8
        delta = 2*self.desired_kl
        c = -safety_constraint
        q = np.dot(approx_g, v)

        if(np.dot(b,b) < eps):
            lam = np.sqrt(q/delta)
            nu = 0
            w = 0
            r,s,A,B = 0,0,0,0
            optim_case = 4
        else:
            norm_b = np.sqrt(np.dot(b,b))
            unit_b = b/norm_b
            w = norm_b * do_conjugate_gradient(self.get_fisher_product, unit_b)
            r = np.dot(w, approx_g)
            s = np.dot(w, self.get_fisher_product(w))
            A = q - (r**2/s)
            B = delta - (c**2/s)
            if (c < 0 and B < 0):
                optim_case = 3
            elif (c < 0 and B > 0):
                optim_case = 2
            elif(c > 0 and B > 0):
                optim_case = 1
            else:
                optim_case = 0
            lam = np.sqrt(q/delta)
            nu = 0

            if(optim_case == 2 or optim_case == 1):
                lam_mid = r / c
                L_mid = - 0.5 * (q / lam_mid + lam_mid * delta)

                lam_a = np.sqrt(A / (B + eps))
                L_a = -np.sqrt(A*B) - r*c / (s + eps)

                lam_b = np.sqrt(q / delta)
                L_b = -np.sqrt(q * delta)

                if lam_mid > 0:
                    if c < 0:
                        if lam_a > lam_mid:
                            lam_a = lam_mid
                            L_a   = L_mid
                        if lam_b < lam_mid:
                            lam_b = lam_mid
                            L_b   = L_mid
                    else:
                        if lam_a < lam_mid:
                            lam_a = lam_mid
                            L_a   = L_mid
                        if lam_b > lam_mid:
                            lam_b = lam_mid
                            L_b   = L_mid

                    if L_a >= L_b:
                        lam = lam_a
                    else:
                        lam = lam_b

                else:
                    if c < 0:
                        lam = lam_b
                    else:
                        lam = lam_a

                nu = max(0, lam * c - r) / (s + eps)

        if optim_case > 0:
            full_step = (1. / (lam + eps) ) * ( v + nu * w )
        else:
            full_step = np.sqrt(delta / (s + eps)) * w

        print('Optimization Case: ', optim_case)

        if(optim_case == 0 or optim_case == 1):
            new_params, status = do_line_search_CPO(self.get_metrics, old_params, full_step, self.desired_kl, linear_constraint_threshold, check_loss = False)
        else:
            new_params, status = do_line_search_CPO(self.get_metrics, old_params, full_step, self.desired_kl, linear_constraint_threshold)

        print('Success: ', status)

        if(status == False):
            self.sess.run(self.params_assign_op, feed_dict = {self.new_params: new_params})

        return old_action_means, old_action_stddevs

    def predict_action(self, states):
        return self.sess.run(self.sample_action, feed_dict = { self.state: states })

    def get_deterministic_action(self, states):
        return np.squeeze(self.sess.run(self.action_means, feed_dict = { self.state: states }))

    def get_summary(self, states, actions, advantages, safety_advantages, n_episodes, old_action_means, old_action_stddevs):
        feed_dict = { self.state: states, self.action: actions, self.advantage: advantages, self.safety_advantage: safety_advantages, self.n_episodes: n_episodes, self.old_action_means: old_action_means, self.old_action_stddevs: old_action_stddevs}
        return self.sess.run([self.summary_op, self.average_kl_divergence], feed_dict)

    def get_metrics(self, new_params):
        self.sess.run(self.params_assign_op, feed_dict = {self.new_params: new_params})
        return self.sess.run(self.metrics, self.feed_dict)


class Critic():
#Define the critic(value function estimator) over here

    def get_fisher_product_op(self):
        directional_gradients = tf.reduce_sum(self.kl_flat_gradients_op*self.vec)
        return get_flat_gradients(directional_gradients, self.trainable_variables)

    def get_fisher_product(self, vec, damping = 1e-3):
        self.feed_dict[self.vec] = vec
        return self.sess.run(self.fisher_product_op, self.feed_dict) + damping*vec

    def __init__(self, desired_kl, filename = None, summary = True):
        
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            self._create_graph(desired_kl, filename, summary)

            self.sess = tf.Session(
                graph=self.graph,
                config=tf.ConfigProto(
                    gpu_options=tf.GPUOptions(allow_growth=True, 
                    per_process_gpu_memory_fraction=0.8)))
            self.sess.run(tf.global_variables_initializer())

            vars = tf.global_variables()
            self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
            if(summary):
                self._createSummaries()
            
            if filename:
                self.simple_load(filename)

    def simple_load(self, filename=None):
        if filename is None:
            print("[network.py] Didn't define simple_load filename")
        self.saver.restore(self.sess, filename)

    def save_weights(self, path):
        self.saver.save(self.sess, path)
        print("Model saved in file: %s" % path)

    def _create_graph(self, desired_kl, session, filename = None, summary = True):
        n_states = Config.NN_INPUT_SIZE
        
        with tf.variable_scope("Critic"):
            with tf.name_scope("EpisodeData"):
                self.state = tf.placeholder(tf.float32, [None, n_states], name = 'States')
                self.target = tf.placeholder(tf.float32, [None], name = "Targets")

            with tf.name_scope('Model'):
                self.train_iteration = 0
                self.n_output = 1

                num_hidden = Config.HIDDEN_NUM
                max_length = Config.MAX_NUM_OTHER_AGENTS_OBSERVED
                self.num_other_agents = self.state[:,0]
                self.host_agent_vec = self.state[:,Config.FIRST_STATE_INDEX:Config.HOST_AGENT_STATE_SIZE+Config.FIRST_STATE_INDEX:]
                self.other_agent_vec = self.state[:,Config.HOST_AGENT_STATE_SIZE+Config.FIRST_STATE_INDEX:]
                self.other_agent_seq = tf.reshape(self.other_agent_vec, [-1, max_length, Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH])
                self.other_agent_seq = tf.unstack(self.other_agent_seq, axis = 1)
                self.rnn_outputs, self.rnn_state = tf.nn.static_rnn(tf.contrib.rnn.LSTMCell(num_hidden), self.other_agent_seq, dtype=tf.float32)
                self.rnn_output = self.rnn_outputs[-1]
                self.layer1_input = tf.concat([self.host_agent_vec, self.rnn_output],1, name='layer1_input')
                self.layer1 = tf.layers.dense(inputs=self.layer1_input, units=1000, activation=tf.nn.tanh, name = 'layer1')
                self.layer2 = tf.layers.dense(inputs=self.layer1, units=300, activation=tf.nn.tanh, name = 'layer2')
                self.layer3 = tf.layers.dense(inputs=self.layer2, units=100, activation=tf.nn.tanh, name = 'layer3')
                self.output_layer = tf.layers.dense(inputs=self.layer3, units=self.n_output, activation=tf.nn.tanh, name = 'output_layer')
                
                self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
                self.trainable_variables_shapes = [var.get_shape().as_list() for var in self.trainable_variables]

                self.value_estimate = tf.squeeze(self.output_layer)

                self.variance = tf.placeholder(tf.float32, [], name = 'Variance')
                self.distribution = tf.contrib.distributions.Normal(self.value_estimate, self.variance)

                self.each_experience_loss = tf.pow(self.value_estimate - self.target, 2)
                self.average_experience_loss = tf.reduce_mean(self.each_experience_loss)

                #May want to add regularization
                self.loss = self.average_experience_loss

                ##For Diagnostics
                self.old_predicted_targets = tf.stop_gradient(tf.placeholder(tf.float32, [None]))
                self.old_distribution = tf.contrib.distributions.Normal(self.old_predicted_targets, self.variance)

                self.each_kl_divergence = tf.contrib.distributions.kl_divergence(self.old_distribution, self.distribution)
                self.average_kl_divergence = tf.reduce_mean(self.each_kl_divergence)

                self.target_variance = get_variance(self.target)
                self.explained_variance_before = 1 - get_variance(self.target-self.old_predicted_targets)/(self.target_variance + 1e-10)
                self.explained_variance_after = 1 - get_variance(self.target-tf.squeeze(self.value_estimate))/(self.target_variance + 1e-10)

                self.desired_kl = desired_kl
                self.metrics = [self.loss, self.average_kl_divergence]

                self.flat_params_op = get_flat_params(self.trainable_variables)
                self.loss_flat_gradients_op = get_flat_gradients(self.loss, self.trainable_variables)
                self.kl_flat_gradients_op = get_flat_gradients(self.average_kl_divergence, self.trainable_variables)

                self.vec = tf.placeholder(tf.float32, [None])
                self.fisher_product_op = self.get_fisher_product_op()

                self.new_params = tf.placeholder(tf.float32, [None])
                self.params_assign_op = assign_network_params_op(self.new_params, self.trainable_variables, self.trainable_variables_shapes)

    def _createSummaries(self):
        with tf.name_scope('Summaries'):
            with tf.name_scope('Loss'):
                self.each_experience_loss_summary = tf.summary.histogram('each_experience_loss', self.each_experience_loss)
                self.average_experience_loss_summary = tf.summary.scalar('average_experience_loss', self.average_experience_loss)
                self.explained_variance_before_summary = tf.summary.scalar('explained_variance_before', self.explained_variance_before)
                self.explained_variance_after_summary = tf.summary.scalar('explained_variance_after', self.explained_variance_after)
            with tf.name_scope('Outputs'):
                self.hidden_layer1_summary = tf.summary.histogram('layer1', self.layer1)
                self.hidden_layer2_summary = tf.summary.histogram('layer2', self.layer2)
                self.hidden_layer3_summary = tf.summary.histogram('layer3', self.layer3)
                self.output_layer_summary = tf.summary.histogram('value_estimate', self.value_estimate)
            self.summary_op = tf.summary.merge([self.average_experience_loss_summary, self.each_experience_loss_summary, self.explained_variance_before_summary, self.explained_variance_after_summary, self.hidden_layer1_summary, self.hidden_layer2_summary, self.hidden_layer3_summary, self.output_layer_summary])

    def update_weights(self, states, targets):
        self.train_iteration += 1
        self.feed_dict = { self.state: states, self.target: targets  }
        loss, loss_gradients, old_predicted_targets, old_params = self.sess.run([self.loss, self.loss_flat_gradients_op, self.value_estimate, self.flat_params_op], self.feed_dict)
        self.feed_dict[self.old_predicted_targets] = old_predicted_targets
        self.feed_dict[self.variance] = loss
        step_direction = do_conjugate_gradient(self.get_fisher_product, loss_gradients)
        fisher_loss_product = self.get_fisher_product(step_direction)
        full_step_size = np.sqrt((2*self.desired_kl)/(np.dot(step_direction, fisher_loss_product)))
        full_step = full_step_size*step_direction
        #new_params = old_params + full_step
        new_params, status = do_line_search(self.get_metrics, old_params, full_step, self.desired_kl)
        if(status == False):
            self.sess.run(self.params_assign_op, feed_dict = {self.new_params: new_params})
        return old_predicted_targets


    def predict_value(self, state):
        return self.sess.run(self.value_estimate, feed_dict = { self.state: state })

    def get_summary(self, states, targets, old_predicted_targets):
        feed_dict = { self.state: states, self.target: targets, self.old_predicted_targets: old_predicted_targets  }
        return self.sess.run(self.summary_op, feed_dict)

    def get_metrics(self, new_params):
        self.sess.run(self.params_assign_op, feed_dict = {self.new_params: new_params})
        return self.sess.run(self.metrics, self.feed_dict)
