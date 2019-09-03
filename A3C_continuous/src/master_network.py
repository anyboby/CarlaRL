import tensorflow as tf
import numpy as np
import constants as Constants
import network_shares as Netshare
class MasterNetwork(object):
    def __init__(self, scope, globalAC=None):

        if scope == Constants.GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                #print ("scope: " + str(scope))
                self.s = tf.placeholder(tf.float32, Netshare.DIM_S, 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                #print ("scope: " + str(scope))
                self.s = tf.placeholder(tf.float32, Netshare.DIM_S, 'S')
                self.a_his = tf.placeholder(tf.float32, Netshare.DIM_A, 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                self.summaries = []

                #mainly for tensorboard recordings
                l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))
                
                summary_v = tf.summary.scalar("v", l2_norm(self.v))
                self.summaries.append(summary_v)
                summary_adv = tf.summary.scalar("adv", l2_norm(td))
                self.summaries.append(summary_adv)
                summary_c_loss = tf.summary.scalar("c_loss", l2_norm(self.c_loss))
                self.summaries.append(summary_c_loss)
                summary_r = tf.summary.scalar("v_target", l2_norm(self.v_target))
                self.summaries.append(summary_r)


                #choose actions from normal dist
                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * Netshare.BOUND_A[1], sigma + 1e-4
                normal_dist = tf.distributions.Normal(mu, sigma)


                summary_mu = tf.summary.scalar("mu", l2_norm(mu))
                self.summaries.append(summary_mu)
                summary_sigma = tf.summary.scalar("sigma", l2_norm(sigma))
                self.summaries.append(summary_sigma)
                ##### Manual Biases #####
                #if Constants.manual_dims:
                #    tf.add(mu, [0.0, 1 ,0.0])
                #########################

                
                #print("mu shape: " + str(mu.shape))
                #print("sigma shape: " + str(sigma.shape))
                #print("normal shape: " + str(normal_dist))

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = Constants.ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                summary_a_loss = tf.summary.scalar("a_loss", l2_norm(self.a_loss))
                self.summaries.append(summary_a_loss)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), Netshare.BOUND_A[0], Netshare.BOUND_A[1])
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
                #print("sigma shape: " + str(self.A))

                for a_grad, c_grad in zip(self.a_grads, self.c_grads):
                    summary_grads_a = tf.summary.scalar("grads_a", l2_norm(a_grad))
                    self.summaries.append(summary_grads_a)
                    summary_grads_c = tf.summary.scalar("grads_c", l2_norm(c_grad))
                    self.summaries.append(summary_grads_c)

                # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
                self.merged = tf.summary.merge(self.summaries)
                self.train_writer = tf.summary.FileWriter(Constants.LOG_DIR2)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = Netshare.OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = Netshare.OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            if Constants.GAME == "CarRacing-v0":
                ######## CarRacing Actor ######### 
                l_conv1 = tf.layers.conv2d(self.s, 32, (6,6), strides=(2,2), activation=tf.nn.elu, kernel_initializer=w_init, name="conv1")
                l_conv2 = tf.layers.conv2d(l_conv1, 16, (4,4), strides=(2,2), activation=tf.nn.elu, kernel_initializer=w_init, name="conv2")
                l_conv3 = tf.layers.conv2d(l_conv2, 16, (3,3), strides=(2,2), activation=tf.nn.elu, kernel_initializer=w_init, name="conv3")

                l_fl = tf.layers.flatten(l_conv3, name="fl_a")
                l_d = tf.layers.dense(l_fl, 256, tf.nn.elu, kernel_initializer=w_init, name="l_d")
                #l_a = tf.layers.dense(l_d, 15, tf.nn.relu, kernel_initializer=w_init, name='la')
                # N_A[0] has None placeholder for samples, so the real number of actions if in N_A[1]
                mu = tf.layers.dense(l_d, Netshare.DIM_A[1], tf.nn.tanh, kernel_initializer=w_init, name='mu')
                sigma = tf.layers.dense(l_d, Netshare.DIM_A[1], tf.nn.softplus, kernel_initializer=w_init, name='sigma')
                ####################################
            elif Constants.GAME == "Pendulum-v0":
                ######## Pendulum Actor #########  lr ca. a: 0.0001
                l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
                mu = tf.layers.dense(l_a, Netshare.DIM_A[1], tf.nn.tanh, kernel_initializer=w_init, name='mu')
                sigma = tf.layers.dense(l_a, Netshare.DIM_A[1], tf.nn.softplus, kernel_initializer=w_init, name='sigma')
                ####################################
            elif Constants.GAME == "MountainCarContinuous-v0":
                ######## MountainCar Actor #########  lr ca. a:0.003
                l_a = tf.layers.dense(self.s, 40, tf.nn.relu6, kernel_initializer=w_init, name='la')
                #l_a2 = tf.layers.dense(l_a, 1, tf.nn.relu6, kernel_initializer=w_init, name='la2')
                mu = tf.layers.dense(l_a, Netshare.DIM_A[1], tf.nn.tanh, kernel_initializer=w_init, name='mu')
                sigma = tf.layers.dense(l_a, Netshare.DIM_A[1], tf.nn.softplus, kernel_initializer=w_init, name='sigma')
                ####################################
            
        with tf.variable_scope('critic'):
            if Constants.GAME == "CarRacing-v0":
                ######## CarRacing Critic ########
                #l_c = tf.layers.dense(l_d, 5, tf.nn.relu6, kernel_initializer=w_init, name='lc')
                v = tf.layers.dense(l_d, 1, kernel_initializer=w_init, name='v')  # state value
                ##################################
            elif Constants.GAME == "Pendulum-v0":
                ######## Pendulum Critic ######## lr ca. c: 0.001
                l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
                v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
                ##################################
            elif Constants.GAME == "MountainCarContinuous-v0":
                ######## MountainCar Critic ######### lr ca. c:0.03
                #l_c = tf.layers.dense(l_a, 5, tf.nn.relu6, kernel_initializer=w_init, name='lc')
                v = tf.layers.dense(l_a, 1, kernel_initializer=w_init, name='v')  # state value
                ####################################

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict, write_summaries=False):  # run by a local
        # local grads applies to global net
        if write_summaries:
            _,_,summary = Netshare.SESS.run([self.update_a_op, self.update_c_op, self.merged], feed_dict)
            self.train_writer.add_summary(summary, Constants.GLOBAL_EP)

        else:
            Netshare.SESS.run([self.update_a_op, self.update_c_op], feed_dict)  

    def pull_global(self):  # run by a local
        Netshare.SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        result = Netshare.SESS.run(self.A, {self.s: s})
        #print (result)
        return result