from actor import Actor
from critic import Critic
from memory_buffer import MemoryBuffer
import numpy as np
import time
import cv2
from rl.random import OrnsteinUhlenbeckProcess
import tensorflow as tf


class DDPG:
    """ Deep Deterministic Policy Gradient (DDPG) Helper Class
    """

    def __init__(self, state_size, action_size, batch_no):
        """ Initialization
        """
        # Environment and parameters
        self.state_size = (batch_no,) + state_size
        self.action_size = action_size
        self.gamma = 0.99
        # TODO: Dynamisch (nur falls alle gleich trainieren) bzw. höhere Lernrate
        self.learning_rate = 0.0005
        # Create actor and critic networks
        self.actor = Actor(state_size, self.action_size, 0.1 * self.learning_rate, 0.001)
        self.critic = Critic(state_size, self.action_size, self.learning_rate, 0.001)
        self.buffer = MemoryBuffer(100000)
        self.steps = 1200
        self.noise_decay = 0.999

    def policy_action(self, s):
        """ Use the actor to predict value
        """
        return self.actor.predict(s)[0]

    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, done, new_state)

    def sample_batch(self, batch_size):
        return self.buffer.sample_batch(batch_size)

    def update_models(self, states, actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        self.critic.train_on_batch(states, actions, critic_target)
        # Q-Value Gradients under Current Policy
        actions = self.actor.model.predict(states)
        grads = self.critic.gradients(states, actions)
        # Train actor
        self.actor.train(states, actions, np.array(grads).reshape((-1, self.action_size)))
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()
        

    def train(self, env, render, batch_size, nb_episodes):
        results = []
        f = open('results.txt', 'r+')
        f.truncate(0)
        f.close()
        summary_writer = tf.summary.FileWriter('./logs')
        
        for e in range(nb_episodes):
            start = time.time()
            with open("./results.txt", "a") as myfile:
                myfile.write("***************** Episode: {}".format(e))
            # Reset episode
            time_step, cumul_reward, done = 0, 0, False
            old_state = env.reset()
            actions, states, rewards = [], [], []
            noise = OrnsteinUhlenbeckProcess(size=self.action_size, theta=.15, mu=0., sigma=.3)

            # Convert to grayscale
            old_state = cv2.resize(old_state, (15, 15))
            old_state = cv2.cvtColor(old_state, cv2.COLOR_BGR2GRAY)
            old_state = old_state.reshape(old_state.shape[0], old_state.shape[1] , 1)


            for step in range(self.steps):
                if render:
                    env.render()                   
                # Actor picks an action (following the deterministic policy)
                a = self.policy_action(old_state)
                noise_sample = noise.sample() * self.noise_decay
                #a = np.clip(a+noise_sample, -1, 1)
                # scaling the acc and brake
                if step % 500 == 0 and step != 0:
                    print("Action sample: {} with decay: {:.4f} and {}".format(a, self.noise_decay, noise_sample))
                new_state, r, done, _ = env.step(a)

                # Reshape new state
                new_state = cv2.resize(new_state, (15, 15))
                new_state = cv2.cvtColor(new_state, cv2.COLOR_BGR2GRAY)
                new_state = new_state.reshape(new_state.shape[0], new_state.shape[1], 1)

                # Remove unused information (differences in grey scale)
                for val in new_state:
                    for val1 in val:
                        if val1[0] > 140:
                            val1[0] = 255

                cv2.imshow('Car Racing', new_state)

                # Append to replay buffer
                self.memorize(old_state, a, r, done, new_state)
                # Update every batch_size steps
                # TODO!!!!!!!!!!!!!!!!!!! BADGES TRAINIEREN
                if self.buffer.count > batch_size and False:
                    states, actions, rewards, dones, new_states, _ = self.sample_batch(batch_size)
                    q_values = self.critic.target_predict([new_states, self.actor.target_predict(new_states)])
                    critic_target = self.bellman(rewards, q_values, dones)
                    self.update_models(states, actions, critic_target)
                old_state = new_state
                cumul_reward += r
                if time_step % 99 == 0:
                    with open("results.txt", "a") as myfile:
                        myfile.write("{} | Action: {}, Reward: {}".format(time_step, a, cumul_reward))
                        myfile.write("\n")
                time_step += 1
                
                if done or (step >= (self.steps - 1)):
                    with open("results.txt", "a") as myfile:
                        end = time.time()
                        myfile.write("Episode took: {} seconds".format(end-start))
                        myfile.write("\n")
                    break
            # Add to  summary
            score = self.tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()

            self.noise_decay = self.noise_decay * 0.999
            decay = self.tfSummary('noise', self.noise_decay)
            summary_writer.add_summary(decay, global_step=e)
            summary_writer.flush()
            
            self.save_weights('')
            print("Score: " + str(cumul_reward))
        return results
    
    def tfSummary(self, tag, val):
        return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])

    def save_weights(self, path):
        path += '_LR_{}'.format(self.learning_rate)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)


# TODO:
# LISTE an parameteränderungen machen!
# 1. Noise decay abschwächen
# Batch größe verringern
# Einzelne rewards mal printen, wann ist er positiv? wann negativ?
# Oscillations:
# - training frühzeitig abbrechen
# - anderes environment ausprobieren
# - Hidden unit sizes
# - buffer größer evtl? oder kleiner?
# nice visualisierung machen so wie andy
# 2. Learning rate evtl kleiner oder Grund für langsames lernen? Target net (teta gamma?)?
# Warum wird er wieder schlecht? Weil tau zu groß?
# 3. Loss funktion von actor & critic printen
