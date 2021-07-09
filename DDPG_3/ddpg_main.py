import numpy as np
import datetime
import time
import socket
import pickle
import scipy.io as scio
import matplotlib.pyplot as plt
import datetime
import math
import gym
from gym.wrappers import Monitor
import tensorflow as tf
from tqdm import tqdm
from network import BaseNetwork
from ddpg_agent import DDPGAgent
from ddpg_network import CriticNetwork, ActorNetwork
from replaybuffer import ReplayBuffer
from explorationnoise import OrnsteinUhlenbeckProcess, GreedyPolicy

# sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)#ipv4,udp
# sock.bind(('169.254.61.68',54377))#UDP服务器端口和IP绑定
# print('等待客户端发送请求...')
# buf, addr = sock.recvfrom(40960)#等待matlab发送请求，这样就能获取matlab client的ip和端口号

flags = tf.app.flags

# ================================
#    UTILITY PARAMETERS
# ================================
# Gym environment name
#'Pendulum-v0''MountainCarContinuous-v0'
flags.DEFINE_string('env_name', 'Pendulum-v0', 'environment name in gym.')
flags.DEFINE_boolean('env_render', False, 'whether render environment (display).')
flags.DEFINE_boolean('env_monitor', True, 'whether use gym monitor.')
DATETIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
RANDOM_SEED = 1234


# ================================
#    TRAINING PARAMETERS
# ================================
#flags.DEFINE_integer('mini_batch', 64, 'mini batch size for training.')
mini_batch = 64
# Learning rates actor and critic
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
# Maximum number of episodes
MAX_EPISODES =  10000 #10000
# Maximum number of steps per episode
MAX_STEPS_EPISODE =  200   #50000
# warmup steps.
WARMUP_STEPS = 2000 #10000
# Exploration duration
EXPLORATION_EPISODES =  1000 # 10000
# Discount factor
GAMMA = 0.99
# Soft target update parameter
TAU = 0.001
# Size of replay buffer
BUFFER_SIZE =  800000   #1000000
# Exploration noise variables Ornstein-Uhlenbeck variables
OU_THETA = 0.1
OU_MU = 0.
OU_SIGMA = 0.3
# Explorationnoise for greedy policy
MIN_EPSILON = 0.1
MAX_EPSILON = 1

#================
# parameters for evaluate.
#================
# evaluate periods
EVAL_PERIODS = 100
# evaluate episodes
EVAL_EPISODES = 500#10


FLAGS = flags.FLAGS

# Directory for storing gym results
#MONITOR_DIR = './results/{}/{}/gym_ddpg'.format(FLAGS.env_name, DATETIME)
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/{}/{}/tf_ddpg'.format("Strcture", DATETIME)


# ================================
#    MAIN
# ================================
def main(_):
    with tf.Session() as sess:
        # env = gym.make(FLAGS.env_name)
        # np.random.seed(RANDOM_SEED)
        # tf.set_random_seed(RANDOM_SEED)
        # env.seed(RANDOM_SEED)

        state_dim = 11
        action_dim = 3
        action_bound = 1
        # Ensure action bound is symmetric
        # assert(np.all(env.action_space.high == -env.action_space.low))
        action_type = 'Continuous'

        # print(action_type)
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             ACTOR_LEARNING_RATE, TAU, action_type)

        critic = CriticNetwork(sess, state_dim, action_dim, action_bound,
                               CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(), action_type)

        # Initialize replay memory
        replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
        noise = OrnsteinUhlenbeckProcess(OU_THETA, mu=OU_MU, sigma=OU_SIGMA, n_steps_annealing=EXPLORATION_EPISODES)

        agent = DDPGAgent(sess, action_type, actor, critic, GAMMA, replay_buffer, noise=noise, exploration_episodes=EXPLORATION_EPISODES,\
                max_episodes=MAX_EPISODES, max_steps_episode=MAX_STEPS_EPISODE, warmup_steps=WARMUP_STEPS,\
                mini_batch=mini_batch, eval_episodes=EVAL_EPISODES, eval_periods=EVAL_PERIODS, \
                env_render=False, summary_dir=SUMMARY_DIR)

        agent.train()
        agent.evaluate()


if __name__ == '__main__':
    tf.app.run()

