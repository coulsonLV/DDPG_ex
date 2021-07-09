import tensorflow as tf
from tqdm import tqdm
import numpy as np
import random
import time
import socket
import pickle
import scipy.io as scio
import matplotlib.pyplot as plt
from datetime import datetime
import math
from agent import BaseAgent
from ddpg_network import ActorNetwork, CriticNetwork
from IDM_MOBIL import idm_mobil


sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)#ipv4,udp
sock.bind(('169.254.61.68',54377))#UDP服务器端口和IP绑定
print('等待客户端发送请求...')
buf, addr = sock.recvfrom(40960)#等待matlab发送请求，这样就能获取matlab client的ip和端口号

sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("logs/", sess.graph)
# saver = tf.train.Saver()

class DDPGAgent(BaseAgent):
    def __init__(self, sess, action_type, actor, critic, gamma, replay_buffer, noise=None, exploration_episodes=7000, max_episodes=10000, max_steps_episode=400,\
            warmup_steps=20000, mini_batch=64, eval_episodes=20, eval_periods=100, env_render=False, summary_dir=None):
        """
        Deep Deterministic Policy Gradient Agent.
        Args:
            actor: actor network.
            critic: critic network.
            gamma: discount factor.
        """
        super(DDPGAgent, self).__init__(sess, replay_buffer, noise=noise, exploration_episodes=exploration_episodes, max_episodes=max_episodes, max_steps_episode=max_steps_episode,\
                warmup_steps=warmup_steps, mini_batch=mini_batch, eval_episodes=eval_episodes, eval_periods=eval_periods, env_render=env_render, summary_dir=summary_dir)

        self.action_type = action_type
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
    def train(self):
        # Initialize target network weights
        self.actor.update_target_network()
        self.critic.update_target_network()
        EP_Reward = []
        for cur_episode in tqdm(range(self.max_episodes)):
            # tic = datetime.now()
            # evaluate here. 
            # if cur_episode % self.eval_periods == 0 & cur_episode != 0:
            #     self.evaluate()

            # state = self.env.reset()
            #print('接收matlab环境中的初始状态值s')
            buf, addr = sock.recvfrom(40960)
            # print(buf) # b'0.1   0.2   0.3   count'
            msg = buf.split()  #以空格和\n为分隔符，一一分隔开并组成一个列表
            msg_up=np.array([msg]) # [ b'0.1' ,b'0.2' ,b'0.3']
            msg_up_up = [np.double(i) for i in msg_up] #将list中的每个bytes变为double型（变成显示的数组）
            msg_up_up_up  = np.transpose(msg_up_up)
            state = msg_up_up_up[0:11,]
            state = state.reshape((11,))
            I_M1 = msg_up_up_up[11:37,]
            I_M1 = I_M1.reshape((26,))
            I_M2 = msg_up_up_up[37:63,]
            I_M2 = I_M2.reshape((26,))
            I_M3 = msg_up_up_up[63:89,]
            I_M3 = I_M3.reshape((26,))

            episode_reward = 0
            episode_ave_max_q = 0

            for cur_step in range(self.max_steps_episode):
                # Add exploratory noise according to Ornstein-Uhlenbeck process to action
                if self.replay_buffer.size() < self.warmup_steps:
                    a1 = random.uniform(-1,1)
                    a2 = random.uniform(-1,1)
                    a3 = random.uniform(-1,1)
                    moxing_1_1 = idm_mobil(t =I_M1[0], d_0 = I_M1[1], T=I_M1[2], v5=I_M1[3], b=I_M1[4], a3=I_M1[5], a4=I_M1[6], a5=I_M1[7], x4=I_M1[8], x5=I_M1[9], an_af=I_M1[10], an_be=I_M1[11], p=I_M1[12], ao_af=I_M1[13],ao_be=I_M1[14], a_th=I_M1[15],a_max=I_M1[16], v4=I_M1[17], v_ex=I_M1[18],theta=I_M1[19],delta_d=I_M1[20],a=I_M1[21],d=I_M1[22],steer=I_M1[23],clock=I_M1[24],t0=I_M1[25])
                    A_1 = moxing_1_1.panduan() # return steer, acce, t0
                    moxing_1_2 = idm_mobil(t =I_M2[0], d_0 = I_M2[1], T=I_M2[2], v5=I_M2[3], b=I_M2[4], a3=I_M2[5], a4=I_M2[6], a5=I_M2[7], x4=I_M2[8], x5=I_M2[9], an_af=I_M2[10], an_be=I_M2[11], p=I_M2[12], ao_af=I_M2[13],ao_be=I_M2[14], a_th=I_M2[15],a_max=I_M2[16], v4=I_M2[17], v_ex=I_M2[18],theta=I_M2[19],delta_d=I_M2[20],a=I_M2[21],d=I_M2[22],steer=I_M2[23],clock=I_M2[24],t0=I_M2[25])
                    A_2 = moxing_1_2.panduan() # return steer, acce, t0
                    moxing_1_3 = idm_mobil(t =I_M3[0], d_0 = I_M3[1], T=I_M3[2], v5=I_M3[3], b=I_M3[4], a3=I_M3[5], a4=I_M3[6], a5=I_M3[7], x4=I_M3[8], x5=I_M3[9], an_af=I_M3[10], an_be=I_M3[11], p=I_M3[12], ao_af=I_M3[13],ao_be=I_M3[14], a_th=I_M3[15],a_max=I_M3[16], v4=I_M3[17], v_ex=I_M3[18],theta=I_M3[19],delta_d=I_M3[20],a=I_M3[21],d=I_M3[22],steer=I_M3[23],clock=I_M3[24],t0=I_M3[25])
                    A_3 = moxing_1_3.panduan() # return steer, acce, t0
                    action = np.array([a1, a2, a3, A_1[0], A_1[1], A_1[2],A_2[0],A_2[1],A_2[2],A_3[0],A_3[1],A_3[2]])
                else: 
                    if cur_episode < self.exploration_episodes and self.noise is not None:
                        action = np.clip(self.actor.predict(np.expand_dims(state, 0))[0] + self.noise.generate(cur_episode), -1, 1) 
                        moxing_2_1 = idm_mobil(t =I_M1[0], d_0 = I_M1[1], T=I_M1[2], v5=I_M1[3], b=I_M1[4], a3=I_M1[5], a4=I_M1[6], a5=I_M1[7], x4=I_M1[8], x5=I_M1[9], an_af=I_M1[10], an_be=I_M1[11], p=I_M1[12], ao_af=I_M1[13],ao_be=I_M1[14], a_th=I_M1[15],a_max=I_M1[16], v4=I_M1[17], v_ex=I_M1[18],theta=I_M1[19],delta_d=I_M1[20],a=I_M1[21],d=I_M1[22],steer=I_M1[23],clock=I_M1[24],t0=I_M1[25])
                        A_1 = moxing_2_1.panduan() # return steer, acce, t0
                        moxing_2_2 = idm_mobil(t =I_M2[0], d_0 = I_M2[1], T=I_M2[2], v5=I_M2[3], b=I_M2[4], a3=I_M2[5], a4=I_M2[6], a5=I_M2[7], x4=I_M2[8], x5=I_M2[9], an_af=I_M2[10], an_be=I_M2[11], p=I_M2[12], ao_af=I_M2[13],ao_be=I_M2[14], a_th=I_M2[15],a_max=I_M2[16], v4=I_M2[17], v_ex=I_M2[18],theta=I_M2[19],delta_d=I_M2[20],a=I_M2[21],d=I_M2[22],steer=I_M2[23],clock=I_M2[24],t0=I_M2[25])
                        A_2 = moxing_2_2.panduan() # return steer, acce, t0
                        moxing_2_3 = idm_mobil(t =I_M3[0], d_0 = I_M3[1], T=I_M3[2], v5=I_M3[3], b=I_M3[4], a3=I_M3[5], a4=I_M3[6], a5=I_M3[7], x4=I_M3[8], x5=I_M3[9], an_af=I_M3[10], an_be=I_M3[11], p=I_M3[12], ao_af=I_M3[13],ao_be=I_M3[14], a_th=I_M3[15],a_max=I_M3[16], v4=I_M3[17], v_ex=I_M3[18],theta=I_M3[19],delta_d=I_M3[20],a=I_M3[21],d=I_M3[22],steer=I_M3[23],clock=I_M3[24],t0=I_M3[25])
                        A_3 = moxing_2_3.panduan() # return steer, acce, t0
                        action = np.array([action[0], action[1], action[2], A_1[0], A_1[1], A_1[2],A_2[0],A_2[1],A_2[2],A_3[0],A_3[1],A_3[2]])
                    else: 
                        action = self.actor.predict(np.expand_dims(state, 0))[0] 
                        moxing_3_1 = idm_mobil(t =I_M1[0], d_0 = I_M1[1], T=I_M1[2], v5=I_M1[3], b=I_M1[4], a3=I_M1[5], a4=I_M1[6], a5=I_M1[7], x4=I_M1[8], x5=I_M1[9], an_af=I_M1[10], an_be=I_M1[11], p=I_M1[12], ao_af=I_M1[13],ao_be=I_M1[14], a_th=I_M1[15],a_max=I_M1[16], v4=I_M1[17], v_ex=I_M1[18],theta=I_M1[19],delta_d=I_M1[20],a=I_M1[21],d=I_M1[22],steer=I_M1[23],clock=I_M1[24],t0=I_M1[25])
                        A_1 = moxing_3_1.panduan() # return steer, acce, t0
                        moxing_3_2 = idm_mobil(t =I_M2[0], d_0 = I_M2[1], T=I_M2[2], v5=I_M2[3], b=I_M2[4], a3=I_M2[5], a4=I_M2[6], a5=I_M2[7], x4=I_M2[8], x5=I_M2[9], an_af=I_M2[10], an_be=I_M2[11], p=I_M2[12], ao_af=I_M2[13],ao_be=I_M2[14], a_th=I_M2[15],a_max=I_M2[16], v4=I_M2[17], v_ex=I_M2[18],theta=I_M2[19],delta_d=I_M2[20],a=I_M2[21],d=I_M2[22],steer=I_M2[23],clock=I_M2[24],t0=I_M2[25])
                        A_2 = moxing_3_2.panduan() # return steer, acce, t0
                        moxing_3_3 = idm_mobil(t =I_M3[0], d_0 = I_M3[1], T=I_M3[2], v5=I_M3[3], b=I_M3[4], a3=I_M3[5], a4=I_M3[6], a5=I_M3[7], x4=I_M3[8], x5=I_M3[9], an_af=I_M3[10], an_be=I_M3[11], p=I_M3[12], ao_af=I_M3[13],ao_be=I_M3[14], a_th=I_M3[15],a_max=I_M3[16], v4=I_M3[17], v_ex=I_M3[18],theta=I_M3[19],delta_d=I_M3[20],a=I_M3[21],d=I_M3[22],steer=I_M3[23],clock=I_M3[24],t0=I_M3[25])
                        A_3 = moxing_3_3.panduan() # return steer, acce, t0
                        action = np.array([action[0], action[1], action[2], A_1[0], A_1[1], A_1[2],A_2[0],A_2[1],A_2[2],A_3[0],A_3[1],A_3[2]])
                action[0] = round(action[0], 1)
                action[1] = round(action[1], 1)
                action[2] = round(action[2], 1)
                action[3] = round(action[3], 1)
                action[4] = round(action[4], 1)
                action[5] = round(action[5], 1)
                action[6] = round(action[6], 1)
                action[7] = round(action[7], 1)
                action[8] = round(action[8], 1)
                action[9] = round(action[9], 1)
                action[10] = round(action[10], 1)
                action[11] = round(action[11], 1)
                action = np.array([action[0],action[1],action[2],action[3],action[4],action[5],action[6],action[7],action[8],action[9],action[10],action[11]])
                a = action.reshape((12,))
                action_replay = np.array([action[0],action[1],action[2]])
                # print('a:',a)
                act = str(a)#将数据转化为String
                # print('a:',act)
                sock.sendto(bytes(act, encoding = "utf8") ,addr)#将数据转为bytes类型发送给matlab的client，utf8只是Unicode和ASCII之间的一种通用转换格式
                # print('a:', action) 
                # print('接收matlab中得到的反馈信息......')
                buf, addr = sock.recvfrom(40960)
                msg = buf.split()  #以空格和\n为分隔符，一一分隔开并组成一个列表
                msg_up=np.array([msg]) # [ b'0.1' ,b'0.2' ,b'0.3']
                msg_up_up = [np.double(i) for i in msg_up] #将list中的每个bytes变为double型（变成显示的数组）
                msg_up_up_up  = np.transpose(msg_up_up)
                obs = msg_up_up_up[0:11,]
                reward = msg_up_up_up[11,]
                reward = reward.reshape((1,))
                isdone = msg_up_up_up[12,]
                next_I_M1 = msg_up_up_up[13:39,] 
                next_I_M1 = next_I_M1.reshape((26,))
                next_I_M2 = msg_up_up_up[39:65,] 
                next_I_M2 = next_I_M2.reshape((26,))
                next_I_M3 = msg_up_up_up[65:91,] 
                next_I_M3 = next_I_M3.reshape((26,))
                #print(next_I_M1)
                I_M1 = next_I_M1
                I_M2 = next_I_M2
                I_M3 = next_I_M3
                # d_x1 = msg_up_up_up[11,]
                # d_x2 = msg_up_up_up[12,]
                # d_x3 = msg_up_up_up[13,]
                # d_x1 = int(d_x1)
                # d_x2 = int(d_x2)
                # d_x3 = int(d_x3)
                isdone = int(isdone)
                if isdone == 1:
                    terminal = True
                else:
                    terminal = False
                next_state = obs.reshape((11,))

                self.replay_buffer.add(state, action_replay, reward, terminal, next_state)

                # Keep adding experience to the memory until there are at least minibatch size samples
                if self.replay_buffer.size() > self.warmup_steps:
                    state_batch, action_batch, reward_batch, terminal_batch, next_state_batch = \
                        self.replay_buffer.sample_batch(self.mini_batch)

                    # Calculate targets
                    target_q = self.critic.predict_target(next_state_batch, self.actor.predict_target(next_state_batch))

                    y_i = np.reshape(reward_batch, (self.mini_batch, 1)) + (1 \
                            - np.reshape(terminal_batch, (self.mini_batch, 1)).astype(float))\
                            * self.gamma * np.reshape(target_q, (self.mini_batch, 1))
                    
                    predicted_q_value, _ = self.critic.train(state_batch, action_batch, y_i)

                    episode_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
                    if self.action_type == 'Continuous':
                        a_outs = self.actor.predict(state_batch)
                        a_grads = self.critic.action_gradients(state_batch, a_outs)
                        self.actor.train(state_batch, a_grads[0])
                    else:
                        a_outs = self.actor.predict(state_batch)
                        a_grads = self.critic.action_gradients(state_batch, a_outs)
                        self.actor.train(state_batch, a_grads[0])


                    # Update target networks
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                state = next_state
                episode_reward += reward
                # if d_x1 == -5 or d_x2==-5 or d_x3==-5:
                #     episode_reward = 0

                if terminal or cur_step == self.max_steps_episode-1:
                    # train_episode_summary = tf.summary.scalar("train/episode_reward", episode_reward)
                    # tf.summary.scalar("train/episode_ave_max_q", episode_ave_max_q/float(cur_step))
                    # summary_op = tf.summary.merge_all()
                    # O = s.reshape((1,9))
                    # P = s_.reshape((1,8))
                    # Q = r.reshape((1,1))
                    # resu = sess.run(summary_op, feed_dict={ActorNetwork.action_gradients, ddpg.S_:P, ddpg.R:Q})
                    # writer.add_summary(resu, cur_episode)
                    # train_episode_summary = tf.Summary() 
                    # train_episode_summary.value.add(simple_value=episode_reward, tag="train/episode_reward")
                    # train_episode_summary.value.add(simple_value=episode_ave_max_q/float(cur_step), tag="train/episode_ave_max_q")
                    # self.writer.flush()
                    # self.writer.add_summary(train_episode_summary, cur_episode)
                   
                    print('Reward: %.2i' % int(episode_reward), ' | Episode:', cur_episode+1, \
                          ' | step:',cur_step+1, '| Qmax: %.4f' % (episode_ave_max_q / float(cur_step+1)))
                    # print("loss:", CriticNetwork.loss)
                    filename =  'reward.txt'
                    d = str(episode_reward)
                    e = str(cur_episode+1)
                    e = e+':'
                    with open(filename,'a') as f:
                        f.write(e)
                        f.write(d)
                        f.write('.\n')
                    break
            # Save model
            # if np.mod(cur_episode,2)==0:
            #     saver.save(sess,'./model/ddpg_model')
            #     with open("./model/buffer.pkl","wb") as buffer_log:
            #         pickle.dump(self.replay_buffer, buffer_log)
            # toc = datetime.now()
            # print('time: %f seconds' % (toc-tic).total_seconds())
            # EP_Reward.append(episode_reward)
        # plt.figure(1)
        # plt.plot(range(10000), EP_Reward, 'r-')
        # plt.show()
    def evaluate(self):        # (self, cur_episode)
        # evaluate here. 
        total_episode_reward = 0 
        for k in tqdm (range(self.eval_episodes)):
            #k =k
            # state = self.env.reset() 
            #print('接收matlab环境中的初始状态值s')
            buf, addr = sock.recvfrom(40960)
            # print(buf) # b'0.1   0.2   0.3   count'
            msg = buf.split()  #以空格和\n为分隔符，一一分隔开并组成一个列表
            msg_up=np.array([msg]) # [ b'0.1' ,b'0.2' ,b'0.3']
            msg_up_up = [np.double(i) for i in msg_up] #将list中的每个bytes变为double型（变成显示的数组）
            msg_up_up_up  = np.transpose(msg_up_up)
            state = msg_up_up_up[0:11,]
            state = state.reshape((11,))
            I_M1 = msg_up_up_up[11:37,]
            I_M1 = I_M1.reshape((26,))
            I_M2 = msg_up_up_up[37:63,]
            I_M2 = I_M2.reshape((26,))
            I_M3 = msg_up_up_up[63:89,]
            I_M3 = I_M3.reshape((26,))
            terminal = False
            while not terminal:
                action = self.actor.predict(np.expand_dims(state, 0))[0]
                moxing_test_1 = idm_mobil(t =I_M1[0], d_0 = I_M1[1], T=I_M1[2], v5=I_M1[3], b=I_M1[4], a3=I_M1[5], a4=I_M1[6], a5=I_M1[7], x4=I_M1[8], x5=I_M1[9], an_af=I_M1[10], an_be=I_M1[11], p=I_M1[12], ao_af=I_M1[13],ao_be=I_M1[14], a_th=I_M1[15],a_max=I_M1[16], v4=I_M1[17], v_ex=I_M1[18],theta=I_M1[19],delta_d=I_M1[20],a=I_M1[21],d=I_M1[22],steer=I_M1[23],clock=I_M1[24],t0=I_M1[25])
                A_1 = moxing_test_1.panduan() # return steer, acce, t0, c
                moxing_test_2 = idm_mobil(t =I_M2[0], d_0 = I_M2[1], T=I_M2[2], v5=I_M2[3], b=I_M2[4], a3=I_M2[5], a4=I_M2[6], a5=I_M2[7], x4=I_M2[8], x5=I_M2[9], an_af=I_M2[10], an_be=I_M2[11], p=I_M2[12], ao_af=I_M2[13],ao_be=I_M2[14], a_th=I_M2[15],a_max=I_M2[16], v4=I_M2[17], v_ex=I_M2[18],theta=I_M2[19],delta_d=I_M2[20],a=I_M2[21],d=I_M2[22],steer=I_M2[23],clock=I_M2[24],t0=I_M2[25])
                A_2 = moxing_test_2.panduan() # return steer, acce, t0
                moxing_test_3 = idm_mobil(t =I_M3[0], d_0 = I_M3[1], T=I_M3[2], v5=I_M3[3], b=I_M3[4], a3=I_M3[5], a4=I_M3[6], a5=I_M3[7], x4=I_M3[8], x5=I_M3[9], an_af=I_M3[10], an_be=I_M3[11], p=I_M3[12], ao_af=I_M3[13],ao_be=I_M3[14], a_th=I_M3[15],a_max=I_M3[16], v4=I_M3[17], v_ex=I_M3[18],theta=I_M3[19],delta_d=I_M3[20],a=I_M3[21],d=I_M3[22],steer=I_M3[23],clock=I_M3[24],t0=I_M3[25])
                A_3 = moxing_test_3.panduan() # return steer, acce, t0
                action = np.array([action[0], action[1], action[2], A_1[0], A_1[1], A_1[2], A_2[0], A_2[1], A_2[2], A_3[0], A_3[1], A_3[2]])
                action[0] = round(action[0], 1)
                action[1] = round(action[1], 1)
                action[2] = round(action[2], 1)
                action[3] = round(action[3], 1)
                action[4] = round(action[4], 1)
                action[5] = round(action[5], 1)
                action[6] = round(action[6], 1)
                action[7] = round(action[7], 1)
                action[8] = round(action[8], 1)
                action[9] = round(action[9], 1)
                action[10] = round(action[10], 1)
                action[11] = round(action[11], 1)
                action = np.array([action[0],action[1],action[2],action[3],action[4],action[5],action[6],action[7],action[8],action[9],action[10],action[11]])
                a = action.reshape((12,))
                # print('a:',a)
                act = str(a)#将数据转化为String
                sock.sendto(bytes(act, encoding = "utf8") ,addr)#将数据转为bytes类型发送给matlab的client，utf8只是Unicode和ASCII之间的一种通用转换格式
                # print('a:', action) 
                # print('接收matlab中得到的反馈信息......')
                buf, addr = sock.recvfrom(40960)
                msg = buf.split()  #以空格和\n为分隔符，一一分隔开并组成一个列表
                msg_up=np.array([msg]) # [ b'0.1' ,b'0.2' ,b'0.3']
                msg_up_up = [np.double(i) for i in msg_up] #将list中的每个bytes变为double型（变成显示的数组）
                msg_up_up_up  = np.transpose(msg_up_up)
                obs = msg_up_up_up[0:11,]
                reward = msg_up_up_up[11,]
                reward = reward.reshape((1,))
                isdone = msg_up_up_up[12,] 
                next_I_M1 = msg_up_up_up[13:39,] 
                next_I_M1 = next_I_M1.reshape((26,))
                next_I_M2 = msg_up_up_up[39:65,] 
                next_I_M2 = next_I_M2.reshape((26,))
                next_I_M3 = msg_up_up_up[65:91,] 
                next_I_M3 = next_I_M3.reshape((26,))
                #print(next_I_M1)
                I_M1 = next_I_M1
                I_M2 = next_I_M2
                I_M3 = next_I_M3
                # d_x1 = msg_up_up_up[11,]
                # d_x2 = msg_up_up_up[12,]
                # d_x3 = msg_up_up_up[13,]
                # d_x1 = int(d_x1)
                # d_x2 = int(d_x2)
                # d_x3 = int(d_x3)
                isdone = int(isdone)
                if isdone == 1:
                    terminal = True
                else:
                    terminal = False
                state = obs.reshape((11,))
                total_episode_reward += reward
                # if d_x1 == -5 or d_x2==-5 or d_x3==-5:
                #     total_episode_reward = 0
        ave_episode_reward = total_episode_reward / float(self.eval_episodes)
        print("\nAverage reward {}\n".format(ave_episode_reward))
        # Add ave reward to Tensorboard
        # eval_episode_summary = tf.Summary.Value(simple_value=ave_episode_reward, tag="eval/reward")
        # summaryy = tf.Summary(vaule=[eval_episode_summary])
        # self.writer.add_summary(summaryy, cur_episode)

