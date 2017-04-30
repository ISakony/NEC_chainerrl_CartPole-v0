from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *
from future import standard_library
standard_library.install_aliases()
import copy
from logging import getLogger
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainerrl import agent
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.copy_param import synchronize_parameters
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import state_reset
from chainer.functions.loss import mean_squared_error
import pickle


class Ma_memory(object):
    def __init__(self,n_of_action,limit_n_of_memory):
        self.limit_n_of_memory = limit_n_of_memory
        self.m = 1
        self.maq = self.first_memory_q(n_of_action)
        self.maq_c = self.m
        self.mah = self.first_memory_key(n_of_action)
        self.mah_c = self.m

    def first_memory_q(self,n_of_action):
        mlist = []
        for ii in range(n_of_action):
            ml = np.zeros((self.limit_n_of_memory,1),dtype=np.float32)
            for i in range(self.m):
                mm = np.random.rand(1)
                mm = mm.astype(np.float32)
                ml[0] = mm
            mlist.append(ml)
        return mlist

    def first_memory_key(self,n_of_action):
        mlist = []
        for ii in range(n_of_action):
            ml = np.zeros((self.limit_n_of_memory,4),dtype=np.float32)
            for i in range(self.m):
                mm = np.random.rand(4)
                mm = mm.astype(np.float32)
                ml[0] = mm
            mlist.append(ml)
        return mlist

    def add_memory(self,h,q,greedy_action,dist,ind,alpha):
        if self.mah_c == self.limit_n_of_memory+1:
                print("change")
                self.mah[greedy_action][ind[greedy_action][0][0]] = h[0]
                self.maq[greedy_action][ind[greedy_action][0][0]] = q[0]

        elif dist[greedy_action][0][0] == 0.0:
                print("update",self.maq[greedy_action][ind[greedy_action][0][0]],alpha * (q[0] - self.maq[greedy_action][ind[greedy_action][0][0]]))
                self.maq[greedy_action][ind[greedy_action][0][0]] = self.maq[greedy_action][ind[greedy_action][0][0]] + alpha * (q[0] - self.maq[greedy_action][ind[greedy_action][0][0]])
        else:

            self.mah[greedy_action][self.mah_c-1] = h[0]
            self.mah_c += 1

            self.maq[greedy_action][self.maq_c-1] = q[0]
            self.maq_c += 1

class D_memory(object):
    def __init__(self,limit_n_of_memory):
        self.limit_n_of_memory = limit_n_of_memory
        self.mem = []

    def add_memory(self,state,action,q,ind):
        if len(self.mem) == self.limit_n_of_memory:
            self.mem = self.replace(state,action,q,ind)
        else:
            self.mem.append((state,action,q,ind))

    def replace(self,state,action,q,ind):
        mem_re = []
        for i in range(len(self.mem)):
            if i == len(self.mem)-1:
                mem_re.append((state,action,q,ind))
            elif len(self.mem) != 1 and i == 0:
                pass
            else:
                mem_re.append(self.mem[i+1])
        return mem_re

class N_step_memory(object):
    def __init__(self,gamma,N_horizon,ma_memory,d_memory,alpha):
        self.mem = []
        self.gamma = gamma
        self.N_horizon = N_horizon
        self.ma_memory = ma_memory
        self.d_memory = d_memory
        self.alpha = alpha

    def add_replace_memory(self,state,action,key,reward,t_step,done,q_ontime,dist,ind):

        poplist = []
        for i in range(len(self.mem)):
            self.mem[i][4] = self.mem[i][4] + 1 #n_step +1
            if done == False and self.mem[i][4] != self.N_horizon:
                self.mem[i][3] += (self.gamma ** (self.mem[i][4]-1)) * reward
            elif done == True:
                self.mem[i][3] += (self.gamma ** (self.mem[i][4]-1)) * reward

                t_qn = np.ndarray((1,1),dtype = np.float32)
                t_qn[0] = self.mem[i][3] #qn

                self.ma_memory.add_memory(self.mem[i][2],t_qn,self.mem[i][1],self.mem[i][5],self.mem[i][6],self.alpha)
                self.d_memory.add_memory(self.mem[i][0],self.mem[i][1],t_qn,self.mem[i][6])

                poplist.append(i)

            elif self.mem[i][4] == self.N_horizon:
                self.mem[i][3] += (self.gamma ** (self.mem[i][4]-1)) * q_ontime[0]

                t_qn = np.ndarray((1,1),dtype = np.float32)
                t_qn[0] = self.mem[i][3] #qn

                self.ma_memory.add_memory(self.mem[i][2],t_qn,self.mem[i][1],self.mem[i][5],self.mem[i][6],self.alpha)
                self.d_memory.add_memory(self.mem[i][0],self.mem[i][1],t_qn,self.mem[i][6])

                poplist.append(i)

        for ii in range(len(poplist)):
            self.mem.pop(poplist[ii]-ii)

        if done == True:
            self.mem = []

        if t_step == 1:
            qn = reward
            self.mem.append([state,action,key,qn,t_step,dist,ind])


class NEC(object):
    def __init__(self,q_function, optimizer, gamma,explorer,n_actions,retrain):
        self.model = q_function
        self.n_actions = n_actions
        self.N_horizon = 100       
        self.phi=lambda x: x
        self.update_frequency=1
        self.minibatch_size = 16#32
        self.size_of_memory = 100000 #5*10^5

        if retrain == False:
            self.ma_memory = Ma_memory(n_actions,self.size_of_memory)
        else:
            self.ma_memory = Ma_memory(n_actions,self.size_of_memory)
            self.ma_memory.maq = pickle.load(open("Ma_memory_maq_.pickle","rb"))
            self.ma_memory.mah = pickle.load(open("Ma_memory_mah_.pickle","rb"))

        self.d_memory = D_memory(100000)
        self.alpha = 0.1 #learning late
        self.n_step_memory = N_step_memory(gamma,self.N_horizon,self.ma_memory,self.d_memory,self.alpha)
        self.optimizer = optimizer
        self.gamma = gamma #0.99
        self.explorer = explorer
        self.n_step = 0
        self.q_ontime = None #
        self.q_target = None
        self.t = 0
        self.average_loss_all = None
        self.average_loss_c =0
        self.average_loss = None

    def act(self, state, episode):#
        self.n_step += 1
        if self.n_step == self.N_horizon + 1:
            self.n_step = 1

        #action
        with chainer.no_backprop_mode():
            action_value = self.model(state,self.ma_memory)


        greedy_action = cuda.to_cpu(action_value[0].greedy_actions.data)[0]

        if episode < 1:
            action = np.random.randint(0,self.n_actions)
        else:
            action = self.explorer.select_action(
                self.t, lambda: greedy_action, action_value=action_value[0])
        self.t += 1

        #print(action_value[1])
        self.q_ontime = cuda.to_cpu(action_value[1][greedy_action].data)

        self.last_action = action
        ind = action_value[2]
        dist = action_value[3]
        key = cuda.to_cpu(action_value[4])

        return self.last_action,ind,dist,key

    def append_memory_and_train(self,state,action,ind,dist,reward,key,done):
        t_step = 1
        self.n_step_memory.add_replace_memory(state,action,key,reward,t_step,done,self.q_ontime,dist,ind)

        if len(self.d_memory.mem) < self.minibatch_size:
            n_loop = len(self.d_memory.mem)
        else:
            n_loop = self.minibatch_size

        if self.t % 5 == 0:#16
            loss = 0.0
            for ii in range(n_loop):
                rnd = np.random.randint(len(self.d_memory.mem))
                obs, action, t_q, t_ind = self.n_step_memory.d_memory.mem[rnd]
                obs = chainer.Variable(cuda.to_cpu(obs))
                t_q = chainer.Variable(cuda.to_cpu(t_q))
                tav = self.model(obs,self.n_step_memory.ma_memory)
                greedy_action = cuda.to_cpu(tav[0].greedy_actions.data)[0]
                train_q = tav[1][greedy_action]

                loss += mean_squared_error.mean_squared_error(train_q, t_q)
                self.average_loss_all += loss
                self.average_loss_c += 1.0
                self.average_loss = self.average_loss_all / self.average_loss_c


            if n_loop != 0:
                loss /= n_loop
                self.optimizer.zero_grads()
                loss.backward()
                self.optimizer.update()


    def stop_episode_and_train(self, state, reward, done=False):
        """Observe a terminal state and a reward.

        This function must be called once when an episode terminates.
        """
        assert self.last_action is not None

        # Add a transition to d_memory
        self.stop_episode()

    def stop_episode(self):
        self.last_action = None
        self.q_ontime = None
        self.n_step = 0
        self.average_loss = 0.0
        self.average_loss_all = 0.0
        self.average_loss_c = 0.0

    def get_statistics(self):
        return [
            ('Ma', "q", len(self.n_step_memory.ma_memory.maq[0]),len(self.n_step_memory.ma_memory.maq[1])),
            ('D', len(self.n_step_memory.d_memory.mem)),
        ]







