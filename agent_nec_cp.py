import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np
from sklearn.neighbors.kd_tree import KDTree
import numpy as np
import Nec_cp
import math
import pickle
from chainer import \
     cuda, gradient_check, optimizers, serializers, utils, \
     Chain, ChainList, Function, Link, Variable
import os


def select_action_epsilon_greedily(epsilon, random_action_func,
                                   greedy_action_func):
    if np.random.rand() < epsilon:
        return random_action_func(), False
    else:
        return greedy_action_func(), True


class LinearDecayEpsilonGreedy(object):
    """Epsilon-greedy with linearyly decayed epsilon

    Args:
      start_epsilon: max value of epsilon
      end_epsilon: min value of epsilon
      decay_steps: how many steps it takes for epsilon to decay
      random_action_func: function with no argument that returns action
      logger: logger used
    """

    def __init__(self, start_epsilon, end_epsilon,
                 decay_steps, random_action_func):
        assert start_epsilon >= 0 and start_epsilon <= 1
        assert end_epsilon >= 0 and end_epsilon <= 1
        assert decay_steps >= 0
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.random_action_func = random_action_func
        self.epsilon = start_epsilon

    def compute_epsilon(self, t):
        if t > self.decay_steps:
            return self.end_epsilon
        else:
            epsilon_diff = self.end_epsilon - self.start_epsilon
            return self.start_epsilon + epsilon_diff * (t / self.decay_steps)

    def select_action(self, t, greedy_action_func, action_value=None):
        self.epsilon = self.compute_epsilon(t)
        a, greedy = select_action_epsilon_greedily(
            self.epsilon, self.random_action_func, greedy_action_func)
        greedy_str = 'greedy' if greedy else 'non-greedy'
        return a

class QFunction(chainer.Chain):
    def __init__(self, n_actions, n_hidden_channels=50):
        self.q_list = np.ndarray((1,n_actions),dtype=np.float32)#for discreteActionValue
        super(QFunction, self).__init__(
            l0=L.Linear(4, 4),)
            #l1=L.Linear(n_hidden_channels, n_hidden_channels),
            #l2=L.Linear(n_hidden_channels, 2))# 4))

    def __call__(self, x, ma):
        h = F.tanh(self.l0(x))
        #h = F.tanh(self.l1(h))
        #h = F.tanh(self.l2(h))

        #kd_tree
        q_train = [] #for train [variable,variable]
        ind_list = []#for train
        dist_list = [] #for train
        for j in range(len(ma.maq)):#loop n_actions
            h_list = ma.mah[j]
            lp = len(h_list)
            leaf_size = lp + (lp/2)

            tree = KDTree(h_list,leaf_size=leaf_size)
            h_ = h.data
            
            if lp < 50:
                k = lp
            else:
                k = 50
            dist, ind = tree.query(h_,k=k)

            count = 0
            for ii in ind[0]:
                mahi = np.zeros((1,4),dtype=np.float32)
                mahi[0] = ma.mah[j][ii]
                hi = chainer.Variable(cuda.to_cpu(mahi))
                wi = F.expand_dims(1/(F.batch_l2_norm_squared((h-hi))+0.001),1)

                if count == 0:
                    w = wi
                    maqi = np.zeros((1,1),dtype=np.float32)
                    maqi[0] = ma.maq[j][ii]
                    q = chainer.Variable(cuda.to_cpu(maqi))
                    qq = wi * q
                    count += 1
                else:
                    w += wi
                    maqi = np.zeros((1,1),dtype=np.float32)
                    maqi[0] = ma.maq[j][ii]
                    q = chainer.Variable(cuda.to_cpu(maqi))
                    qq += wi * q
            qq /= w

            q_train.append(qq)
            ind_list.append(ind)
            dist_list.append(dist)
            self.q_list[0][j] = qq.data[0][0]
        qa = chainer.Variable(cuda.to_cpu(self.q_list))
        return chainerrl.action_value.DiscreteActionValue(qa),q_train,ind_list,dist_list,h.data


env = gym.make('CartPole-v0')
print('observation space:', env.observation_space)
print('action space:', env.action_space)
obs = env.reset()

n_actions = env.action_space.n
q_func = QFunction(n_actions)

q_func.to_cpu()

explorer = LinearDecayEpsilonGreedy(1.0,0.1,10**4,env.action_space.sample)

optimizer = chainer.optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.0001)
optimizer.setup(q_func)

gamma = 0.99

phi = lambda x: x.astype(np.float32, copy=False)

agent = Nec_cp.NEC(q_func, optimizer, gamma, explorer,n_actions,False)


n_episodes = 1000
max_episode_len = 200

train_Flug = True
all_frame = 0

try:
    os.mkdir("./models")
except:
    pass

for i in range(1, n_episodes + 1):
    obs = env.reset()
    reward = 0
    done = False
    R = 0 # return (sum of rewards)
    t = 0 # time step
    while not done and t < max_episode_len:
        # Uncomment to watch the behaviour
        #env.render()
        xx1 = np.zeros((1,4),dtype=np.float32)
        xx1[0] = obs
        xx = chainer.Variable(cuda.to_cpu(xx1))
        action, ind, dist, key = agent.act(xx,i)
        obs, reward, done, _ = env.step(action)

        if train_Flug == True:
            agent.append_memory_and_train(xx1,action,ind,dist,reward,key,done)
        R += reward
        t += 1
        all_frame += 1

    if i % 1 == 0:
        print('episode:', i,
            'R:', R,
            'statistics:', agent.get_statistics())
        f = open("nec_log_cartpole.txt","a")
        f.write("episode"+" "+str(i)+" "+"R"+" "+str(R)+" " + str(all_frame)+ "\n")
        f.close
    if i == 1:
        serializers.save_npz("%s/dec_pole_%d.npz"%("models", i),q_func)
        pickle.dump(agent.n_step_memory.ma_memory.maq,open("%s/Ma_memory_maq_%d.pickle"%("models",i),"wb"),-1)
        pickle.dump(agent.n_step_memory.ma_memory.mah,open("%s/Ma_memory_mah_%d.pickle"%("models",i),"wb"),-1)

    if i % 10 == 0 and train_Flug == True:
        serializers.save_npz("%s/dec_pole_%d.npz"%("models", i),q_func)
        pickle.dump(agent.n_step_memory.ma_memory.maq,open("%s/Ma_memory_maq_%d.pickle"%("models",i),"wb"),-1)
        pickle.dump(agent.n_step_memory.ma_memory.mah,open("%s/Ma_memory_mah_%d.pickle"%("models",i),"wb"),-1)
    agent.stop_episode_and_train(obs, reward, done)

print('Finished.')


