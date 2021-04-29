import sys
sys.path.insert(1, '/home/home3/abef/research/control/LinearizingStateRepresentation/')

from lib.restartable_pendulum import RestartablePendulumEnv
from lib import state_rep_torch as srt
from lib import encoder_wrappers as ew
from gym import Wrapper, ObservationWrapper
from gym.wrappers.time_limit import TimeLimit
from lib import repeated_wrapper
import gym
import numpy as np
from matplotlib import pyplot as plt
import itertools
import sys
import torch
import torch.optim as optim
import tensorflow as tf
from scipy import linalg as la
import random

from stable_baselines.common.policies import FeedForwardPolicy as FFP
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO, A2C, ACKTR, DDPG, PPO2, SAC, TD3
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.ddpg import policies as ddpgPolicies
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.sac import policies as sacPolicies
from stable_baselines.td3 import policies as td3Policies


# training hyperparameters
T = 10 # trajectory length
n_repeats = 3
n_episodes = 6 # number of episodes (or outer loops -- consists of training rep and policy)
n_batches = 20 # how many batches of trajectories to train on during each episode
n_sweeps = 2 # number of times to loop through training data during each episode
pol_timesteps = 5000 # how many steps to learn the policy in each episode
lr=1e-3 #learning rate for encoder training
batch_size = 25 # batch size for encoder training
track_loss_every=10 # print a progress statement every _ batches during encoder training
layers = [50, 50,5] # fully-connected layers of the encoder
nonlin = torch.nn.ReLU() # nonlinearity for encoder
action_dimension = 1



class policy():
    """
    Simple policy class that has a predict method which maps states to actions
    
    Parameters
    ----------
    pred : callable function that takes one input (the state) and outputs an action
    """
    def __init__(self, pred):
        self.pred = pred
    def predict(self, state):
        return self.pred(state)
    
class EncoderPolicy():
    """
    Policy class that composes a state encoder with a stable-baselines style policy.
    
    Parameters
    ----------
    encoder : callable function that maps environment observation to encoded state
    model : stable-baselines style model with a predict method
    """
    def __init__(self, encoder, model):
        self.encoder = encoder
        self.model = model
    def predict(self, state):
        return self.model.predict(self.encoder(state))[0]
    
class TorchStateEncoder():
    """
    Wraps a torch encoder module to allow easy encoding of environment observations.
    We use this wrapper to first convert the numpy array observation to a torch tensor,
    and then feed that to the torch encoder.
    
    Parameters
    ----------
    encoder : pytorch module that encodes the environment observations.
    """
    def __init__(self, encoder):
        self.encoder = encoder
        
    def __call__(self,obs):
        with torch.no_grad(): # we aren't doing autograd, just computation
            # the torch encoder expects a batch of inputs, so create new axis
            rep = self.encoder(torch.from_numpy(np.expand_dims(obs,0)).float()).numpy()[0]
        return rep
    
    
env = RestartablePendulumEnv(repeats=n_repeats,pixels=True)
pol = policy(lambda x: np.random.rand(1)*2-1) # initial random policy
sampler = srt.PolicyTrajectorySampler(env,pol,T)

encnet = srt.ConvEncoderNet(layers,env.observation_space.shape[1:],sigma=nonlin)
rep_model = srt.ForwardNet(encnet, layers[-1],action_dimension, fixed_B = True)



for ep in range(n_episodes):

    print("\nStarting episode {0} of {1}".format(ep+1,n_episodes))
    


    # run the sampler and create a dataset
    samples = []
    print("Gathering data...")
    for zz in range(n_batches):
        samples.extend(sampler.get_batch(batch_size))
    random.shuffle(samples)

    # turn on gradients and train mode
    encnet.train()
    for param in encnet.parameters():
        param.requires_grad = True
    
    # initialize optimizer
    optimizer = optim.Adam(rep_model.parameters(),lr=lr)
    running_loss = 0.0
    
   
    print("Training encoder...")
    for j in range(n_sweeps):
        print("Starting loop {0} of {1}...".format(j+1,n_sweeps))
        for i,batch in enumerate(samples): # for each batch, do a gradient step
            optimizer.zero_grad()
            loss = rep_model.loss(batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if ((i+1)%track_loss_every) == 0:
                print("Loss: {0:.5f}".format(running_loss/track_loss_every),end="\r",flush=True)
                running_loss = 0.0
        print("")

    # turn off gradients and put in eval mode
    encnet.eval()
    for param in encnet.parameters():
        param.requires_grad = False
    
    
    # now create and train a policy...
   
    def make_policy_env():
        repeats = 3
        pol_env = RestartablePendulumEnv(repeats=repeats,pixels=True) # can specify cost="dm_control"
        pol_env = TimeLimit(pol_env,max_episode_steps=int(200/repeats)) # only run the environment for 200 true steps
        proj = np.eye(rep_model.enc_dim)
        return ew.TorchEncoderWrapper(pol_env,encnet,proj)

    print("Training policy...")
    pol_env = DummyVecEnv([make_policy_env])
    # nonlinear policy trained by PPO
    #model = PPO2(MlpPolicy, pol_env, verbose=0)
    
    # linear policy trained by TRPO
    pol_kwargs = {"net_arch" : [dict(vf=[64,64], pi=[])], 
                         "feature_extraction" : "mlp",
                     "act_fun" : tf.keras.activations.linear}
    model = TRPO(FFP, pol_env, verbose=0, policy_kwargs = pol_kwargs)
    model.learn(total_timesteps=pol_timesteps)
    
    # evaluate the policy
    print("Evaluating policy...")
    n_evals = 5
    eval_rollout = int(200/3)
    eval_rewards = []
    for _ in range(n_evals):
        obs = pol_env.reset()
        rollout_rewards = []
        for _ in range(eval_rollout):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = pol_env.step(action)
            rollout_rewards.append(rewards/3)
        eval_rewards.append(np.mean(rollout_rewards))
    print("Mean eval step reward: {}".format(np.mean(eval_rewards)))
    
    # update the policy and sampler objects
    pol = EncoderPolicy(TorchStateEncoder(encnet),model)
    sampler = srt.PolicyTrajectorySampler(env,pol,T)

# save stuff
torch.save(rep_model,"./repnet")
model.save("./model")


# train the model more?
"""
repmodel = torch.load("./repnet")
encnet = repmodel.encoder
#model = PPO2.load("./model")

def make_policy_env():
    repeats = 3
    pol_env = RestartablePendulumEnv(repeats=repeats,pixels=True) # can specify cost="dm_control"
    pol_env = TimeLimit(pol_env,max_episode_steps=int(200/repeats)) # only run the environment for 200 true steps
    proj = np.eye(rep_model.enc_dim)
    return ew.TorchEncoderWrapper(pol_env,encnet,proj)

print("Training policy linear...")
pol_env = DummyVecEnv([make_policy_env])
pol_kwargs = {"net_arch" : [dict(vf=[64,64], pi=[])], 
                         "feature_extraction" : "mlp",
                     "act_fun" : tf.keras.activations.linear}
model = TRPO(FFP, pol_env, verbose=1, policy_kwargs = pol_kwargs)
model.learn(total_timesteps=50000)
model.save("./linear_model")
for _ in range(0):
    model.learn(total_timesteps=1000)
    print("Evaluating policy...")
    n_evals = 5
    eval_rollout = int(200/3)
    eval_rewards = []
    for _ in range(n_evals):
        obs = pol_env.reset()
        rollout_rewards = []
        for _ in range(eval_rollout):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = pol_env.step(action)
            rollout_rewards.append(rewards/3)
        eval_rewards.append(np.mean(rollout_rewards))
    print("Mean eval step reward: {}".format(np.mean(eval_rewards)))

"""