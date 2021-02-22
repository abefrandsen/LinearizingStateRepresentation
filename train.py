import sys
from lib.restartable_pendulum import RestartablePendulumEnv
from lib import state_rep_torch as srt
import gym
import numpy as np
from matplotlib import pyplot as plt
import torch
from lib import utils
from lib import encoder_wrappers as ew


# specify environment information
n_repeats = 3 # step the environment this many times for each action, concatenate the pixel observations
env = RestartablePendulumEnv(repeats=n_repeats,pixels=True)


nonlin = torch.nn.ELU()
#nonlin = torch.nn.functional.relu
layers = [50, 50, 5] # architecture of encoder after the 2 conv layers
save_dir = "./"

n_episodes = 500 # total batches to draw
batch_size = 25
learning_rate = .01
save_every = int(n_episodes/1) # save the model every so often
track_loss_every = int(n_episodes/n_episodes)

encnet = srt.ConvEncoderNet(layers,env.observation_space.shape[1:],sigma=nonlin)

# use the following commented out lines for PredictorNet (I changed deterministic sampling though...
#prednet = srt.PredictorNet(encnet,T,layers[-1],1)
#deterministic_args = (samples[i], batch_size, 35, method, n_repeats,T) 

prednet = srt.ForwardNet(encnet,layers[-1],1)
#prednet = srt.PiecewiseForwardNet(encnet,layers[-1],1,2)
deterministic_args = None 

traj_sampler = srt.SimpleTrajectorySampler(env,
                                     srt.sample_pendulum_action_batch,
                                     srt.sample_pendulum_state_batch_old,
                                     device=torch.device("cpu"),
                                     deterministic=False,
                                     deterministic_args=deterministic_args)

net, losses = srt.train_encoder(prednet,traj_sampler,n_episodes,
                                batch_size=batch_size,
                                track_loss_every=track_loss_every,
                                lr=learning_rate,
                                save_every=save_every,
                                save_path=save_dir)

torch.save(net,save_dir+"net")
np.savetxt("losses.txt",np.array(losses))


# what follows is code to visualize the representations
d = 5 # must match the final entry in layers
n_samps = 500
env = ew.TorchEncoderWrapper(env,net.encoder,np.eye(d))
X = np.empty((n_samps,d))
for i,ang in enumerate(np.linspace(0,2*np.pi,n_samps)): # go through the angles from 0 to 2pi
    X[i,:] = env.reset(state=[ang,0])
utils.visualize_trajectory(X)
