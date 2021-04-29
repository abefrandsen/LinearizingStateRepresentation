import sys
sys.path.insert(1, '/home/home3/abef/research/control/LinearizingStateRepresentation/')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import gym
from scipy import linalg as la
import itertools
from gym.wrappers.time_limit import TimeLimit
from lib.restartable_pendulum import RestartablePendulumEnv
import lib.encoder_wrappers as ew
from stable_baselines.common.vec_env import DummyVecEnv


# format a batch as a list of states (each entry of list is shape (batch_size, state_size), corresponding to x0, x1,...)
# and list of actions (each entry of shape (batch_size, action_size), u0,u1,...)

#device = torch.device("cuda:0")

def sample_pendulum_action_batch_zero(batch_size):
    return np.zeros((batch_size,1))

def sample_car_action_batch(batch_size):
    return 2*np.random.rand(batch_size,1)-1

def sample_racecar_action_batch(batch_size):
    return np.random.rand(batch_size,3)*np.array([2,1,1]) - np.array([1,0,0])
def sample_racecar_state_batch(batch_size):
    # do nothing here, just return something with the right indexable shape
    return np.zeros((batch_size,1))

def sample_car_state_batch(batch_size):
    p = np.random.uniform(low=-1.2,high=.6,size=(batch_size,1))
    v = np.random.uniform(low=-.07,high=.07,size=(batch_size,1))
    return np.concatenate((p,v),axis=1)

def sample_hopper_action_batch(batch_size):
    return np.random.uniform(low=-1,high=1,size=(batch_size,3))

def sample_cheetah_action_batch(batch_size):
    return np.random.uniform(low=-1,high=1,size=(batch_size,6))

def sample_pendulum_action_batch(batch_size):
    """
    Generate a batch of valid random actions for pybullet pendulum.
    
    Parameters
    ----------
    batch_size : int
    
    Returns
    -------
    actions : ndarray of shape (batch_size,1)
    """
    return 2*np.random.rand(batch_size,1)-1

def sample_pendulum_state_batch_old(batch_size):
    """
    Generate a batch of valid random initial states for the openai gym inverted pendulum.
    
    Parameters
    ----------
    batch_size : int
    
    Returns
    -------
    states : ndarray of shape (batch_size,2)
    """
    # first column is angle theta, it is valid between -pi and pi
    # second column is thetadot, range is [-8,8]
    states = np.random.uniform(low=-1,high=1,size=(batch_size,2))
    return states*np.array([np.pi,8])

def sample_pendulum_state_batch(batch_size):
    """
    Generate a batch of valid random initial states for the pybullet inverted pendulum.
    
    Parameters
    ----------
    batch_size : int
    
    Returns
    -------
    states : ndarray of shape (batch_size,4)
    """
    # first column is angle theta, it is valid between -.2 and .2
    # second column is thetadot, its range overall can be at least [-15,15]
    # third column is cart position x, its range can be [-1,1]
    # fourth column is xv (cart vel), its range can be at least [-2.6,2.6]
    states = np.random.uniform(low=-1,high=1,size=(batch_size,4))
    return states*np.array([.2,5,.8,2])

def get_trajectory_drift(env,x0,u_list,t,drift_action=[0]):
    """
    Generate a trajectory of length t starting at initial state, taking the first action, 
    drifting until the last step, then finally taking the second action.
    
    Parameters
    ----------
    env : restartable gym environment
    x0 : initial state of the environment
    u_list : list of two actions
    t : length of trajectory (aside from the initial state)
    drift_action : optional, specifies the drift action to take
    
    Returns
    -------
    traj : list of length 2, giving state observations of initial state and final state
    """
    x_list = []
    x_list.append(env.reset(state=x0))
    if t==1: # already have the complete trajectory
        x_list.append(env.step(u_list[0])[0])
        return x_list
    env.step(u_list[0])
    for _ in range(t-2):
        env.step(drift_action)
    x_list.append(env.step(u_list[1])[0])
    return x_list
        
def get_drift_batch(env,T,batch_size,action_sampler,state_sampler):
    """
    Get a batch of drift trajectories suitable as input to the loss function.

    Want to return a list of length T, each element is a list of length 4.
    [X0, Xi, U0, Ui]. The first entry is special though, just three entries.
    X0 : ndarray of shape (batch_size,*observation_shape)
    U0 : ndarray of shape (batch_size,*action_shape)
    """
    out = []
    # get the first batch -- special because no drift
    u0_batch = action_sampler(batch_size)
    x0_batch = state_sampler(batch_size)
    batch_traj = [get_trajectory_drift(env,x0_batch[i],[u0_batch[i]],1) for i in range(batch_size)]
    out.append([torch.from_numpy(np.stack([b[0] for b in batch_traj])).float(),
                torch.from_numpy(np.stack([b[1] for b in batch_traj])).float(),
                torch.from_numpy(u0_batch).float()])
    for t in range(T-1): # now get longer trajectories
        u0_batch = action_sampler(batch_size)
        ut_batch = action_sampler(batch_size)
        x0_batch = state_sampler(batch_size)
        batch_traj = [get_trajectory_drift(env,x0_batch[i],[u0_batch[i],ut_batch[i]],t+2) for i in range(batch_size)]
        out.append([torch.from_numpy(np.stack([b[0] for b in batch_traj])).float(),
                    torch.from_numpy(np.stack([b[1] for b in batch_traj])).float(),
                    torch.from_numpy(u0_batch).float(),
                    torch.from_numpy(ut_batch).float()])
    return out
    

class TeacherTrajectorySampler():
    """
    Samples a batch of trajectories, where the start states are produced by a pretrained agent.
    The reason for this is to focus the samples around the good part of the state space.
    
    Parameters
    ----------
    env : restartable gym-style environment
    action_sampler : callable that takes a single input parameter and produces a batch of actions
    teacher_states : list of states produced by a teacher agent in which to start the trajectories
    """
    def __init__(self,env,action_sampler,teacher_states,traj_len,device=torch.device('cpu')):
        self.T = traj_len
        self.state_ind = 0
        self.trajectories = [] # state sequences for each trajectory
        self.actions = [] # action sequences for each trajectory
        self.obs_shape = env.observation_space.shape
        self.act_shape = action_sampler(1).shape[1:]
        self.device = device
        self.teacher_states = teacher_states
        self.env = env
        self.action_sampler = action_sampler
        # compute all the trajectories at once, since we cache and iterate over them in training
        self._create_trajectories()
           
    def _create_trajectories(self):
        self.trajectories = []
        if isinstance(self.teacher_states,int):
            for i in range(self.teacher_states): # number of random initial states
                traj = []
                traj.append(self.env.reset())
                actions = self.action_sampler(self.T)
                for a in actions:
                    traj.append(self.env.step(a)[0])
                self.actions.append(actions)
                self.trajectories.append(traj)
        else:
            for i,state in enumerate(self.teacher_states):
                traj = []
                traj.append(self.env.reset(state=state))
                actions = self.action_sampler(self.T)
                for a in actions:
                    traj.append(self.env.step(a)[0])
                self.actions.append(actions)
                self.trajectories.append(traj)
    
    def get_batch(self,batch_size):
        """
        Sample a batch from the trajectories, return it in the correct format.
        """
        X_list = [np.zeros((batch_size,*self.obs_shape)) for _ in range(self.T+1)]
        U_list = [np.zeros((batch_size,*self.act_shape)) for _ in range(self.T)]
        batch_inds = np.random.randint(len(self.trajectories),size=batch_size)
        for j,i in enumerate(batch_inds):
            for k in range(self.T):
                X_list[k][j,:] = self.trajectories[i][k]
                U_list[k][j,:] = self.actions[i][k]
            X_list[self.T][j,:] = self.trajectories[i][self.T]
        return ([torch.from_numpy(X).float().to(self.device) for X in X_list], 
                [torch.from_numpy(U).float().to(self.device) for U in U_list])  
    
class FixedDatasetSampler():
    """
    Maintains a fixed dataset, samples from this cyclically. 
    """
    def __init__(self,env,device=torch.device('cpu')):
        self.env = env
        self.device = device
    

class PolicyTrajectorySampler():
    """
    Given a policy and environment, sample a batch of trajectories.
    
    Parameters
    ----------
    env : gym-like environment object (needs reset and step methods)
    pol : stable-baselines-like model (needs a predict method to get action)
    T : integer, trajectory length to be gathered
    
    """
    def __init__(self, env, pol, T, device=torch.device('cpu')):
        self.env = env
        self.pol = pol
        self.device = device
        self.T = T
        self.obs_shape = env.observation_space.shape
        self.act_shape = env.action_space.shape
        
    def get_trajectory(self):
        states = []
        actions = []
        x = self.env.reset()
        states.append(x)
        for _ in range(self.T):
            u = self.pol.predict(x)
            x = self.env.step(u)[0]
            actions.append(u)
            states.append(x)
        return states, actions
    
    def get_batch(self,batch_size):
        trajectories = [self.get_trajectory() for _ in range(batch_size)]
        tups = []
        for i in range(self.T):
            X1 = np.empty((batch_size, *self.obs_shape))
            X0 = np.empty((batch_size, *self.obs_shape))
            U = np.empty((batch_size, *self.act_shape))
            for j in range(batch_size):
                X0[j,:] = trajectories[j][0][i]
                X1[j,:] = trajectories[j][0][i+1]
                U[j,:] = trajectories[j][1][i]
            tups.append([torch.from_numpy(T).float().to(self.device) for T in [X1, X0, U]])
        return tups
            
        
        
    
class SimpleTrajectorySampler():
    """
    Samples a batch of trajectories but stores simple (X1,X0,U) tuples.
    
    Parameters
    ----------
    env : restartable gym-style environment
        The environment must also have a _get_state method that returns the internal state, so that it may be reset.
    action_sampler : callable function that takes a single input parameter (batch_size),
        and returns a numpy array (batch_size, *action_shape) compatible with env
    state_sampler : similar to action_sampler, but generates a batch of states that the environment
        can be reset to
    drift_action : optional, specifies what the drift action should be. Defaults to all zeros.
    device : optional, specifies which device to perform computation on (default gpu)
    """
    def __init__(self,env,action_sampler,state_sampler,device=torch.device('cpu'),
                 deterministic=False,
                 deterministic_args = None):
        self.env = env
        self.action_sampler = action_sampler
        self.state_sampler = state_sampler
        self.deterministic=deterministic
        self.obs_shape = env.observation_space.shape
        self.state_shape = state_sampler(1).shape[1:]
        self.act_shape = action_sampler(1).shape[1:]
        self.device = device
        
        if deterministic:
            print("\ngenerating dataset...\n")
            n_samples, batch_size, largeT, repeats = deterministic_args # total samples, size of batch, 
            n_large_batches = int(n_samples/(batch_size*largeT*repeats))
            print("making {} large batches".format(n_large_batches))
            print("total samples used: {}".format(n_large_batches*batch_size*largeT*repeats))
            self.batches = []
            for _ in range(n_large_batches):
                X,U = self.get_new_batch(batch_size,largeT)
                self.batches.extend([(X[i+1],X[i],U[i]) for i in range(len(U))])
            np.random.shuffle(self.batches)
            self.n_batches = len(self.batches)
            print("done with {} batches...\n".format(self.n_batches))
            self.batch_index = 0
 
    
    
    def get_new_batch(self,batch_size,T):
        """
        sample a fresh batch from the environment.
        """
        return self._forward_batch(batch_size,T)
        
        
    def get_batch(self,batch_size,T,method):
        """
        Sample a batch of trajectories.
        
        Parameters
        ----------
        batch_size : positive integer, number of trajectories to sample
        T : positive integer, length of trajectory
        method : string, specifies which type of trajectory to sample
        """
        if not self.deterministic:
            return self.get_new_batch(batch_size,T)
        else:
            batch = self.batches[self.batch_index]
            self.batch_index = (self.batch_index + 1)%self.n_batches
            # maybe reshuffle the data after each pass
            return batch
    
    def _forward_batch(self,batch_size):
        S0 = self.state_sampler(batch_size)
        U = self.action_sampler(batch_size)
        X1 = np.empty((batch_size,*self.obs_shape))
        X0 = np.empty((batch_size,*self.obs_shape))
        for j in range(batch_size):
            X0[j,:] = self.env.reset(state=S0[j])
            X1[j,:] = self.env.step(U[j])[0]
        return [torch.from_numpy(T).float().to(self.device) for T in [X1, X0, U]]
    
    def _full_batch(self,batch_size,T):
        X_list = [np.empty((batch_size,*self.obs_shape)) for _ in range(T+1)]
        X0 = self.state_sampler(batch_size)
        U_list = [self.action_sampler(batch_size) for _ in range(T)]
        for j in range(batch_size):
            X_list[0][j,:] = self.env.reset(state=X0[j])
            for i in range(T):
                X_list[i+1][j,:] = self.env.step(U_list[i][j])[0]
        return ([torch.from_numpy(X).float().to(self.device) for X in X_list], 
                [torch.from_numpy(U).float().to(self.device) for U in U_list])        
          
    
class TrajectorySampler():
    """
    Samples a batch of trajectories.
    
    Parameters
    ----------
    env : restartable gym-style environment
        The environment must also have a _get_state method that returns the internal state, so that it may be reset.
    action_sampler : callable function that takes a single input parameter (batch_size),
        and returns a numpy array (batch_size, *action_shape) compatible with env
    state_sampler : similar to action_sampler, but generates a batch of states that the environment
        can be reset to
    drift_action : optional, specifies what the drift action should be. Defaults to all zeros.
    device : optional, specifies which device to perform computation on (default gpu)
    """
    def __init__(self,env,action_sampler,state_sampler,T,drift_action=None,device=torch.device('cpu'),
                 deterministic=False,
                 deterministic_args = None,
                method="full"):
        self.env = env
        self.action_sampler = action_sampler
        self.state_sampler = state_sampler
        self.deterministic=deterministic
        self.method = method
        self.T = T
        self.traj_method = {"full" : self._full_batch,
                            "full2" : self._full_batch,
                            "drift1" : self._drift1_batch,
                            "drift2" : self._drift2_batch}
        
        self.obs_shape = env.observation_space.shape
        self.state_shape = state_sampler(1).shape[1:]
        self.act_shape = action_sampler(1).shape[1:]
        self.device = device
        if drift_action is None:
            self.drift_action = np.zeros(self.act_shape)
        else:
            self.drift_action = drift_action
        
        if deterministic:
            print("\ngenerating dataset...\n")
            n_samples, batch_size, largeT, method, repeats, T = deterministic_args # total samples, size of batch, 
            n_large_batches = int(n_samples/(batch_size*largeT*repeats))
            print("making {} large batches".format(n_large_batches))
            print("total samples used: {}".format(n_large_batches*batch_size*largeT*repeats))
            self.batches = []
            for _ in range(n_large_batches):
                X,U = self.get_new_batch(batch_size,largeT,method)
                self.batches.extend([(X[i:i+T+1], U[i:i+T]) for i in range(len(U)-T+1)])
            np.random.shuffle(self.batches)
            self.n_batches = len(self.batches)
            print("done with {} batches...\n".format(self.n_batches))
            self.batch_index = 0
 
    
    
    def get_new_batch(self,batch_size):
        """
        sample a fresh batch from the environment.
        """
        return self.traj_method[self.method](batch_size,self.T)
        
        
    def get_batch(self,batch_size):
        """
        Sample a batch of trajectories.
        
        Parameters
        ----------
        batch_size : positive integer, number of trajectories to sample
        T : positive integer, length of trajectory
        method : string, specifies which type of trajectory to sample
        """
        if not self.deterministic:
            return self.get_new_batch(batch_size)
        else:
            batch = self.batches[self.batch_index]
            self.batch_index = (self.batch_index + 1)%self.n_batches
            return batch
    
    def _full_batch(self,batch_size,T):
        X_list = [np.empty((batch_size,*self.obs_shape)) for _ in range(T+1)]
        X0 = self.state_sampler(batch_size)
        U_list = [self.action_sampler(batch_size) for _ in range(T)]
        for j in range(batch_size):
            X_list[0][j,:] = self.env.reset(state=X0[j])
            for i in range(T):
                X_list[i+1][j,:] = self.env.step(U_list[i][j])[0]
        return ([torch.from_numpy(X).float().to(self.device) for X in X_list], 
                [torch.from_numpy(U).float().to(self.device) for U in U_list])        
    
    def _drift1_batch(self,batch_size,T):
        
        X_list = [np.empty((batch_size,*self.obs_shape)) for _ in range(T+1)]
        X_drift_list = [np.empty((batch_size,*self.state_shape)) for _ in range(T+1)]
        X0 = self.state_sampler(batch_size)
        U0 = self.action_sampler(batch_size)
        U1 = self.action_sampler(batch_size)
        
        for j in range(batch_size):
            X_list[0][j,:] = self.env.reset(state=X0[j])
            X_drift_list[0][j,:] = X0[j]
            X_list[1][j,:] = self.env.step(U0[j])[0] # always take this first action
            X_drift_list[1][j,:] = self.env._get_state()
            for i in range(T-1):
                X_list[i+2][j,:] = self.env.step(U1[j])[0]
                self.env.reset(state = X_drift_list[i+1][j,:]) # reset to the  drift state previous
                self.env.step(self.drift_action)
                X_drift_list[i+2][j,:] = self.env._get_state() # drift to get the new drift state

        return ([torch.from_numpy(X).float().to(self.device) for X in X_list], 
                [torch.from_numpy(U).float().to(self.device) for U in [U0,U1]])
    
    def _drift2_batch(self,batch_size,T):
        """
        Get a batch of drift trajectories suitable as input to the loss function.
        This is appropriate for the drift2 loss.
        
        Returns
        -------
        List of length T, each element is a list of length 4 : [X0, Xi, U0, Ui]
            The first entry is special though, just three entries.
            X0 : ndarray of shape (batch_size,*observation_shape)
            U0 : ndarray of shape (batch_size,*action_shape)
            (similar Xi, Ui)
        """
        out = []
        # get the first batch -- special because no drift
        u0_batch = self.action_sampler(batch_size)
        x0_batch = self.state_sampler(batch_size)
        batch_traj = [get_trajectory_drift(self.env,x0_batch[i],[u0_batch[i]],1) for i in range(batch_size)]
        out.append([torch.from_numpy(np.stack([b[0] for b in batch_traj])).float().to(self.device),
                    torch.from_numpy(np.stack([b[1] for b in batch_traj])).float().to(self.device),
                    torch.from_numpy(u0_batch).float().to(self.device)])
        for t in range(T-1): # now get longer trajectories
            u0_batch = self.action_sampler(batch_size)
            ut_batch = self.action_sampler(batch_size)
            x0_batch = self.state_sampler(batch_size)
            batch_traj = [get_trajectory_drift(self.env,x0_batch[i],[u0_batch[i],ut_batch[i]],
                                               t+2,drift_action=self.drift_action) 
                          for i in range(batch_size)]
            out.append([torch.from_numpy(np.stack([b[0] for b in batch_traj])).float().to(self.device),
                        torch.from_numpy(np.stack([b[1] for b in batch_traj])).float().to(self.device),
                        torch.from_numpy(u0_batch).float().to(self.device),
                        torch.from_numpy(ut_batch).float().to(self.device)])
        return out


class EncoderNet(nn.Module):
    """
    Basic feedforward ReLu network. Nothing fancy going on here.
    """
    def __init__(self, widths):
        super(EncoderNet, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(widths[i],widths[i+1]) for i in range(len(widths)-1)])
        
    def forward(self, x):
        z = x
        for layer in self.layers:
            z = F.relu(layer(z))
        return z

class SoftmaxNet(nn.Module):
    def __init__(self, widths):
        super(SoftmaxNet, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(widths[i],widths[i+1]) for i in range(len(widths)-1)])
        
    def forward(self, x):
        z = x
        for layer in self.layers[:-1]:
            z = F.relu(layer(z))
        z = nn.Softmax(dim=-1)(self.layers[-1](z))
        return z    

    
class ConvEncoderNet(nn.Module):
    """
    Create a PyTorch module that implements a convolutional network (2 conv layers, followed by some fc layers).

    Parameters
    ----------
    widths : list of ints, gives sizes of fully connected layers following the conv layers
    input_shape : tuple of positive integers, gives shape (height, width) of the input image 
    """
    def __init__(self, widths, input_shape,n_channels=1, sigma = torch.tanh):

        
        # input_shape just gives height and widths of input image, not channels
        super(ConvEncoderNet, self).__init__()
        
        # nonlinearity
        self.sigma = sigma 
        
        # create convolutional layers
        self.conv1 = nn.Conv2d(n_channels,16,8,stride=4)
        #self.conv1.requires_grad = False
        
        self.conv2 = nn.Conv2d(16,16,4,stride=2)
        #self.conv2.requires_grad = False
        
        # get shape of output after convolutional layers
        shape = [self._compute_new_size(s,8,4) for s in input_shape]
        shape = [self._compute_new_size(s,4,2) for s in shape]
        self.size = 16 * shape[0] * shape[1]
        
        # create fc layers following convolutional layers
        widths = [self.size]+widths
        self.layers = nn.ModuleList([nn.Linear(widths[i],widths[i+1]) for i in range(len(widths)-1)])
        #for l in self.layers:
            #l.requires_grad = False
        
    def _compute_new_size(self, s_in, kern, stride):
        return int(np.floor((s_in-kern)/stride+1))
    
    def forward(self, x):
        z = self.sigma(self.conv1(x))
        z = self.sigma(self.conv2(z))
        z = z.view(-1, self.size)
        for layer in self.layers[:-1]:
            z = self.sigma(layer(z))
           
        return self.layers[-1](z)
    

class MixtureForwardNet(nn.Module):
    def __init__(self, encoder, enc_dim, act_dim, k, mixer, fit_reward=False,mu=0, r_encoder = None, alpha=1, mean_coeff=0,covar_coeff=0,normalize=False):
        super(MixtureForwardNet, self).__init__()
        self.encoder = encoder
        self.act_dim = act_dim
        self.enc_dim = enc_dim
        self.k = k
        self.Alist = nn.ModuleList([nn.Linear(enc_dim,enc_dim,bias=False) for i in range(k)])
        self.Blist = nn.ModuleList([nn.Linear(act_dim,enc_dim,bias=False) for i in range(k)])
        self.mixer = mixer
        self.fit_reward = fit_reward
        self.mu = mu
        self.alpha = alpha
        self.r_encoder = r_encoder
        self.covar_coeff=covar_coeff
        self.mean_coeff=mean_coeff
        self.normalize = normalize
    
    def regularizer(self,batch):
        if self.fit_reward:
            X1, X0, U, R = batch
        else:
            X1, X0, U = batch
        X0 = self.encoder(X0)
        batch_size = X0.shape[0]
        mean_loss = (torch.mean(X0,0)**2).sum()
        covar_loss = ((torch.matmul(X0.T, X0)/batch_size - torch.eye(self.enc_dim))**2).sum()
        return self.mean_coeff * mean_loss + self.covar_coeff * covar_loss
    
    def forward_loss(self,batch):
        if self.fit_reward:
            X1, X0, U, R = batch
        else:
            X1, X0, U = batch
        X1 = self.encoder(X1)
        X0 = self.encoder(X0)
        batch_size = X0.shape[0]
        # batch * k
        coeffs = self.mixer(X0)
        #print(coeffs)
        # batch * k * dim
        outputs = torch.cat([(A(X0)+B(U))[:,None,:] for (A,B) in zip(self.Alist,self.Blist)],dim=1)
        # batch * dim
        #print(coeffs.shape, outputs.shape)
        #outputs = outputs[:,:,None]
        coeffs = coeffs[:,None,:]
        #print(coeffs.shape, outputs.shape)
        pred = torch.matmul(coeffs, outputs)[:,0,:]
        loss = 0
        #mean_loss = (torch.mean(X0,0)**2).sum()
        #covar_loss = ((torch.matmul(X0.T, X0)/batch_size - torch.eye(self.enc_dim))**2).sum()
        if self.normalize:
            norm_factor = (torch.inverse(torch.matmul(X0.T, X0)/batch_size))
            unnorm_err = X1-pred 
            #print(X1.shape, pred.shape, unnorm_err.shape, norm_factor.shape)
            loss += self.alpha*(torch.mm(torch.mm(unnorm_err,norm_factor), unnorm_err.T)).sum()    
        else:
            loss += self.alpha*((X1 - pred)**2).sum()
        if self.fit_reward:
            r_input = torch.cat([X0,X1,U], dim=1)
            pred_rs = self.r_encoder(r_input)
            R = R[:,None]
            loss += self.mu*((R - pred_rs)**2).sum()
        loss /= batch_size #(self.enc_dim*batch_size)
        #loss += self.mean_coeff * mean_loss + self.covar_coeff * covar_loss
        return loss
    
    
class MixtureBNForwardNet(nn.Module):
    def __init__(self, encoder, enc_dim, act_dim, k, mixer, fit_reward=False,mu=0, r_encoder = None, alpha=1):
        super(MixtureBNForwardNet, self).__init__()
        self.encoder = encoder
        self.act_dim = act_dim
        self.enc_dim = enc_dim
        self.k = k
        self.Alist = nn.ModuleList([nn.Linear(enc_dim,enc_dim,bias=False) for i in range(k)])
        self.Blist = nn.ModuleList([nn.Linear(act_dim,enc_dim,bias=False) for i in range(k)])
        self.mixer = mixer
        self.fit_reward = fit_reward
        self.mu = mu
        self.alpha = alpha
        self.r_encoder = r_encoder
        self.bn1 = nn.BatchNorm1d(enc_dim)
        self.bn2 = nn.BatchNorm1d(enc_dim)
    
    def forward_loss(self,batch):
        if self.fit_reward:
            X1, X0, U, R = batch
        else:
            X1, X0, U = batch
        X1 = self.bn1(self.encoder(X1))
        X0 = self.bn2(self.encoder(X0))
        # is it ok for these two bn layers to be shared?
        batch_size = X0.shape[0]
        # batch * k
        coeffs = self.mixer(X0)
        #print(coeffs)
        # batch * k * dim
        outputs = torch.cat([(A(X0)+B(U))[:,None,:] for (A,B) in zip(self.Alist,self.Blist)],dim=1)
        # batch * dim
        #print(coeffs.shape, outputs.shape)
        pred = torch.matmul(coeffs, outputs)
        loss = 0
        loss += self.alpha*((X1 - pred)**2).sum()
        if self.fit_reward:
            r_input = torch.cat([X0,X1,U], dim=1)
            pred_rs = self.r_encoder(r_input)
            R = R[:,None]
            loss += self.mu*((R - pred_rs)**2).sum()
        loss /= batch_size #(self.enc_dim*batch_size)
        return loss    
    
    
    
class ForwardInverseNet(nn.Module):
    def __init__(self,encoder,enc_dim, act_dim):

        super(ForwardInverseNet, self).__init__()
        self.encoder = encoder

        self.act_dim = act_dim
        self.enc_dim = enc_dim
        self.A = nn.Linear(enc_dim,enc_dim,bias=False) # drift matrix, maps state to state
        self.B = nn.Linear(act_dim,enc_dim,bias=False) # control matrix, maps action to state
        self.P = nn.Linear(enc_dim,act_dim,bias=False) # projection matrix, maps state to action
        self.L = nn.Linear(enc_dim,act_dim,bias=False) # projection matrix, state to action

    def forward_loss(self,batch):
        """
        batch : list/tuple of 3 tensors, corresponding to X1, X0, and U
        """
        X1, X0, U = batch
        X1 = self.encoder(X1)
        X0 = self.encoder(X0)
        state_pred = self.A(X0) + self.B(U)
        return (((X1-state_pred)**2).sum()/self.enc_dim)/X1.shape[0]
    
    def inverse_loss(self,batch):
        X1, X0, U = batch
        X1 = self.encoder(X1)
        X0 = self.encoder(X0)
        act_pred = self.P(X1) - self.L(X0)
        return (((U-act_pred)**2).sum()/self.act_dim)/X1.shape[0]

class ForwardNet(nn.Module):
    def __init__(self,encoder,enc_dim, act_dim, fixed_B = False):

        super(ForwardNet, self).__init__()
        self.encoder = encoder
        self.fixed_B = fixed_B
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.act_dim = act_dim
        self.enc_dim = enc_dim
        self.A = nn.Linear(enc_dim,enc_dim,bias=False) # drift matrix, maps state to state
        if fixed_B:
            self.B = torch.eye(enc_dim)[:act_dim,:]
        else:
            self.B = nn.Linear(act_dim,enc_dim,bias=False)
        
    def loss(self,batch):
        return self.forward_loss(batch)
    
    def forward_loss(self,batch):
        """
        batch : list/tuple of 3 tensors, corresponding to X1, X0, and U
        """
        X1, X0, U = batch
        X1 = self.encoder(X1)
        X0 = self.encoder(X0)
        if self.fixed_B:
            state_pred = self.A(X0) + torch.matmul(U,self.B)
        else:
            state_pred = self.A(X0) + self.B(U)
        return (((X1-state_pred)**2).sum()/self.enc_dim)/X1.shape[0]
       

class PiecewiseForwardNet(nn.Module):
    def __init__(self, encoder, enc_dim, act_dim, k):
        super(PiecewiseForwardNet, self).__init__()
        self.encoder = encoder
        self.act_dim = act_dim
        self.enc_dim = enc_dim
        self.k = k
        self.A0 = nn.Linear(enc_dim,enc_dim,bias=False)
        self.B0 = torch.eye(enc_dim)[:act_dim,:]
        self.Alist = nn.ModuleList([nn.Linear(enc_dim,enc_dim,bias=False) for i in range(k-1)])
        self.Blist = nn.ModuleList([nn.Linear(act_dim,enc_dim,bias=False) for i in range(k-1)])
        self.C = nn.Linear(enc_dim,k) # used to decide which linear model is active
    
    def forward_loss(self,batch):
        X1, X0, U = batch
        X1 = self.encoder(X1)
        X0 = self.encoder(X0)
        inds = torch.argmax(self.C(X0),axis=1)
        loss = 0
        class_inds = inds==0
        if True in class_inds:
            pred = self.A0(X0[class_inds]) + torch.matmul(U[class_inds],self.B0)
            loss += ((X1[class_inds] - pred)**2).sum()
        for i in range(self.k-1):
            class_inds = inds==i+1
            if True in class_inds:
                pred = self.Alist[i](X0[class_inds]) + self.Blist[i](U[class_inds])
                loss += ((X1[class_inds] - pred)**2).sum()
        return loss/(self.enc_dim*X1.shape[0])
                
class MultiStepForward(nn.Module):
    def __init__(self, encoder, t, enc_dim, act_dim):
        """
        Parameters
        ----------
        t : integer, length of the trajectory
        encoder : state representation encoder
        enc_dim, act_dim : dimensions of representation and action
        """
        super(MultiStepForward, self).__init__()
        self.encoder = encoder
        self.t = t
        self.T = nn.ModuleList([nn.Linear(enc_dim,enc_dim,bias=False) for i in range(t)])
        self.L = nn.ModuleList([nn.Linear(act_dim,enc_dim,bias=False) for i in range(t)])
        
        # initiaze L_0 and fix it
        with torch.no_grad():
            self.L[0].weight = torch.nn.Parameter(torch.eye(enc_dim)[:,:act_dim])
        self.L[0].requires_grad_(False)
            
        
        
    
    def loss(self, batch):
        """
        batch is a list of matrices whose columns give the x0s, x1s, ..., u0s, u1s...
        batch = [[X0, X1, ..., Xt], [U0, U1, ..., Ut-1]]
        """
        X_list, U = batch
        
        # edit the following
        enc_X_list = [self.encoder(X) for X in X_list] # encode the states Xs
        loss = 0
        for n in range(1,self.t+1):
            pred = self.T[n-1](enc_X_list[0])
            for i in range(n):
                pred += self.L[n-i-1](U[i])
            loss += ((pred - enc_X_list[n])**2).mean()
        return loss/self.t
        
        
class PredictorNet(nn.Module):
    """
    PyTorch Module object that implements the loss functions for state representation learning.
    Learnable parameters include an encoder (maps original state observations to representation),
    as well as a list of state and action projector matrices.

    Parameters
    ----------
    encoder : PyTorch Module that encodes state observations
    T : positive integer, length of trajectory for the learning objective
    enc_dim : positive integer, dimension of the state encoding
    act_dim : positive integer, dimension of the action space
    """
    def __init__(self,encoder,T, enc_dim, act_dim):

        super(PredictorNet, self).__init__()
        self.encoder = encoder
        self.T = T # length of the trajectory
        self.state_projectors = nn.ModuleList([nn.Linear(enc_dim,act_dim,bias=False) for i in range(T+1)])
        self.action_projectors = nn.ModuleList([nn.Linear(act_dim,act_dim,bias=False) for i in range(T-1)])
        self.loss_methods = {"full" : self._full_loss,
                             "full2" : self._full2_loss,
                             "drift1" : self._drift1_loss,
                             "drift2" : self._drift2_loss}
    
    def loss(self,traj_batch,method):
        return self.loss_methods[method](traj_batch)
    
    def _full_loss(self, traj_batch):
        """
        Parameters
        ----------
        traj_batch: tuple consisting of the following two parameters
            X_list : length T+1 list of tensors, each of shape (N,state_dim) (N is batch size)
                The ith tensor in the list gives the batch of ith states in the trajectories
            U_list : length T list of tensors, each of shape (N,act_dim)
                The ith tensor in the list gives the batch of ith actions in the trajectories
            
        Returns
        -------
        loss : scalar, the average squared error (over all coordinates of the batch, and over the trajectory)
        """
        X_list, U_list = traj_batch
        enc_X_list = [self.encoder(X) for X in X_list]
        loss = 0
        for i in range(self.T):
            pred = self.state_projectors[-1](enc_X_list[i+1])
            pred -= self.state_projectors[i](enc_X_list[0])
            for j in range(i): # previously had range(i-1) which seemed off
                pred -= self.action_projectors[i-j-1](U_list[j])
            loss += ((pred - U_list[i])**2).mean()
        return loss/self.T
    
    def _full2_loss(self, traj_batch):
        """
        Parameters
        ----------
        traj_batch: tuple consisting of the following two parameters
            X_list : length T+1 list of tensors, each of shape (N,state_dim) (N is batch size)
                The ith tensor in the list gives the batch of ith states in the trajectories
            U_list : length T list of tensors, each of shape (N,act_dim)
                The ith tensor in the list gives the batch of ith actions in the trajectories
            
        Returns
        -------
        loss : scalar, the average squared error (over all coordinates of the batch, and over the trajectory)
        """
        X_list, U_list = traj_batch
        enc_X_list = [self.encoder(X) for X in X_list]
        loss = 0
        for k in range(self.T): # let k index the starting point of the trajectory
            
            for i in range(self.T-k): # i is offset from starting point k
                pred = self.state_projectors[-1](enc_X_list[k+i+1])
                pred -= self.state_projectors[i](enc_X_list[k])
                for j in range(i): # previously had range(i-1) which seemed off
                    pred -= self.action_projectors[i-j-1](U_list[j+k])
                loss += ((pred - U_list[i+k])**2).mean()
        return loss/((self.T+1)*(self.T)/2)
    
    def _drift1_loss(self, traj_batch):
        """
        Parameters
        ----------
        traj_batch : tuple consisting of following two parameters
            X_list : length T+1 list of tensors, each of shape (N,state_dim) (N is batch size)
                The ith tensor in the list gives the batch of ith states in the trajectories
            U_list : length 2 list of tensors, each of shape (N,act_dim)
                The ith tensor in the list gives the batch of ith actions in the trajectories
            
        Returns
        -------
        loss : scalar, the average squared error (over all coordinates of the batch, and over the trajectory)
        """
        X_list, U_list = traj_batch
        enc_X_list = [self.encoder(X) for X in X_list]
        loss = 0
        for i in range(self.T):
            pred = self.state_projectors[-1](enc_X_list[i+1])
            pred -= self.state_projectors[i](enc_X_list[0])
            if i > 0: # previously I didn't have this check
                pred -= self.action_projectors[i-1](U_list[0])
            loss += ((pred - U_list[1])**2).mean()
        return loss/self.T
    
    def _drift2_loss(self,traj):
        """
        Parameters
        ----------
        traj : list of length T of lists
            Each entry list consists of four tensors: X0, Xi, U0, Ui
            The first two tensors are of shape (batch_size,*observation_shape)
            The last two tensors are of shape (batch_size,*action_shape)
            
        Returns
        -------
        loss : scalar, the average squared error
        """
        loss = 0
        
        # the loss on a trajectory of length 1 is a special case: use x0 and x1 to predict u0
        X0, X1, U0 = traj[0]
        X0 = self.encoder(X0)
        X1 = self.encoder(X1)
        pred = self.state_projectors[-1](X1)
        pred -= self.state_projectors[0](X0)
        loss += ((pred - U0)**2).mean()
        
        # now get the losses for longer trajectories
        for i in range(1,self.T):
            X0, X1, U0, U1 = traj[i]
            X0 = self.encoder(X0)
            X1 = self.encoder(X1)
            pred = self.state_projectors[-1](X1)
            pred -= self.state_projectors[i](X0)
            pred -= self.action_projectors[i-1](U0)
            loss += ((pred - U1)**2).mean()
        return loss/self.T 
            
        
    
def train_encoder(predNet, # pytorch Module-style object, encodes states and computes loss
                  traj_sampler, # object that produces batches of trajectories
                  n_episodes, # how many batches of trajectories to train on
                  lr=1e-3, #learning rate for optimizer
                  batch_size = 50,
                  show_progress=True,
                  track_loss_every=10, # print a progress statement every _ batches
                  save_every=None,
                  save_path=None,
                  use_teacher=False,
                  passes=None, # either an integer (number of passes to iterate over cached trajectories) or None
                  refresh_trajectories_every=np.inf, # specifies how often to create new trajectories from the teacher states
                 ):
    
    if save_every is None:
        save_every = n_episodes
    
    env = traj_sampler.env
    angs = np.linspace(-np.pi, np.pi, 5)[:-1] # test angles
    vels = np.linspace(-8, 8, 5) # test velocities
    test_states = np.array(list(itertools.product(angs,vels)))
    X0 = np.empty((len(test_states),*(env.observation_space.shape)))
    for j in range(len(test_states)):
        X0[j,:] = env.reset(state=test_states[j])
    # initialize optimizer
    optimizer = optim.Adam(predNet.parameters(),lr=lr)
    losses = []
    running_loss = 0.0
    n_passes = 0
    
    if passes is not None:
        cached_trajectories = []
        n_passes = passes
    
    for i in range(n_episodes):
        # generate a batch of trajectories
        if not use_teacher:
            traj_batch = traj_sampler.get_batch(batch_size)
            #traj_batch = traj_sampler._forward_batch(batch_size)
        else:
            if (i+1)%refresh_trajectories_every==0:
                traj_sampler._create_trajectories() # create a new set of trajectories from teacher states
            traj_batch = traj_sampler.get_batch(batch_size)

        if passes is not None:
            cached_trajectories.append(traj_batch)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        #floss = predNet.forward_loss(traj_batch,loss_method)
        #iloss = predNet.inverse_loss(traj_batch,loss_method)
        #lam = .01 # .05 worked a couple times
        #loss = lam*floss + (1-lam)*iloss
        #loss = predNet.forward_loss(traj_batch)
        #loss = predNet._full_loss(traj_batch)
        loss = predNet.loss(traj_batch)
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if ((i+1)%track_loss_every) == 0:
            #T_norm = la.norm(predNet.T[0].weight.detach().numpy())
            #norms = [la.norm(p.detach().numpy()) for p in predNet.parameters()][-1]
            
            X_enc = predNet.encoder(torch.from_numpy(X0).float()).detach().numpy()
            cov = np.dot(X_enc.T,X_enc)


            if show_progress:
                print("Epoch Completion: {0:.3f}%, Loss: {1:.5f}, mean: {2:.4f}, top_singular: {3:.4f}, L0: {4:.3f}".format(
                    100*(i+1)/n_episodes,
                    running_loss/track_loss_every,
                    X_enc.mean(),
                    la.svdvals(cov)[0],
                    predNet.L[0].weight.detach().numpy().sum()))
                     
            losses.append(running_loss/track_loss_every)
            running_loss = 0.0
            
        if ((i+1)%save_every) == 0:
            torch.save(predNet,save_path+"_{}net".format((i+1)/save_every))
            
                    
                 
                    
    return predNet, losses

def train_encoder_policy(predNet, # pytorch Module-style object, encodes states and computes loss
                          traj_sampler, # object that produces batches of trajectories
                          n_episodes, # how many batches of trajectories to train on
                          #POL, # stable baselines policy network
                          #ALG, # stable baselines policy algorithm
                          #alg_kwargs,
                          #pol_kwargs,
                     
                          model,
                          pol_timesteps, # how many steps to learn the policy in each iteration
                          outer_loop,
                          lr=1e-3, #learning rate for optimizer
                          batch_size = 50,
                          show_progress=True,
                          track_loss_every=10, # print a progress statement every _ batches
                          save_every=None,
                          save_path=None,
                          use_teacher=False,
                          passes=None, # either an integer (number of passes to iterate over cached trajectories) or None
                          refresh_trajectories_every=np.inf, # specifies how often to create new trajectories from the teacher states
                         ):
    
    if save_every is None:
        save_every = n_episodes
    
    env = traj_sampler.env
    # initialize optimizer
    optimizer = optim.Adam(predNet.parameters(),lr=lr)
    losses = []
    
    for _ in range(outer_loop): # first do some encoder training steps, then train the policy for some steps
    

        running_loss = 0.0

        for i in range(n_episodes):
            # generate a batch of trajectories
            traj_batch = traj_sampler.get_batch(batch_size)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            #floss = predNet.forward_loss(traj_batch,loss_method)
            #iloss = predNet.inverse_loss(traj_batch,loss_method)
            #lam = .01 # .05 worked a couple times
            #loss = lam*floss + (1-lam)*iloss
            #loss = predNet.forward_loss(traj_batch)
            #loss = predNet._full_loss(traj_batch)
            loss = predNet.loss(traj_batch,"full")

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if ((i+1)%track_loss_every) == 0:
                #T_norm = la.norm(predNet.T[0].weight.detach().numpy())
                #norms = [la.norm(p.detach().numpy()) for p in predNet.parameters()][-1]

                


                if show_progress:
                    print("Epoch Completion: {0:.3f}%, Loss: {1:.5f}".format(
                        100*(i+1)/n_episodes,
                        running_loss/track_loss_every),end="\r",flush=True)

                losses.append(running_loss/track_loss_every)
                running_loss = 0.0

            if ((i+1)%save_every) == 0:
                torch.save(predNet,save_path+"_{}net".format((i+1)/save_every))
            
                    
        # initialize env with latest encoder
        def make_policy_env():
            repeats = 3
            pol_env = RestartablePendulumEnv(repeats=repeats,pixels=True) # can specify cost="dm_control"
            pol_env = TimeLimit(pol_env,max_episode_steps=int(200/repeats)) # only run the environment for 200 true steps
            proj = [p.data.numpy() for p in predNet.state_projectors.parameters()]
            proj = np.row_stack(proj)
            svd = la.svd(proj, full_matrices=False)
            proj = svd[2][:5]
            return ew.TorchEncoderWrapper(pol_env,predNet.encoder,proj)
        
        pol_env = DummyVecEnv([make_policy_env])
        

        model.env = pol_env
        model.learn(total_timesteps=pol_timesteps)
                    
    return predNet, losses

def train_encoder3(predNet, # pytorch Module-style object, encodes states and computes loss
                  traj_sampler, # object that produces batches of trajectories
                  n_episodes, # how many batches of trajectories to train on
                  lr=1e-3, #learning rate for optimizer
                  batch_size = 50,
                  show_progress=True,
                  track_loss_every=10, # print a progress statement every _ batches
                  save_every=None,
                  save_path=None,
                  use_teacher=False,
                  passes=None, # either an integer (number of passes to iterate over cached trajectories) or None
                  refresh_trajectories_every=np.inf, # specifies how often to create new trajectories from the teacher states
                 ):
    
    if save_every is None:
        save_every = n_episodes
    
    # initialize optimizer
    optimizer = optim.Adam(predNet.parameters(),lr=lr)
    losses = []
    running_loss = 0.0
    n_passes = 0
    
    if passes is not None:
        cached_trajectories = []
        n_passes = passes
    
    for i in range(n_episodes):
        # generate a batch of trajectories
        if not use_teacher:
            #traj_batch = traj_sampler.get_batch(batch_size,T,traj_method)
            traj_batch = traj_sampler._forward_batch(batch_size)
        else:
            if (i+1)%refresh_trajectories_every==0:
                traj_sampler._create_trajectories() # create a new set of trajectories from teacher states
            traj_batch = traj_sampler.get_batch(batch_size)

        if passes is not None:
            cached_trajectories.append(traj_batch)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        #floss = predNet.forward_loss(traj_batch,loss_method)
        #iloss = predNet.inverse_loss(traj_batch,loss_method)
        #lam = .01 # .05 worked a couple times
        #loss = lam*floss + (1-lam)*iloss
        fwd = predNet.forward_loss(traj_batch)
        reg = predNet.regularizer(traj_batch)
        loss = fwd+reg
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += fwd.item()
        #running_floss += floss.item()
        #running_iloss += iloss.item()
        if ((i+1)%track_loss_every) == 0:
            norms = [la.norm(p.detach().numpy()) for p in predNet.parameters()][-1]
            if show_progress:
                print("Epoch Completion: {0:.3f}%, forward loss: {1:.3f}".format(100*(i+1)/n_episodes, running_loss/track_loss_every),
                      end="\r",flush=True)
            losses.append(running_loss/track_loss_every)
            running_loss = 0.0
            
        if ((i+1)%save_every) == 0:
            torch.save(predNet,save_path+"_{}net".format((i+1)/save_every))
            
                    
                 
                    
    return predNet, losses
def train_encoder2(predNet, # pytorch Module-style object, encodes states and computes loss
                   traj_sampler, # object that produces batches of trajectories
                   n_episodes, # how many batches of trajectories to train on
                   optimizer,
                   lr=1e-3, #learning rate for optimizer
                   batch_size = 50,
                   show_progress=True,
                   track_loss_every=10, # print a progress statement every _ batches
                   save_every=None,
                   save_path=None
                  ):
    
    if save_every is None:
        save_every = n_episodes
    
    losses = []
    running_loss = 0.0
    n_passes = 0
    
    for i in range(n_episodes):
        # generate a batch of trajectories
        traj_batch = traj_sampler._forward_batch(batch_size)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        loss = predNet.forward_loss(traj_batch)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        """
        # show a visualization of the representation
        # plot a bunch of points with 0 angular velocity?
        d = 5 # must match the final entry in layers
        n_samps = 500
        env = ew.TorchEncoderWrapper(env,net.encoder,np.eye(d))
        X = np.empty((n_samps,d))
        for i,ang in enumerate(np.linspace(0,2*np.pi,n_samps)): # go through the angles from 0 to 2pi
            X[i,:] = env.reset(state=[ang,0])
        utils.visualize_trajectory(X)
        """

        if ((i+1)%track_loss_every) == 0:
            if show_progress:
                print("Epoch Completion: {0:.3f}%, Loss: {1:.3f}".format(100*(i+1)/n_episodes, running_loss/track_loss_every),
                      end="\r",flush=True)
            losses.append(running_loss/track_loss_every)
            running_loss = 0.0
            
        if ((i+1)%save_every) == 0:
            torch.save(predNet,save_path+"_{}net".format((i+1)/save_every))
            
                    
                 
                    
    return predNet, losses