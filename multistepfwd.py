import sys
sys.path.insert(1, '/home/home3/abef/research/control/LinearizingStateRepresentation/')

from lib.restartable_pendulum import RestartablePendulumEnv
from lib import state_rep_torch as srt
import gym
import numpy as np
from matplotlib import pyplot as plt
import itertools
import sys
import torch

def main():
    
    for arg in sys.argv:
        if arg.startswith('--job='):
            i = int(arg.split('--job=')[1])
    
    # specify environment information
    env = RestartablePendulumEnv(repeats=3,pixels=True)

    
    # specify training details to loop over
    archs = [[50],[100],[50,50],[100,100]] # specifies the fully-connected layers of the encoder
    traj_lens = [10] # specifies trajectory length
    lrs = [.0001, .0005, .001, .005] # learning rate
    param_lists = [archs, traj_lens, lrs]
    
    
    tup = list(itertools.product(*param_lists))[i]
    
    
    tup = [[50,50], 7,.001] # this just hardcodes the hyperparameters
    
    parameters = {
        "n_episodes" :80000,
        "batch_size" : 25, # was 50...
        "learning_rate" : tup[2],
        "widths" : tup[0],
        "traj_len" : tup[1],
    }

    layers = parameters["widths"]
    T = parameters["traj_len"]
    save_path = "./multifwd"
    n_episodes = parameters["n_episodes"]
    batch_size = parameters["batch_size"]
    learning_rate = parameters["learning_rate"]
    save_every = int(n_episodes/5)


    encnet = srt.ConvEncoderNet(layers,env.observation_space.shape[1:])
    prednet = srt.MultiStepForward(encnet,T,layers[-1],1)
    traj_sampler = srt.TrajectorySampler(env,
                                         srt.sample_pendulum_action_batch,
                                         srt.sample_pendulum_state_batch_old,
                                         T,
                                         device=torch.device("cpu"))

    
    net, losses = srt.train_encoder(prednet,traj_sampler,n_episodes,
                                batch_size=batch_size,
                                track_loss_every=int(n_episodes/100),
                                    lr=learning_rate,
                                   save_every=save_every,
                                   save_path=save_path)

    

    
    
    
    torch.save(net,save_path+"net")
    
    # save the training params
    with open(save_dir + "train_params.txt","w") as f:
        for tup in parameters.items():
            f.write(" ".join([str(v) for v in tup]))
            f.write("\n")


    np.savetxt(save_dir+"losses.txt",np.array(losses))
    plt.plot(losses)
    plt.savefig(save_dir + "losses.png")
    plt.clf()
        

if __name__ == '__main__':
    
    main()