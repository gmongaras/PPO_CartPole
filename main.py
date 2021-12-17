import gym
import numpy as np
from PPO import Player




if __name__ == '__main__':
    env = gym.make('CartPole-v0') # Our environemnt

    # Hyperparameters
    T = 128                      # The horizon or total time per batch
    stepSize_start = 0.00025     # The starting Adam optimizer step size
                                 # this will be updated as alpha updates
    numEpochs = 3                # The total number of epochs
    numActors = 8                # (N) The total number of different actors to use
    batchSize = 32//numActors    # The size of each batch to run all actors
    #minibatchSize = 32*numActors # The size of each minibatch
    gamma = 0.99                 # The discount rate
    Lambda = 0.95                # The GAE parameter
    epsilon_start = 0.1          # The clipping paramter. This value will
                                 # be updated as alpha updates
    alpha = 1                    # Starting value of lambda which will update
    c1 = 1                       # The VF coefficient in the Loss
    c2 = 0.01                    # The entropy coefficient in the Loss
    numIters = 10000             # The number of times to iterate the entire program
    
    
    # Setup the observation space
    #observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    # Create a player
    player = Player(env.observation_space.shape, env.action_space.n, Lambda=Lambda, gamma=gamma, numActors=numActors, T=T, c1=c1, c2=c2)
    
    
    # Iterate for numIters times
    for iteration in range(1, numIters):
        # Iterate over all actors
        for actor in range(1, numActors):
            # Reset the environment variables
            observation = env.reset()
            
            # Update the hyperparameters
            alpha = 1-(iteration*actor)/(numIters*numActors)
            stepSize = stepSize_start*alpha
            epsilon = epsilon_start*alpha
            
            # # Run all actors in the environment for T timesteps
            # for t in range(1, T):
            #     # Render the environment
            #     env.render()
            
            # Run the models for T timesteps and save the results to memory
            player.runPolicy(env, observation, T)
        
            # Optimize the models
            player.updateModels(alpha=alpha, stepSize=stepSize, epsilon=epsilon)
        
            # Reset the memory
            player.resetMemory()