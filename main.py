import gym
import numpy as np
from PPO import Player
import torch




if __name__ == '__main__':
    env = gym.make('CartPole-v0') # Our environemnt

    # Hyperparameters
    T = 128                      # The horizon or total time per batch
    stepSize_start = 0.00025     # The starting Adam optimizer step size
                                 # this will be updated as alpha updates
    numEpochs = 3                # The total number of epochs
    numActors = 8                # (N) The total number of different actors to use
    batchSize = 32//numActors    # The size of each batch to run all actors
    gamma = 0.99                 # The discount rate
    Lambda = 0.95                # The GAE parameter
    epsilon_start = 0.1          # The clipping paramter. This value will
                                 # be updated as alpha updates
    alpha = 0.001                # Starting value of the larning rate which
                                 # will decrease as the model updates
    c1 = 1                       # The VF coefficient in the Loss
    c2 = 0.01                    # The entropy coefficient in the Loss
    numIters = 1000              # The number of times to iterate the entire program
    modelDir = ".\\models"       # The location to save the models
    actorFilename = "actor"      # The name of the file to save the actor to
    criticFilename = "critic"    # The name of the file to save the critic to
    
    
    # Setup the observation space
    #observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    # Create a player
    player = Player(env.observation_space.shape, env.action_space.n, Lambda=Lambda, gamma=gamma, numActors=numActors, T=T, c1=c1, c2=c2, alpha=alpha)
    
    
    # The best average reward so far
    bestAvgReward = 0
    
    # player.loadModels(modelDir=modelDir, actorFilename=actorFilename, criticFilename=criticFilename)
    # while True:
    #     observation = env.reset()
    #     player.runPolicy(env, observation, T)
    
    
    # Iterate for numIters times
    torch.autograd.set_detect_anomaly(True)
    for iteration in range(1, numIters):
        # Iterate over all actors
        for actor in range(1, numActors):
            # The average reward across epochs
            avgReward = 0
            
            # Run and Update the model numEpoch times before 
            # reseting the memory
            for epoch in range(numEpochs):
                # Reset the environment variables
                observation = env.reset()
                
                # Update the hyperparameters
                alpha = 1-(iteration*actor)/(numIters*numActors)
                stepSize = stepSize_start*alpha
                epsilon = epsilon_start*alpha
                
                # Run the models for T timesteps and save the results to memory
                player.runPolicy(env, observation, T)
            
                # Compute the gradients for the models to optimize the policy
                avgReward += player.computeGrads(alpha=alpha, stepSize=stepSize, epsilon=epsilon)
            
            
            avgReward = avgReward/numEpochs
            print(f"Current average reward: {avgReward}")
        
            # Reset the memory and update the models
            player.resetMemory()
            player.updateModels()
            
            # If the current average reward is better than the best average
            # reward, save the models
            if avgReward > bestAvgReward:
                print("Updating Models")
                player.saveModels(modelDir=modelDir, actorFilename=actorFilename, criticFilename=criticFilename)
                bestAvgReward = avgReward