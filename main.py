import gym
import numpy as np
from PPO import Player
import torch
import matplotlib.pyplot as plt
import os




if __name__ == '__main__':
    env = gym.make('CartPole-v0') # Our environemnt
    
    # Helps with debugging PyTorch models
    torch.autograd.set_detect_anomaly(True)

    # Hyperparameters
    T = 128                       # The horizon or total time per batch
    stepSize_start = 0.00025      # The starting Adam optimizer step size
                                  # this will be updated as alpha updates
    numEpochs = 3                 # The total number of epochs
    numActors = 8                 # (N) The total number of different actors to use
    minibatchSize = 32            # The size of each minibatch to sample batch data
    gamma = 0.99                  # The discount rate
    Lambda = 0.95                 # The GAE parameter
    epsilon_start = 0.1           # The clipping paramter. This value will
                                  # be updated as alpha updates
    alpha = 0.0001                # Starting value of the larning rate which
                                  # will decrease as the model updates
    c1 = 1                        # The VF coefficient in the Loss
    c2 = 0.01                     # The entropy coefficient in the Loss
    numIters = 500                # The number of times to iterate the entire program
    
    
    
    
    
    # Model saving variables
    modelDir = ".\\models"        # The location to save the models
    actorFilename = "actor"       # The name of the file to save the actor to
    criticFilename = "critic"     # The name of the file to save the critic to
    loadPreSaved = False          # True to use a pre saved model. False otherwise
    
    
    
    # Graph saving variables
    graphDir = ".\\graphs"          # The location to save the graph of the model training
    graphFilename = "training.png"  # The filename of the graph picture
    graphX = []                     # Array to hold the X axis data of the graph
                                    # which is number of iterations
    graphY = []                     # Array to hold the Y axis data of the graph
                                    # which is the current average reward
                                    
                                    
                                    
    # Other variables
    showTraining = True             # True if the environment should be shown during
                                    # training. False otherwise.
    
    
    
    
    # Setup the observation space
    #observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    # Create a player
    player = Player(env.observation_space.shape, env.action_space.n, Lambda=Lambda, gamma=gamma, numActors=numActors, T=T, c1=c1, c2=c2, alpha=alpha)
    
    
    # Run a presaved model
    if loadPreSaved == True:
        player.loadModels(modelDir=modelDir, actorFilename=actorFilename, criticFilename=criticFilename)
        for iteration in range(1, numIters):
            observation = env.reset()
            player.runPolicy(0, env, observation, T, showTraining)
    
    
    # Train a model
    else:
        
        # The best average reward so far
        bestAvgReward = -np.inf
        
        # The average rewards
        avgRewards = []
        
        # The max count of the average rewards
        maxCount = 25
        
        
        # Iterate for numIters times
        for iteration in range(0, numIters):
            # Iterate over all actors
            #for actor in range(1, numActors):
            
            
            
            
            
            # Run the model in the environment numActors number of times
            for actor in range(0, numActors):
                # Reset the environment variables
                observation = env.reset()
                
                # Update the hyperparameters
                #alpha = 1-((iteration*actor)/(numIters*numActors))
                alpha = 1-(iteration/numIters)
                stepSize = stepSize_start*alpha
                epsilon = epsilon_start*alpha
                
                # Run the models for T timesteps and save the results to memory
                player.runPolicy(actor, env, observation, T, showTraining)
            
            
            
            
            # Update the model numEpochs times
            avgReward = player.computeGrads(minibatchSize=minibatchSize, alpha=alpha, numEpochs=numEpochs, stepSize=stepSize, epsilon=epsilon, numActors=numActors)
        
            # Reset the memory and update the models
            player.resetMemory()
            player.updateModels()
            
            # Store the average reward
            avgRewards.append(avgReward)
            if len(avgRewards) > maxCount:
                avgRewards = avgRewards[1:]
            
            # If the current average rewards are better than the best average
            # rewards, save the models
            totalAvgRewards = np.average(np.array(avgRewards))
            print(f"Step {iteration+1}. Current average reward: {totalAvgRewards}")
            if totalAvgRewards > bestAvgReward:
                if len(avgRewards) > 1:
                    if avgReward > avgRewards[-2]:
                        print("Saving Models")
                        player.saveModels(modelDir=modelDir, actorFilename=actorFilename, criticFilename=criticFilename)
                        bestAvgReward = totalAvgRewards
                
                else:
                    print("Saving Models")
                    player.saveModels(modelDir=modelDir, actorFilename=actorFilename, criticFilename=criticFilename)
                    bestAvgReward = totalAvgRewards
            
            # Update the graph lists
            graphX.append(iteration)
            graphY.append(totalAvgRewards)
        
        
        # When training is over, save the graph of the model training
        plt.plot(graphX, graphY)
        plt.xlabel("Number of Model Updates")
        plt.ylabel("Average Reward at Iteration")
        plt.title("Number of Model Updates vs. Average Reward")
        if not os.path.isdir(graphDir):
            os.mkdir(graphDir)
        plt.savefig(os.path.join(graphDir, graphFilename))