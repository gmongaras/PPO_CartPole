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
    T = 200                       # The horizon or total time per batch
    stepSize_start = 0.00025      # The starting Adam optimizer step size
                                  # this will be updated as alpha updates
    numEpochs = 3                 # The total number of epochs
    numActors = 5                 # (N) The total number of different actors to use
    minibatchSize = 5             # The size of each minibatch to sample batch data
    numActions = 20               # Number of actions before an update
    gamma = 0.99                  # The discount rate
    Lambda = 0.95                 # The GAE parameter
    epsilon_start = 0.2           # The clipping paramter. This value will
                                  # be updated as alpha updates
    alpha = 0.0001                # Starting value of the learning rate which
                                  # will decrease as the model updates
    c1 = 0.6                      # The VF coefficient in the Loss
    c2 = 0.01                     # The entropy coefficient in the Loss
    numIters = 200                # The number of times to iterate the entire program
    
    
    
    
    
    # Model saving variables
    modelDir = ".\\models"          # The location to save the models
    actorFilename = "actor"         # The name of the file to save the actor to
    criticFilename = "critic"       # The name of the file to save the critic to
    loadPreSaved = False            # True to use a pre saved model. False otherwise
    actorFilenameMax = "actor-200"  # The name of the file to save the actor if it reaches
                                    # The set max reward
    criticFilenameMax = "critic-200"# The name of the file to save the critic if it reaches
                                    # The set max reward
    maxReward = 200                 # The max possible reward in this environment
    
    
    
    # Graph saving variables
    graphDir = ".\\graphs"          # The location to save the graph of the model training
    graphFilename = "training.png"  # The filename of the graph picture
    graphX = []                     # Array to hold the X axis data of the graph
                                    # which is number of iterations
    graphY = []                     # Array to hold the Y axis data of the graph
                                    # which is the current average reward
                                    
                                    
                                    
    # Other variables
    showTraining = True            # True if the environment should be shown during
                                    # training. False otherwise.
    updateParameters = False        # True if the alpha, epsilon, and the stepSize should
                                    # update as the iterations increases. False otherwise
    maxCount = 3                    # The max count of the average rewards
    
    
    
    
    # Create a player
    player = Player(env.observation_space.shape, env.action_space.n, Lambda=Lambda, gamma=gamma, numActors=numActors, T=T, c1=c1, c2=c2, alpha=alpha)
    
    
    # Run a presaved model
    if loadPreSaved == True:
        player.loadModels(modelDir=modelDir, actorFilename=actorFilename, criticFilename=criticFilename)
        for iteration in range(1, numIters):
            observation = env.reset()
            player.runPolicy(0, env, observation, T, showTraining, numActions, minibatchSize)
    
    
    # Train the models
    else:
        # The best average reward so far
        bestAvgReward = -np.inf
        
        # The average rewards
        avgRewards = []
        
        
        # Iterate for numIters times
        for iteration in range(0, numIters):
            
            
            # Update the hyperparameters if specified
            if updateParameters == True:
                alpha = 1-(iteration/numIters)
                stepSize = stepSize_start*alpha
                epsilon = epsilon_start*alpha
            else:
                stepSize = stepSize_start
                epsilon = epsilon_start
            
            
            
            # Run the model in the environment numActors number of times
            avgReward = 0
            for actor in range(0, numActors):
                # Reset the environment variables
                observation = env.reset()
                
                # Run the models for T timesteps and save the results to memory
                avgReward += player.runPolicy(actor, env, observation, T, showTraining, numActions, minibatchSize, alpha, numEpochs, stepSize, epsilon)
                
                # Update the model when it's done performing
                player.computeGrads(actor, minibatchSize=minibatchSize, alpha=alpha, numEpochs=numEpochs, stepSize=stepSize, epsilon=epsilon)
            
            
            # Divide the average reward by the number of actors
            avgReward /= numActors
        
            
            
            
            
            
            # Store the average reward
            avgRewards.append(avgReward)
            if len(avgRewards) > maxCount:
                avgRewards = avgRewards[1:]
            
            # If the current average rewards are better than the best average
            # rewards, save the models
            totalAvgRewards = np.average(np.array(avgRewards))
            print(f"Step {iteration+1}.\t  Current average reward: {totalAvgRewards:.2f}.\t  Current reward: {avgReward}")
            if totalAvgRewards > bestAvgReward:
                if len(avgRewards) > 1:
                    if avgReward > avgRewards[-2]:
                        print("Saving Models")
                        player.saveModels(modelDir=modelDir, actorFilename=actorFilename, criticFilename=criticFilename)
                
                else:
                    print("Saving Models")
                    player.saveModels(modelDir=modelDir, actorFilename=actorFilename, criticFilename=criticFilename)
                bestAvgReward = totalAvgRewards
            
            # If the current average rewards equal the max value,
            # save the models
            if avgReward >= maxReward:
                print("Saving Models Due To Max Reward")
                player.saveModels(modelDir=modelDir, actorFilename=actorFilenameMax, criticFilename=criticFilenameMax)
            
            # Update the graph lists
            graphX.append(iteration)
            graphY.append(totalAvgRewards)
        
        
        
        
        # When training is over, save the graph of the model training
        plt.plot(graphX, graphY)
        plt.xlabel("Number of Model Updates")
        plt.ylabel("Average Reward")
        plt.title("Number of Model Updates vs. Average Reward")
        if not os.path.isdir(graphDir):
            os.mkdir(graphDir)
        plt.savefig(os.path.join(graphDir, graphFilename))