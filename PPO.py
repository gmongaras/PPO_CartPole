import numpy as np
from numpy.core.arrayprint import repr_format
import torch
import torch.nn as nn
import copy
import os



device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')




class Actor(nn.Module):
    # Inputs:
    #   stateShape - The shape of a state which the actor takes as input
    #   numActions - The number of actions which the actor uses as output
    #   alpha - The learning rate of the model
    def __init__(self, stateShape, numActions, alpha):
        super(Actor, self).__init__()
        self.stateShape = stateShape
        self.numActions = numActions


        # Actor network:
        # Input shape: The shape of the state
        # Output shape: The number of actions
        # Note: The output is in log form
        self.actor = nn.Sequential(
            nn.Linear(np.array(stateShape)[0], 256),
            nn.ReLU(),
            
            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, numActions),
            nn.LogSoftmax(dim=-1)
        ).to(device)
        
        
        # Actor optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
    

    # Feed forward
    def forward(self, state):
        if type(state) is np.ndarray:
            state = torch.from_numpy(state).to(device=device)
        
        # Get the forward feed value from the network and return it
        return self.actor(state)


    # Save the model to a file
    def saveModel(self, filename):
        torch.save(self.state_dict(), filename)
    
    # Load the model from a file
    def loadModel(self, filename):
        self.load_state_dict(torch.load(filename))





class Critic(nn.Module):
    # Inputs:
    #   stateShape - The shape of a state which the actor takes as input
    #   alpha - The learning rate of the model
    def __init__(self, stateShape, alpha):
        super(Critic, self).__init__()
        
        
        self.stateShape = stateShape


        # Critic network:
        # Input shape: The number of possible actions
        # Output shape: 1 representing how good a state is
        self.critic = nn.Sequential(
            nn.Linear(np.array(stateShape)[0], 256),
            nn.ReLU(),
            
            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 1)
        ).to(device)
        
        
        # Critic optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
    

    # Feed forward
    def forward(self, state):
        if type(state) is np.ndarray:
            state = torch.from_numpy(state).to(device=device)
        
        # Return the critic value of the state
        return self.critic(state)
    
    
    # Save the model to a file
    def saveModel(self, filename):
        torch.save(self.state_dict(), filename)
    
    # Load the model from a file
    def loadModel(self, filename):
        self.load_state_dict(torch.load(filename))




# Class to hold memory each iteration
class Memory:
    def __init__(self):
        # Normal memory
        self.states = []
        self.rewards = []
        self.actorProbs = []
        self.oldActorProbs = []
        self.currCriticVals = []
        self.prevCriticVals = []
        self.r_ts = []
        self.dones = []
        
        
        # Advantage memory
        # The delta value at each timestep used to calculate the advantage
        self.deltas = torch.zeros(0, dtype=torch.float, requires_grad=True, device=device)
        
        # The advantages for each timestep
        self.advantages = torch.zeros(0, dtype=torch.float, requires_grad=True, device=device)


        # The count of all memory
        self.memCount = 0
    

    # Store some new memory
    def addMemory(self, state, reward, actorProbs, oldActorProbs,
                    currCriticVal, prevCriticVal, r_t, done):
        self.states.append(state)
        self.rewards.append(reward)
        self.actorProbs.append(actorProbs)
        self.oldActorProbs.append(oldActorProbs)
        self.currCriticVals.append(currCriticVal)
        self.prevCriticVals.append(prevCriticVal)
        self.r_ts.append(r_t)
        self.dones.append(done)

        self.memCount += 1


    # Clears all memory from the object
    def clearMemory(self):
        self.states = []
        self.rewards = []
        self.actorProbs = []
        self.oldActorProbs = []
        self.currCriticVals = []
        self.prevCriticVals = []
        self.r_ts = []
        self.dones = []
        
        self.deltas = torch.zeros(0, dtype=torch.float, requires_grad=True, device=device)
        self.advantages = torch.zeros(0, dtype=torch.float, requires_grad=True, device=device)

        self.memCount = 0
    
    # Get the size of the memory
    def getMemorySize(self):
        return self.memCount
    
    # Get the data from the memory
    def getMemory(self):
        return self.states, self.rewards, self.actorProbs, self.oldActorProbs,\
               self.currCriticVals, self.prevCriticVals, self.r_ts, self.dones,\
               self.deltas, self.advantages
    
    
    # Using the stored memory, calculate the advantage at each timestep
    # Inputs:
    #    gamma - The gamma hyperparameter
    #    Lambda - The lambda hyperparameter
    def calculateAdvantages(self, gamma, Lambda):
        # if the memory is empty, there is nothing to compute
        if self.isEmpty() == True:
            return
        
        # Reset the advantages
        self.deltas = torch.zeros(0, dtype=torch.float, requires_grad=True, device=device)
        self.advantages = torch.zeros(0, dtype=torch.float, requires_grad=True, device=device)
        
        # Get the data from the memory
        states = self.states
        rewards = self.rewards
        actorProbs = self.actorProbs
        oldActorProbs = self.oldActorProbs
        currCriticVals = self.currCriticVals
        prevCriticVals = self.prevCriticVals
        r_ts = self.r_ts
        dones = self.dones

        # Iterate over all parts of memory and compute the delta values
        for m in range(0, self.memCount-1):
            delta = rewards[m] + gamma*currCriticVals[m+1] - currCriticVals[m]
            self.deltas = torch.cat([self.deltas, delta])

        # Iterate over all parts of memory and compute the advantages using
        # the delta values
        for m in range(0, self.memCount-1):
            advantage = 0
            for m2 in range(m, self.memCount-1):
                advantage += ((gamma*Lambda)**(m2-m))*self.deltas[m2]*(1 if dones[m] == False else 0)
            self.advantages = torch.cat([self.advantages, advantage.view(1)])
        
    
    
    # Randomize the order of all memory and return that minibatch of memory
    # Inputs:
    #    minibatchSize - The size of each minibatch which is the size of each memory sample 
    #    gamma - The gamma hyperparameter
    #    Lambda - The lambda hyperparameter
    def sampleMemory(self, minibatchSize, gamma, Lambda):
        # Create an array containing the indicies of all items in memory
        indices = np.array([i for i in range(0, self.memCount-1)])
        
        # Randomize the memory indices
        np.random.shuffle(indices)
        
        # Get the first `minibatchSize` number of indices from the list
        # and return those memory pieces
        returnIndices = indices[0:minibatchSize]
        
        # Change the needed lists to tensors for operations while retaining
        # the gradient graph.
        # NOTE: torch.tensor() does not retain the graph so we use torch.cat
        # or torch.stack.
        states = torch.tensor(self.states, requires_grad=True, device=device)
        rewards = torch.tensor(self.rewards, dtype=torch.float, requires_grad=True, device=device)
        actorProbs = torch.stack(self.actorProbs)
        oldActorProbs = torch.stack(self.oldActorProbs)
        currCriticVals = torch.cat(self.currCriticVals)
        prevCriticVals = torch.cat(self.prevCriticVals)
        r_ts = torch.stack(self.r_ts)
        dones = torch.tensor(self.dones, dtype=torch.bool, requires_grad=False, device=device)
        
        
        # Calculate the total reward and return it
        totalReward = torch.sum(rewards).item()
        
        # Return the subarrays
        return states[returnIndices], rewards[returnIndices],\
               actorProbs[returnIndices], oldActorProbs[returnIndices],\
               currCriticVals[returnIndices], prevCriticVals[returnIndices],\
               r_ts[returnIndices], dones[returnIndices],\
               self.deltas[returnIndices], self.advantages[returnIndices],\
               totalReward
    
    
    # Calculate and return the total reward
    def getTotalReward(self):
        return np.sum(np.array(self.rewards))
    
    
    # Returns True if the memory is empty, False otherwise
    def isEmpty(self):
        return len(self.states)<=1
    
    



class Player:
    # Setup the Actor-Critic player
    #   stateShape - The shape of a state which the actor takes as input
    #   numActions - The number of actions which the actor uses as output
    #   Lambda - The lambda hyperparameter
    #   gamma - The gamma hyperparameter
    #   numActors - The number of toal actors playing the game
    #   numEpochs - The number of epochs to train the model
    #   T - number of timesteps to run the model
    #   c1 - A coefficient for the VF loss term
    #   c2 - A coefficient for the entropy loss term
    #   alpha - The learning rate of the model
    def __init__(self, stateShape, numActions, Lambda=0.95, gamma=0.99, numActors=8, numEpochs=3, T=128, c1=1, c2=0.01, alpha=0.001):
        # Store the hyperparameters
        self.Lambda = Lambda
        self.gamma = gamma
        self.numActors = numActors
        self.numEpochs = numEpochs
        self.T = T
        self.c1 = torch.tensor(c1, dtype=torch.float, requires_grad=True, device=device)
        self.c2 = torch.tensor(c2, dtype=torch.float, requires_grad=True, device=device)

        # Create a memory object for each actor
        self.memory = []
        for actor in range(0, numActors):
            self.memory.append(Memory())

        # Create the actors and critic
        self.actor = Actor(stateShape=stateShape, numActions=numActions, alpha=alpha)
        self.oldActor = None
        self.critic = Critic(stateShape=stateShape, alpha=alpha)
        
        # Zero the gradients
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
    

    # Store memory using the given state
    #   memNum - The memory index to store the given memory
    #   state - The state to store memory on
    #   reward - The reward received at the current state
    #   actorProbs - The probabilities the actor returned for the given state
    #   oldActorProbs - The probabilities the old actor returned for the given state
    #   currCriticVal - The critic value of the current state
    #   prevCriticVal - The critic value of the previous state
    #   r_t - The ratio of the current action to the previous action
    #   done - If the player is done playing the game
    def storeMemory(self, memNum, state, reward, actorProbs, oldActorProbs,
                    currCriticVal, prevCriticVal, r_t, done):
        # Store the information in memory in the given memory location
        self.memory[memNum].addMemory(state, reward, actorProbs, oldActorProbs, currCriticVal, prevCriticVal, r_t, done)
        
        
        
    # Reset all memory in the model
    def resetMemory(self):
        for m in self.memory:
            m.clearMemory()
            
            
    
    
    
    
    
    # Run the policy in the environment and store the advantages
    #   actorNum - The index of the current actor bein ran
    #   env - The environemnt to run the models in
    #   observation - The current observed environment
    #   T - The number of timesteps to run the model
    #   showTraining - If the environemnt should be shown during training
    #   numActions - The number of actions before an update
    #   minibatchSize - The size of each minibatch to train the network on
    #   alpha - The learning rate
    #   numEpochs - The number of times to update the model
    #   stepSize - The stepping size used for the Adam optimizer
    #   epsilon - The epsilon hyperparameter for the clipped loss
    def runPolicy(self, actorNum, env, observation, T, showTraining, numActions, minibatchSize, alpha, numEpochs, stepSize, epsilon):
        # Calculate the first action and critic value
        # and update the environment
        prevCriticVal = torch.tensor([0], device=device)
        
        # Hold the rewards so we know what the total reward is
        rewards = 0
        
        # run the models T times
        for t in range(1, T):
            # Render the environment
            if showTraining == True:
                env.render()
            
            # Get an action sample from the actor distribution
            # Note: The actions are in log form
            actions = self.actor(observation)
            bestAction = torch.argmax(actions)
            bestAction.requires_grad = False
            
            # Calculate the critic value for the new state
            currCriticVal = self.critic(observation)
            
            # Calculate r_t, the ratio of the old policy action
            # and the current policy action
            if (self.oldActor == None):
                oldActions = torch.zeros(actions.shape[0], device=device)
                oldBestAction = 0
                r_t = torch.exp(actions)
            else:
                oldActions = self.oldActor(state=observation)
                oldActions.detach()
                oldBestAction = torch.argmax(oldActions)
                
                # Calculate ratio using log division. Same as (actions/oldActions)
                # Note that the output is in log form so we just subtract the
                # two values and exponentiate them
                r_t = torch.exp(actions-oldActions)
            r_t = r_t.mean()
            
            # Get the data from the new environment
            observation, reward, done, info = env.step(bestAction.item())
            
            # Save the new info to memory
            self.storeMemory(actorNum, observation, reward=reward, actorProbs=actions, oldActorProbs=oldActions, currCriticVal=currCriticVal, prevCriticVal=prevCriticVal, r_t=r_t, done=done)
            
            # Update the previous critic value to the current values
            prevCriticVal = currCriticVal
            
            if t % numActions == 0:
                rewards += self.getAvgReward(actorNum)
                self.computeGrads(actorNum, minibatchSize, alpha, numEpochs, stepSize, epsilon)
            
            # If done is true, stop the iteration
            if done == True:
                break
                
        # Return the average reward
        rewards += self.getAvgReward(actorNum)
        return rewards
    
    



    # Compute the gradients for the policy based on the loss functions
    #   minibatchSize - The size of each minibatch which is the size of each memory sample
    #   alpha - The learning rate
    #   numEpochs - The number of times to update the model
    #   stepSize - The stepping size used for the Adam optimizer
    #   epsilon - The epsilon hyperparameter for the clipped loss
    def computeGrads(self, actor, minibatchSize, alpha=1, numEpochs=3, stepSize=0.00025, epsilon=0.1):        
        # Compute the advantages for all memory
        self.memory[actor].calculateAdvantages(self.gamma, self.Lambda)
        
        # Convert the parameters to torch arrays
        epsilon = torch.tensor(epsilon, dtype=torch.float, requires_grad=False, device=device)
        
        
        # Update the learning rate in the optimizers
        #self.actor.optimizer.param_groups[0]["lr"] = alpha
        #self.critic.optimizer.param_groups[0]["lr"] = alpha
        
        
        
        # Update the model numEpochs times
        for epoch in range(0, numEpochs):
            
            
            
            m = self.memory[actor]
            # If the memory is empty, skip it
            if m.isEmpty():
                continue
            
            # Randomize the memory and sample it.
            states, rewards, actorProbs, oldActorProbs, currCriticVals, prevCriticVals, r_ts, dones, deltas, advantages, reward = m.sampleMemory(minibatchSize=minibatchSize, gamma=self.gamma, Lambda=self.Lambda)
            
            newActorProbs = self.actor(states)
            newCriticVal = self.critic(states)
            new_rts = torch.mean(torch.exp(newActorProbs - actorProbs.detach()), dim=-1)
            
            
            # Calculate the actor loss (L_CLIP)
            L_CLIP = -(torch.min(new_rts*advantages.detach(), torch.clip(new_rts, 1-epsilon, 1+epsilon)*advantages.detach()).mean())
            
            # Calculate the critic loss (L_VF)
            #L_VF = -(torch.square(rewards-currCriticVals).mean())
            #L_VF = -(torch.square(rewards-newCriticVal).mean())
            L_VF = -(torch.square((advantages.detach()+currCriticVals.detach())-newCriticVal).mean())
            
            # Get the entropy bonus from a normal distribution
            S = torch.tensor(np.random.normal(), dtype=torch.float, requires_grad=False)
            
            # Calculate the final loss (L_CLIP_VF_S)
            L_Final = L_CLIP - self.c1*L_VF + self.c2*S
            
            
            # Backpropogate the total loss to get the gradients
            L_Final.backward(retain_graph=False)
            
            # Step the optimizers and update the models
            self.actor.optimizer.step()
            self.critic.optimizer.step()
            
            # Zero the gradients
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            
            self.oldActor = copy.deepcopy(self.actor)
            self.oldCritic = copy.deepcopy(self.critic)

        
        # Reset the memory
        self.resetMemory()
        
        
    # Save the models to specified filenames
    def saveModels(self, modelDir, actorFilename, criticFilename):
        if not os.path.isdir(modelDir):
            os.mkdir(modelDir)
        self.actor.saveModel(os.path.join(modelDir, actorFilename))
        self.critic.saveModel(os.path.join(modelDir, criticFilename))
        
    # Load the models from the specified filenames
    def loadModels(self, modelDir, actorFilename, criticFilename):
        self.actor.loadModel(os.path.join(modelDir, actorFilename))
        self.critic.loadModel(os.path.join(modelDir, criticFilename))
    
    # Get the average reward from memory
    def getAvgReward(self, actorNum):
        return self.memory[actorNum].getTotalReward()