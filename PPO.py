from math import gamma
import numpy as np
from numpy.testing._private.utils import requires_memory
from tensorflow.python.util.tf_stack import CurrentModuleFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import copy



device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')




class Actor(nn.Module):
    # Inputs:
    #   stateShape - The shape of a state which the actor takes as input
    #   numActions - The number of actions which the actor uses as output
    def __init__(self, stateShape, numActions):
        super(Actor, self).__init__()
        self.stateShape = stateShape
        self.numActions = numActions


        # Actor network:
        # Input shape: The shape of the state
        # Output shape: The number of actions
        self.actor = nn.Sequential(
            nn.Linear(np.array(stateShape)[0], 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),
            
            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, numActions),
            nn.Softmax()
        ).to(device)
        
        
        # Actor optimizer
        self.optimizer = torch.optim.Adam(self.parameters())
    

    # Feed forward
    def forward(self, state):
        if type(state) is np.ndarray:
            state = torch.from_numpy(state)
            #state.requires_grad = True
        
        # Get the forward feed value from the network and return it
        return self.actor(state)





class Critic(nn.Module):
    # Inputs:
    #   stateShape - The shape of a state which the actor takes as input
    def __init__(self, stateShape):
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
            
            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 1)
        ).to(device)
        
        
        # Critic optimizer
        self.optimizer = torch.optim.Adam(self.critic.parameters())
    

    # Feed forward
    def forward(self, state):
        if type(state) is np.ndarray:
            state = torch.from_numpy(state)
        
        # Return the critic value of the state
        return self.critic(state)




# Class to hold memory each iteration
class Memory:
    def __init__(self):
        self.states = []
        self.rewards = []
        self.currBestActions = []
        self.oldBestActions = []
        self.currCriticVals = []
        self.prevCriticVals = []
        self.r_ts = []
        self.deltas = []

        self.memCount = 0
    

    # Store some new memory
    def addMemory(self, state, reward, currBestAction, oldBestAction,
                    currCriticVal, prevCriticVal, r_t, delta):
        self.states.append(state)
        self.rewards.append(reward)
        self.currBestActions.append(currBestAction)
        self.oldBestActions.append(oldBestAction)
        self.currCriticVals.append(currCriticVal)
        self.prevCriticVals.append(prevCriticVal)
        self.r_ts.append(r_t)
        self.deltas.append(delta)

        self.memCount += 1


    # Clears all memory from the object
    def clearMemory(self):
        self.states = []
        self.rewards = []
        self.currBestActions = []
        self.oldBestActions = []
        self.currCriticVals = []
        self.prevCriticVals = []
        self.r_ts = []
        self.deltas = []

        self.memCount = 0
    
    # Get the size of the memory
    def getMemorySize(self):
        return self.memCount
    
    # Get the data from the memory
    def getMemory(self):
        return self.states, self.rewards, self.currBestActions, self.oldBestActions,\
               self.currCriticVals, self.prevCriticVals, self.r_ts, self.deltas



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
    def __init__(self, stateShape, numActions, Lambda=0.95, gamma=0.99, numActors=8, numEpochs=3, T=128, c1=1, c2=0.01):
        # Store the hyperparameters
        self.Lambda = Lambda
        self.gamma = gamma
        self.numActors = numActors
        self.numEpochs = numEpochs
        self.T = T
        self.c1 = torch.tensor(c1, dtype=torch.float, requires_grad=True, device=device)
        self.c2 = torch.tensor(c2, dtype=torch.float, requires_grad=True, device=device)

        # Create a memory object
        self.memory = Memory()

        # Create the actors and critic
        #self.actors = [Actor(stateShape=stateShape, numActions=numActions) for i in range(0, numActors)]
        self.actor = Actor(stateShape=stateShape, numActions=numActions)
        self.oldActor = copy.deepcopy(self.actor)
        self.critic = Critic(stateShape=stateShape)
    

    # Store memory using the given state
    #   state - The state to store memory on
    #   reward - The reward received at the current state
    #   currBestAction - The action taken to get to this state
    #   oldBestAction - The action the old policy would take given the actions
    #                   taken to get to this state.
    #   currCriticVal - The critic value of the current state
    #   prevCriticVal - The critic value of the previous state
    #   r_t - The ratio of the current action to the previous action
    def storeMemory(self, state, reward, currBestAction, oldBestAction,
                    currCriticVal, prevCriticVal, r_t):
        # Calculate the delta value
        delta = reward + self.gamma*currCriticVal - prevCriticVal
        
        # Store the information in memory
        self.memory.addMemory(state, reward, currBestAction, oldBestAction, currCriticVal, prevCriticVal, r_t, delta)
        
        
        
    # Reset all memory in the model
    def resetMemory(self):
        self.memory.clearMemory()
    
    
    
    
    
    # Run the policy in the environment and store the advantages
    #   env - The environemnt to run the models in
    #   observation - The current observed environment
    #   T - The number of timesteps to run the model
    def runPolicy(self, env, observation, T):
        # # Calculate the first action and critic value
        # # and update the environment
        # actions = self.actor.forward(observation)
        # bestAction = torch.argmax(actions).numpy()
        # observation, reward, done, info = env.step(bestAction)
        # prevCriticVal = self.critic.forward(actions)
        prevCriticVal = 0
        
        # run the models T times
        for t in range(0, T):
            # Render the environment
            env.render()
            
            # Get an action sample from the actor distribution
            actions = self.actor(observation)
            bestAction = torch.argmax(actions)
            #actions_cat = Categorical(actions)
            #bestAction = actions_cat.sample()
            
            # Get the data from the new environment
            observation, reward, done, info = env.step(bestAction.item())
            
            # Calculate the critic value for the new state
            currCriticVal = self.critic(observation)
            
            # Calculate r_t, the ratio of the old policy action
            # and the current policy action
            oldActions = self.oldActor(state=observation)
            #oldActions.detach()
            oldBestAction = torch.argmax(oldActions)
            #oldActions_cat = Categorical(oldActions)
            #oldBestAction = oldActions_cat.sample()
            r_t = (actions/oldActions)
            r_t = torch.nan_to_num(r_t, nan=0.0)
            r_t = r_t.mean()
            
            # Save the new info to memory
            # observation = torch.tensor(observation, dtype=torch.float, requires_grad=True)#, requires_grad=True)
            # reward = torch.tensor(reward, dtype=torch.float, requires_grad=True)#, requires_grad=True)
            # bestAction = torch.tensor(bestAction, dtype=torch.float, requires_grad=True)
            # oldBestAction = torch.tensor(oldBestAction, dtype=torch.float, requires_grad=True,)#, requires_grad=False)
            # currCriticVal = torch.tensor(currCriticVal, dtype=torch.float, requires_grad=True)
            # prevCriticVal = torch.tensor(prevCriticVal, dtype=torch.float, requires_grad=True)#, requires_grad=False)
            # r_t = torch.tensor(r_t, dtype=torch.float, requires_grad=True)
            self.storeMemory(observation, reward=reward, currBestAction=bestAction, oldBestAction=oldBestAction, currCriticVal=currCriticVal, prevCriticVal=prevCriticVal, r_t=r_t)
            
            # Update the previous critic value to the current values
            prevCriticVal = currCriticVal
            
            # If done is true, stop the iteration
            if done == True:
                break
    
    



    # Update the models based on the loss functions
    #   alpha - The learning rate
    #   stepSize - The stepping size used for the Adam optimizer
    #   epsilon - The epsilon hyperparameter for the clipped loss
    def updateModels(self, alpha=1, stepSize=0.00025, epsilon=0.1):
        # Convert the parameters to torch arrays
        epsilon = torch.tensor(epsilon, dtype=torch.float, requires_grad=False, device=device)
        
        # Before updating, store the old models
        self.oldActor = copy.deepcopy(self.actor)
        self.oldCritic = copy.deepcopy(self.critic)
        
        
        # The delta value at each timestep used to calculate the advantage
        deltas = torch.zeros(self.memory.getMemorySize()-1, dtype=torch.float, requires_grad=True)
        
        # The advantages for each timestep
        advantages = torch.zeros(self.memory.getMemorySize()-1, dtype=torch.float, requires_grad=True)

        # Get the data from the memory
        states, rewards, currBestActions, oldBestActions, currCriticVals, prevCriticVals, r_ts, _ = self.memory.getMemory()

        # Iterate over all parts of memory and compute the delta values
        for m in range(0, self.memory.getMemorySize()-1):
            deltas.data[m] = (rewards[m] + self.gamma*currCriticVals[m+1] - currCriticVals[m])
            #torch.cat((deltas, rewards[m] + self.gamma*currCriticVals[m+1] - currCriticVals[m]))

        # Iterate over all parts of memory and compute the advantages
        for m in range(0, self.memory.getMemorySize()-1):
            advantage = deltas[m]
            for i in range(m+1, self.memory.getMemorySize()-1):
                advantage.data += (self.gamma*self.Lambda)*deltas[i]
            advantages.data[m] = advantage
            #torch.stack((advantages[m], advantage))
            #torch.cat((advantages, (self.gamma*self.Lambda)*deltas[m]))
        
        
        # Update the learning rate in the optimizers
        self.actor.optimizer.param_groups[0]["lr"] = alpha
        self.critic.optimizer.param_groups[0]["lr"] = alpha
        
        
        # Change the needed lists to tensors for operations while retaining
        # the gradient graph.
        # NOTE: torch.tensor() does not retain the graph so we use torch.cat
        # or torch.stack.
        rewards_Tensor = torch.tensor(rewards[0:advantages.shape[0]], dtype=torch.float, requires_grad=True, device=device)
        r_ts_Tensor = torch.stack(r_ts[0:advantages.shape[0]])
        currCriticVals_Tensor = torch.cat(currCriticVals[0:advantages.shape[0]])
        
        
        # Update the loss numEpochs times
        for epoch in range(0, self.numEpochs):
            # Calculate the actor loss (L_CLIP)
            L_CLIP = torch.min(r_ts_Tensor*advantages, torch.clip(r_ts_Tensor, 1-epsilon, 1+epsilon)*advantages).mean()
            
            # Calculate the critic loss (L_VF)
            L_VF = torch.pow(rewards_Tensor-currCriticVals_Tensor, 2).mean()
            
            # Get the entropy bonus from a normal distribution
            S = torch.tensor(np.random.normal(), dtype=torch.float, requires_grad=False)
            
            # Calculate the final loss (L_CLIP_VF_S)
            L_Final = L_CLIP - self.c1*L_VF + self.c2*S
            
            # Zero the gradients
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            
            
            # Backpropogate the total loss to get the gradients
            L_Final.backward()
            
            # Step the optimizers and update the models
            self.actor.optimizer.step()
            self.critic.optimizer.step()
        
        print("Reward:", torch.sum(rewards_Tensor).item(), "Total Loss:", L_Final.item(), "Actor Loss:", torch.mean(L_CLIP).item(), "Critic Loss:", torch.mean(L_VF).item())
        
        # Clear the memory
        self.memory.clearMemory()