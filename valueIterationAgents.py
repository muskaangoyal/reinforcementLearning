# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.actions = {} # A dictionary of best actions for every state
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        numIterations = self.iterations
        while numIterations != 0:
            numIterations -= 1
            updatedValues = util.Counter()
            actionValues = util.Counter()
            states = self.mdp.getStates()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                maxVal = float("-inf")
                for action in actions:
                    val = self.computeQValueFromValues(state, action)
                    actionValues[action] = val
                    if val > maxVal:
                        maxVal = val
                        updatedValues[state] = val
                self.actions[state] = actionValues.argMax()
            self.values = updatedValues   
            

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]
    
    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        val = 0
        for transitionState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, transitionState)
            discountedVal = self.discount * self.values[transitionState]
            val += prob*(reward + discountedVal)
        # print("Total Val:", val)
        return val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # if self.mdp.isTerminal(state): 
        #     return None
        # else:
        #     return self.actions[state]          
        if self.mdp.isTerminal(state): 
            return None
        values = util.Counter()
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            values[action] = self.computeQValueFromValues(state, action)
        return values.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        numStates = len(states)
        numIterations = self.iterations
        index = 0
        while numIterations != 0:
            numIterations -= 1
            state = states[index]
            index = (index + 1) % numStates
            actions = self.mdp.getPossibleActions(state)
            if not actions: 
                continue
            maxVal = float("-inf")
            for action in actions:
                val = self.computeQValueFromValues(state, action)
                maxVal = max(maxVal, val)
            self.values[state] = maxVal

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # 1. Initialize predecessors for all states.
        predecessors = {}

        # 2. Initialize an empty priority queue and push states into priority queue
        pq = util.PriorityQueue()
        states = self.mdp.getStates()
        for state in states:
            actions = self.mdp.getPossibleActions(state)
            maxVal = float("-inf")
            for action in actions:
                qVal = self.getQValue(state, action)
                maxVal = max(maxVal, qVal)
                # Compute predecessors of all states.
                for transitionState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        transitionSet = predecessors.get(transitionState, set()) #default value = empty set
                        newSet = set([state])
                        predecessors[transitionState] = set.union(newSet,transitionSet)
            
            # if state is a terminal state then countinue
            if self.mdp.isTerminal(state): 
                continue
            avgVal = self.values[state]
            difference = abs(maxVal - avgVal)
            pq.push(state, -difference)

        # 3. Iterations
        numIterations = self.iterations
        while numIterations != 0:
            numIterations -= 1

            # Terminate if pq is empty
            if pq.isEmpty(): 
                break

            # Pop a state s off the priority queue.
            state = pq.pop()

            # Update the value of s (if it is not a terminal state) in self.values.
            actions = self.mdp.getPossibleActions(state)
            maxVal = float("-inf")
            for action in actions:
                qVal = self.getQValue(state, action)
                maxVal = max(maxVal, qVal)
            self.values[state] = maxVal

            # For each predecessor p of s and update
            statePredecessors = predecessors.get(state, set())
            for predecessor in statePredecessors:
                predecessorActions = self.mdp.getPossibleActions(predecessor)
                maxVal = float("-inf")
                for preAction in predecessorActions:
                    qVal = self.getQValue(predecessor, preAction)
                    maxVal = max(maxVal, qVal)
                error = abs(self.values[predecessor] - maxVal)
                if (error > self.theta):
                    pq.update(predecessor, -error)