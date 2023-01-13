# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        g_pos = [(ghost.getPosition()) for ghost in newGhostStates]
        def dist(x):
            return util.manhattanDistance(x, newPos)
        if min(newScaredTimes) <= 0 and newPos in g_pos:
            return -1
        curFood = currentGameState.getFood()
        if newPos in curFood.asList():
            return 100

        closeFood = sorted(newFood.asList(), key = dist)
        closeGhost = sorted(g_pos, key = dist)
        return 1/dist(closeFood[0]) - 1/dist(closeGhost[0])

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        result = self.minimaxhelper(gameState, agentIndex=0, depth=self.depth)
        return result[1]

    def minimaxhelper(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            return self.maxhelper(gameState, agentIndex, depth)
        else:
            return self.minhelper(gameState, agentIndex, depth)

    def minhelper(self, gameState, agentIndex, depth):
        if agentIndex + 1 == gameState.getNumAgents():
            next_agent, next_depth = 0, depth - 1
        else:
            next_agent, next_depth = agentIndex + 1, depth
        score = 1000000000000
        action = Directions.STOP
        for actions in gameState.getLegalActions(agentIndex):
            succ = gameState.generateSuccessor(agentIndex, actions)
            succ_score = self.minimaxhelper(succ, next_agent, next_depth)
            if score > succ_score[0]:
                score, action = succ_score[0], actions
        return score, action

    def maxhelper(self, gameState, agentIndex, depth):
        if agentIndex + 1 == gameState.getNumAgents():
            next_agent, next_depth = 0, depth - 1
        else:
            next_agent, next_depth = agentIndex + 1, depth
        score = -1000000000000
        action = Directions.STOP
        for actions in gameState.getLegalActions(agentIndex):
            succ = gameState.generateSuccessor(agentIndex, actions)
            succ_score = self.minimaxhelper(succ, next_agent, next_depth)
            if score < succ_score[0]:
                score, action = succ_score[0], actions
        return score, action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        result = self.alphabetahelper(gameState, agentIndex=0, depth=self.depth, alpha = -1000000000000, beta = 1000000000000)
        return result[1]

    def alphabetahelper(self, gameState, agentIndex, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            return self.alphahelper(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.betahelper(gameState, agentIndex, depth, alpha, beta)

    def betahelper(self, gameState, agentIndex, depth, alpha, beta):
        if agentIndex + 1 == gameState.getNumAgents():
            next_agent, next_depth = 0, depth - 1
        else:
            next_agent, next_depth = agentIndex + 1, depth
        score = 1000000000000
        action = Directions.STOP
        for actions in gameState.getLegalActions(agentIndex):
            succ = gameState.generateSuccessor(agentIndex, actions)
            succ_score = self.alphabetahelper(succ, next_agent, next_depth, alpha, beta)
            if score > succ_score[0]:
                score, action = succ_score[0], actions
            if alpha > succ_score[0]:
                return succ_score[0], actions
            beta = min(beta, score)
        return score, action

    def alphahelper(self, gameState, agentIndex, depth, alpha, beta):
        if agentIndex + 1 == gameState.getNumAgents():
            next_agent, next_depth = 0, depth - 1
        else:
            next_agent, next_depth = agentIndex + 1, depth
        score = -1000000000000
        action = Directions.STOP
        for actions in gameState.getLegalActions(agentIndex):
            succ = gameState.generateSuccessor(agentIndex, actions)
            succ_score = self.alphabetahelper(succ, next_agent, next_depth, alpha, beta)
            if score < succ_score[0]:
                score, action = succ_score[0], actions
            if beta < succ_score[0]:
                return succ_score[0], actions
            alpha = max(alpha, score)
        return score, action

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        result = self.expectimaxhelper(gameState, agentIndex=0, depth=self.depth)
        return result[1]

    def expectimaxhelper(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:
            return self.maxhelper(gameState, agentIndex, depth)
        else:
            return self.expecthelper(gameState, agentIndex, depth)

    def expecthelper(self, gameState, agentIndex, depth):
        if agentIndex + 1 == gameState.getNumAgents():
            next_agent, next_depth = 0, depth - 1
        else:
            next_agent, next_depth = agentIndex + 1, depth
        score = 0
        action = Directions.STOP
        for actions in gameState.getLegalActions(agentIndex):
            succ = gameState.generateSuccessor(agentIndex, actions)
            score += self.expectimaxhelper(succ, next_agent, next_depth)[0]
        score /= len(gameState.getLegalActions(agentIndex))
        return score, action

    def maxhelper(self, gameState, agentIndex, depth):
        if agentIndex + 1 == gameState.getNumAgents():
            next_agent, next_depth = 0, depth - 1
        else:
            next_agent, next_depth = agentIndex + 1, depth
        score = -1000000000000
        action = Directions.STOP
        for actions in gameState.getLegalActions(agentIndex):
            succ = gameState.generateSuccessor(agentIndex, actions)
            succ_score = self.expectimaxhelper(succ, next_agent, next_depth)
            if score < succ_score[0]:
                score, action = succ_score[0], actions
        return score, action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    I am keeping track of the current score, ghost's position, capsule's position and the food left.
    I am also using manhattan distance as a heuristic function.
    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    food_list  = currentGameState.getFood().asList()
    gdist = 1000000000000
    for ghost in currentGameState.getGhostStates():
        if ghost.scaredTimer == 0:
            gdist = min(gdist, manhattanDistance(ghost.getPosition(), position))
        else:
            gdist = -100
    fdist = 1000000000000
    if not food_list:
        fdist = 0
    for food in food_list:
        fdist = min(fdist, manhattanDistance(food, position))
    return currentGameState.getScore() - 10/(gdist + 1) - fdist/3
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
