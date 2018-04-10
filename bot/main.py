#! /usr/bin/python3

import argparse
import copy
import time
import random
import sys
import os
import tensorflow as tf
import collections
import numpy as np
import pickle
import uuid
import time
from typing import TypeVar, Generic, List, Tuple, NamedTuple, ClassVar, Any
from GameSession import GameSession


# Improvement notes:
# - Use double-DQN (to stabilize and not overestimates future rewards)
# - Use boltzman or bayesian exploration (to avoid bias and stubbornness)
# - Rotate the state so the tank is alway heading north
#     - Cause less possibilities and less channels: more accurate and learn faster
# - Fix the issue with target network (batchs help to simulate a target network)
# - Use batch normalization if deeper networks are used
# References:
# - DQN: https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df
# - Exploration: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf
# - Batch normalization: https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/README_BATCHNORM.md


def geomScaling(inputSize:int, outputSize:int, internalStepCount:int):
    values = [inputSize] + [0]*internalStepCount + [outputSize]
    for i in range(1, internalStepCount+1):
        pow1 = (internalStepCount+1-i) / (internalStepCount+1)
        pow2 = i / (internalStepCount+1)
        values[i] = int(round((inputSize**pow1 * outputSize**pow2)))
    return values

def remapReward(realReward:float) -> float:
    virtualReward = np.sign(realReward) * np.log(np.abs(realReward)+1)
    return virtualReward

# An implementation of QTable
class DeepNeuralQTable:
    class State:
        precision : ClassVar[int] = 3
        value : np.ndarray

        def __init__(self, value:np.ndarray) -> None:
            self.value = value

        def __eq__(self, other:np.ndarray) -> bool:
            return (self.value.round(DeepNeuralQTable.State.precision) == other.value.round(DeepNeuralQTable.State.precision)).all()

        def __ne__(self, other:np.ndarray) -> bool:
            return (self.value.round(DeepNeuralQTable.State.precision) != other.value.round(DeepNeuralQTable.State.precision)).all()

        def __hash__(self) -> int:
            flatten = self.value.reshape(np.prod(self.value.shape))
            scaledInt = (flatten * 10**DeepNeuralQTable.State.precision).astype(int)
            return hash(tuple(scaledInt))

        def print(self, colored:bool=True) -> None:
            #for i in range(self.value.shape[3]):
            #    print(f'::MAP_CHANEL_{i}::')
            #    print(self.value[0,:,:,i].reshape((self.radius*2+1,self.radius*2+1)))
            assert self.value.shape[0] == 1
            assert self.value.shape[3] == 3
            for y in range(self.value.shape[1]):
                for x in range(self.value.shape[2]):
                    nextPos = False
                    curPos = x*2+1 == self.value.shape[2] and y*2+1 == self.value.shape[1]
                    if self.value[0,y,x,0] >= 0.8:
                        nextPos = True
                    wall = int(round(self.value[0,y,x,1]*2))
                    bonus = self.value[0,y,x,2] >= 0.01
                    if colored:
                        if curPos:
                            print('\033[48;5;5m', end='')
                            print('❰❱', end='')
                        elif nextPos:
                            print('\033[48;5;4m', end='')
                        else:
                            print('\033[48;5;0m', end='')
                    if wall == 0:
                        if bonus:
                            if colored:
                                if curPos:
                                    print('\033[48;5;94m', end='')
                                elif nextPos:
                                    print('\033[48;5;35m', end='')
                                else:
                                    print('\033[48;5;2m', end='')
                            else:
                                print('..', end='')
                        else:
                            if not colored:
                                print('  ', end='')
                    elif wall == 1:
                        if colored:
                            # bonus not supported
                            if curPos or nextPos:
                                print('\033[48;5;110m', end='')
                            else:
                                print('\033[48;5;245m', end='')
                        else:
                            print('xx', end='')
                    elif wall == 2:
                        if colored:
                            if curPos or nextPos:
                                print('\033[48;5;117m', end='')
                            else:
                                print('\033[48;5;7m', end='')
                        else:
                            print('XX', end='')
                    else:
                        assert False
                    if colored:
                        if not curPos:
                            print('  ', end='')
                        print('\033[0m', end='')
                print()

    # TODO: attribut types
    session : tf.Session
    radius : int
    careBonus : bool
    carePlayers : bool
    careShoots : bool
    outputSize : int
    channels : int
    inputs : Any
    predictedQ : Any
    realQ : Any
    loss : Any
    model : Any
    summary : Any
    writer : Any
    runId : int

    actions = [
        #GameSession.Order.STAY,
        GameSession.Order.TURNLEFT,
        GameSession.Order.TURNRIGHT,
        GameSession.Order.MOVE,
        GameSession.Order.STAY_SHOOT,
        #GameSession.Order.TURNLEFT_SHOOT,
        #GameSession.Order.TURNRIGHT_SHOOT,
        #GameSession.Order.MOVE_SHOOT,
        #GameSession.Order.SHOOT_TURNLEFT,
        #GameSession.Order.SHOOT_TURNRIGHT
    ]

    # radius: vision radius of the bot
    def __init__(self, radius:int, careBonus:bool, carePlayers:bool, careShoots:bool, 
                    learningRate:float, isTraining:bool) -> None:
        self.game = None
        self.session = None
        self.radius = radius
        self.careBonus = careBonus
        self.carePlayers = carePlayers
        self.careShoots = careShoots
        self.isTraining = isTraining
        self.normLayers = []

        self.outputSize = len(DeepNeuralQTable.actions)

        self.channels = 2
        if self.careBonus:
            self.channels += 1
        if self.carePlayers:
            self.channels += 1
        if self.careShoots:
            self.channels += 1

        self.inputs = tf.placeholder(shape=[None,self.radius*2+1,self.radius*2+1,self.channels], dtype=tf.float32)

        filters = np.array([16, 32, 40, 44, 48]) * self.channels
        kernelSizes = [2, 2, 2, 2, 2]
        kernelStrides = [1, 1, 1, 1, 1]

        # Note: batch normization and dropout cause both stabilization issues during training

        # CNN layers
        self.weightList = []
        convLayer = self._makeConvLayer(self.inputs, filters[0], kernelSizes[0], kernelStrides[0])
        if radius >= 2:
            convLayer = self._makeConvLayer(convLayer, filters[1], kernelSizes[1], kernelStrides[1])
            convLayer = self._makeConvLayer(convLayer, filters[2], kernelSizes[2], kernelStrides[2])
        if radius >= 3:
            convLayer = self._makeConvLayer(convLayer, filters[3], kernelSizes[3], kernelStrides[3])
            convLayer = self._makeConvLayer(convLayer, filters[4], kernelSizes[4], kernelStrides[4])
        lastCnnLayer = convLayer

        # Flatten layer
        lastCnnLayerFlat = tf.contrib.layers.flatten(lastCnnLayer)

        # Fully-connected layers
        lastCnnLayerSize = np.prod(lastCnnLayer.shape.as_list()[1:])
        denseLayerNeurons = geomScaling(lastCnnLayerSize, self.outputSize, 1)
        denseLayer = self._makeDenseLayer(lastCnnLayerFlat, denseLayerNeurons[1])
        self.predictedQ = self._makeDenseLayer(denseLayer, denseLayerNeurons[2])

        # Training model
        self.realQ = tf.placeholder(shape=[None,len(DeepNeuralQTable.actions)], dtype=tf.float32)
        with tf.name_scope('loss'):
            self.loss = tf.reduce_sum(tf.square(self.realQ - self.predictedQ))
        # GradientDescentOptimizer | AdamOptimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
        #updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(updateOps):
        #    self.model = optimizer.minimize(self.loss)
        self.model = optimizer.minimize(self.loss)
        self.model = tf.group(self.model, *self.normLayers)
        gradMagnitude = tf.reduce_sum([tf.reduce_sum(g**2) for g in tf.gradients(self.loss, self.weightList)])**0.5
        tf.summary.scalar('loss', self.loss)
        #tf.summary.scalar('grad_magnitude', gradMagnitude)
        self.summary = tf.summary.merge_all()

    def _makeConvLayer(self, inputLayer:Any, filterSize:int, kernelSize:int, kernelStride:int, normalize:bool=False) -> Any:
        with tf.name_scope('normalized-conv'):
            convLayer = tf.layers.conv2d(inputLayer, filters=filterSize, 
                                            kernel_size=kernelSize, strides=kernelStride, 
                                            padding='valid', use_bias=not normalize)
            #self.weightList.append(tf.get_default_graph().get_tensor_by_name('/'.join(convLayer.name.partition('/')[:-1]) + '/kernel:0'))
            if normalize:
                normLayer = tf.layers.batch_normalization(convLayer, scale=False, training=self.isTraining)
                self.normLayers.append(normLayer)
            else:
                normLayer = convLayer
            activationLayer = tf.nn.leaky_relu(normLayer, alpha=0.1)
        return activationLayer

    def _makeDenseLayer(self, inputLayer:Any, neurons:int, normalize:bool=False) -> Any:
        with tf.name_scope('normalized-dense'):
            denseLayer = tf.layers.dense(inputLayer, units=neurons, use_bias=not normalize)
            #self.weightList.append(tf.get_default_graph().get_tensor_by_name('/'.join(denseLayer.name.partition('/')[:-1]) + '/kernel:0'))
            if normalize:
                normLayer = tf.layers.batch_normalization(denseLayer, scale=False, training=self.isTraining)
                self.normLayers.append(normLayer)
            else:
                normLayer = denseLayer
            activationLayer = tf.nn.selu(normLayer)
        return denseLayer

    def configure(self, session:tf.Session, modelPath:str) -> None:
        self.session = session
        self.writer = tf.summary.FileWriter(os.path.join(modelPath, 'train'))
        self.writer.add_graph(self.session.graph)
        self.runId = 0

    def get(self, state:State) -> np.ndarray:
        return self.session.run(self.predictedQ, feed_dict={self.inputs: state.value})[0]

    def set(self, state:State, qLine:np.ndarray) -> None:
        actualQLine = np.ndarray(shape=(1, len(DeepNeuralQTable.actions)), dtype=float)
        actualQLine[0,:] = qLine
        _, summary = self.session.run([self.model, self.summary], feed_dict={self.inputs: state.value, self.realQ: actualQLine})
        self.writer.add_summary(summary, self.runId)
        self.runId += 1

    def supportBatch(self) -> bool:
        return True

    def getBatch(self, states:List[State]) -> np.ndarray:
        actualStates = [e.value[0,:,:,:] for e in states]
        return self.session.run(self.predictedQ, feed_dict={self.inputs: actualStates})

    def setBatch(self, states:List[State], qLines:List[np.ndarray]) -> None:
        actualStates = [e.value[0,:,:,:] for e in states]
        actualQLines = qLines
        summary = self.session.run([self.model, self.summary], feed_dict={self.inputs: actualStates, self.realQ: actualQLines})[1]
        self.writer.add_summary(summary, self.runId)
        self.runId += 1

    def stateFromGameSession(self, game:Any) -> State:
        assert game.playerId in game.playerInfos
        pos = game.playerInfos[game.playerId].position
        direction = game.playerInfos[game.playerId].direction
        # note: self.inputs should have a shape [batchSize/dynamic, width, height, channels]
        state = np.ndarray(shape=(1, self.radius*2+1, self.radius*2+1, self.channels), dtype=float)
        state.fill(-1)
        channel = 0
        state[0,:,:,channel] = 0
        # Direction
        state[0,self.radius+direction.y,self.radius+direction.x,channel] = 1
        channel += 1
        # Walls
        for y in range(-self.radius,self.radius+1):
            for x in range(-self.radius,self.radius+1):
                value = 2
                if pos.y+y in range(game.height):
                    if pos.x+x in range(game.width):
                        value = game.walls[pos.y+y, pos.x+x]
                state[0,y+self.radius,x+self.radius,channel] = value / 2.0
        channel += 1
        if self.careBonus:
            for y in range(-self.radius,self.radius+1):
                for x in range(-self.radius,self.radius+1):
                    value = 0
                    if pos.y+y in range(game.height):
                        if pos.x+x in range(game.width):
                            value = game.bonus[pos.y+y, pos.x+x]
                    state[0,y+self.radius,x+self.radius,channel] = remapReward(value)
            channel += 1
        return DeepNeuralQTable.State(value=state)

    def idToAction(self, actionId:int) -> GameSession.Order:
        return DeepNeuralQTable.actions[actionId]

# Generic Q-Learning class
QTable = TypeVar('QTable')
QState = TypeVar('QState')
class QLearning(Generic[QTable, QState]):
    qTable : QTable
    state : QState
    qLine : np.ndarray
    actionId : int
    learningRate : float
    discountFactor : float
    explorationTemperature : float
    memories : List[Any]

    def __init__(self, qTable:QTable, learningRate:float=0.05, discountFactor:float=0.5, 
                    explorationTemperature:float=0.5) -> None:
        self.qTable = qTable
        self.state = None
        self.qLine = None
        self.actionId = None
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.explorationTemperature = explorationTemperature
        self.memories = []

    def initState(self, state:QState) -> None:
        self.state = copy.deepcopy(state)

    # Return the best action by looking the Q-Table up (with randomness)
    def react(self) -> GameSession.Order:
        # Bolztman exploration: rewarding actions are more prone to be choosen
        self.qLine = self.qTable.get(self.state)
        rewards = self.qLine / self.explorationTemperature
        tmp = np.exp(rewards - np.max(rewards))
        softmax = tmp / tmp.sum()
        self.actionId = np.random.choice(range(len(self.qLine)), p=softmax)
        return self.qTable.idToAction(self.actionId)

    # Update Q-Table with new knowledge
    def update(self, newState:QState, reward:float, register:bool=False, learn:bool=True) -> None:
        if register:
            self.memories.append((copy.deepcopy(self.state), copy.deepcopy(self.actionId), copy.deepcopy(newState), float(reward)))
        if self.discountFactor >= 1e-6:
            maxFutureReward = self.discountFactor*np.max(self.qTable.get(newState))
            self.qLine[self.actionId] += self.learningRate*(reward + maxFutureReward - self.qLine[self.actionId])
        else:
            self.qLine[self.actionId] += self.learningRate*(reward - self.qLine[self.actionId])
        print('before:', self.qTable.get(self.state))
        print('requested:', self.qLine, ' -- ', self.actionId, self.qTable.idToAction(self.actionId))
        if learn:
            self.qTable.set(self.state, self.qLine)
        self.state = copy.deepcopy(newState)

    # Regularization of memories to reduce bias/overfitting
    # Remove duplicate elements <input-state, action, output-state>
    # When aggregation is use, elements are aggregated by input states
    def memoryRegularization(self, aggregation:bool=False) -> None:
        assert not aggregation, 'Not yet supported'
        regularizedMemory = dict()
        # Indexing
        for memory in self.memories:
            state, actionId, newState, reward = memory
            actions = regularizedMemory.get(state)
            if actions is None:
                actions = dict()
                regularizedMemory[state] = actions
            results = actions.get(actionId, set())
            if results is None:
                results = set()
                actions[actionId] = results
            results.add((newState, reward))
        # Checks
        replicates = 0
        for state,actions in regularizedMemory.items():
            if len({newState for newState, reward in actions.values()}) > 1:
                replicates += 1
        if replicates > 0:
            print(f'Warning: {replicates} replicated entries')
        # Reduction
        for state,actions in regularizedMemory.items():
            pass
            #actions = regularizedMemory.get(state, dict())
            #results = actions.get(actionId, set())
            #results.add((newState, reward))
        pass

    # Experience replay of memories
    def replayMemories(self, batchSize:int=64) -> None:
        # TODO: prevent the replay of the same action to prevent huge bias
        # Mixing the memories prevent a bias if an ANN is used for the Q-Table
        random.shuffle(self.memories)
        if self.qTable.supportBatch():
            for i in range(0, len(self.memories), batchSize):
                batchStates = []
                batchNewStates = []
                batchInfos = []
                for state, actionId, newState, reward in self.memories[i:i+batchSize]:
                    batchStates.append(state)
                    batchNewStates.append(newState)
                    batchInfos.append((actionId, reward))
                qLines = self.qTable.getBatch(batchStates)
                if self.discountFactor >= 1e-6:
                    qNextLines = self.qTable.getBatch(batchNewStates)
                for i in range(len(batchStates)):
                    actionId, reward = batchInfos[i]
                    if self.discountFactor >= 1e-6:
                        maxFutureReward = self.discountFactor*np.max(qNextLines[i])
                        qLines[i,actionId] += self.learningRate*(reward + maxFutureReward - qLines[i][actionId])
                    else:
                        qLines[i,actionId] += self.learningRate*(reward - qLines[i,actionId])
                self.qTable.setBatch(batchStates, qLines)
        else:
            for state, actionId, newState, reward in self.memories:
                qLine = self.qTable.get(state)
                qLine[actionId] += self.learningRate*(reward + self.discountFactor*np.max(self.qTable.get(newState)) - qLine[actionId])
                self.qTable.set(state, qLine)

    # Store all memories to a file
    def saveMemories(self, traceDir:str) -> None:
        os.makedirs(traceDir, exist_ok=True)
        tracePath = os.path.join(traceDir, '%s.pkl' % uuid.uuid4())
        with open(tracePath, 'wb+') as traceFile:
            pickle.dump(self.memories, traceFile)

    # Load all memories from a file (forget current memories)
    def loadMemories(self, traceDir:str) -> None:
        self.memories = []
        for traceName in os.listdir(traceDir):
            tracePath = os.path.join(traceDir, traceName)
            with open(tracePath, 'rb') as traceFile:
                self.memories += pickle.load(traceFile)

    # Forget every memories
    def forgetMemories(self) -> None:
        self.memories = []

def learn(agent:Tuple[Any,Any], iterationCount:int, tracePath:str, modelPath:str) -> None:
    (qTable, qLearning) = agent

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        # Restore the last session
        if os.path.exists(os.path.join(modelPath, 'lock')):
            print('Loading model from existing session...')
            saver.restore(sess, os.path.join(modelPath, 'model'))

        qTable.configure(sess, modelPath)
        qLearning.initState(None)

        print('Loading previous memories...')
        qLearning.loadMemories(tracePath)
        #print('Regularizing memories...')
        #qLearning.memoryRegularization()
        print('Entering in deep sleep...')
        for i in range(iterationCount):
            print(f'Iteration {i+1}...')
            qLearning.replayMemories(batchSize=64)

        # Save the session
        print('Saving model...')
        os.makedirs(modelPath, exist_ok=True)
        open(os.path.join(modelPath, 'lock'), 'wb+').close()
        saver.save(sess, os.path.join(modelPath, 'model'))

def play(agent:Tuple[Any,Any], game:Any, login:str, iterationCount:int, 
            tracePath:str, modelPath:str, record:bool=False) -> None:
    (qTable, qLearning) = agent

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        # Restore the last session
        if os.path.exists(os.path.join(modelPath, 'lock')):
            print('Loading model from existing session...')
            saver.restore(sess, os.path.join(modelPath, 'model'))

        game.login(login)

        game.startRound()
        s=time.time()
        game.printMap()

        qTable.configure(sess, modelPath)
        state = qTable.stateFromGameSession(game)
        qLearning.initState(state)

        #while True:
        for i in range(iterationCount):
            order = qLearning.react()
            #game.printMap()
            oldScore = game.playerInfos[game.playerId].score
            game.endRound(order)
            e=time.time()
            if not game.startRound():
                break
            print(f'Time: {e-s}')
            newScore = game.playerInfos[game.playerId].score
            reward = newScore-oldScore #-0.05
            state.print()
            print(f'order={order} | reward={reward}')
            s=time.time()
            virtualReward = remapReward(reward)
            state = qTable.stateFromGameSession(game)
            qLearning.update(state, virtualReward, register=record, learn=False)

        if record:
            qLearning.saveMemories(tracePath)

def main() -> int:
    cmdParser = argparse.ArgumentParser(
        description='A bot based on deep Q-network', 
        formatter_class=argparse.RawTextHelpFormatter
    )
    cmdParser.add_argument('-m', '--mode', dest='mode', metavar='MODE', choices=['play', 'record', 'learn'], default='play', 
                            help='set the bot mode (play, record or learn)')
    cmdParser.add_argument('-s', '--host', dest='host', metavar='HOST', default='127.0.0.1', 
                            help='server hostname (used to play/record)')
    cmdParser.add_argument('-p', '--port', dest='port', metavar='PORT', default=8889, type=int, 
                            help='server port (used to play/record)')
    cmdParser.add_argument('-l', '--login', dest='login', metavar='LOGIN', default='DeepDepth', 
                            help='bot login (used to play/record)')
    cmdParser.add_argument('-tp', '--trace-path', dest='tracepath', metavar='TRACEPATH', default='/tmp/masterbot/traces', 
                            help='directory of the learning trace (used to record/learn)')
    cmdParser.add_argument('-mp', '--model-path', dest='modelpath', metavar='MODELPATH', default='/tmp/masterbot/model', 
                            help='directory of the tensorflow model files (used in every modes)')
    cmdParser.add_argument('-c', '--iteration-count', dest='iterationcount', metavar='ITERATIONCOUNT', type=int, 
                            help='number of iteration to play/record or to learn', default=None)
    cmdArgs = cmdParser.parse_args()

    assert cmdArgs.mode not in ('play', 'record') or cmdArgs.port > 0

    if cmdArgs.iterationcount is None:
        if cmdArgs.mode in ('play', 'record'):
            cmdArgs.iterationcount = 500
        else:
            cmdArgs.iterationcount = 3

    # Note: to change over time (as the network learn)
    # 2.00: very random
    # 1.00: mainly random
    # 0.50: semi random
    # 0.10: sometime random
    # 0.05: very reward driven random (good for playing)
    # 0.00: 100% deterministic (biased)
    explorationTemperature = 0.05 if cmdArgs.mode=='play' else 0.2

    DeepQNetwork = QLearning[DeepNeuralQTable, DeepNeuralQTable.State]
    qTable = DeepNeuralQTable(careBonus=True, carePlayers=False, careShoots=False, radius=3, 
                                                    learningRate=3e-4, isTraining=cmdArgs.mode=='learn')
    qLearning = DeepQNetwork(qTable, learningRate=1.0, discountFactor=0.6, explorationTemperature=explorationTemperature)
    agent = (qTable, qLearning)

    if cmdArgs.mode in ['play', 'record']:
        gameSession = GameSession(cmdArgs.host, cmdArgs.port)
        play(agent, gameSession, cmdArgs.login, cmdArgs.iterationcount, cmdArgs.tracepath, cmdArgs.modelpath, cmdArgs.mode=='record')
    else:
        learn(agent, cmdArgs.iterationcount, cmdArgs.tracepath, cmdArgs.modelpath)

    return 0

if __name__ == '__main__':
    exit(main())

