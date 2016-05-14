"""
FILENAME: controller.py
controller.py is the client and SUMO is the server
"""

"""			DIRECTORIES & PATHS				"""
PORT = 8813

"""				LIBRARIES 					"""

import os, sys
import subprocess
import traci
import random
import pandas as pd
import numpy as np
import math
from numpy import random
import numpy.matlib
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from xml.dom import minidom
import heapq

import arrivalRateGen

import sklearn
from sklearn.cluster import KMeans
from sklearn import datasets
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import ggplot as gg

"""				PARAMETERS 					"""

secondsInDay = 24*60*60
secondsInHour = 60*60
totalDays = 1 # days to run simulation
alpha = 0.5 # learning rate
SL = "65546898" # ID of stoplight

# counters
currSod = 0
currPhaseID = 0
secsThisPhase = 0

# state objects and boolean helpers
phaseNum = 0
lastObjValue = 0
lastAction = 0
stepThru = 1
arrivalTracker = 0
waitingTime = 0
currState = 0
lastState = 0

# discretization parameters
numPhasesForAction = 4 # 8 including the yellow phases
numEdges = 4
numLanes = 8
numQueueSizeBuckets = 4
numwaitingBuckets = 4
hoursInDay = 24 #
numActions = 2 # 1 = switch to yellow phase; stay in current phase
secsPerInterval = 4
minPhaseTime = 4
maxPhaseTime = 36
yellowPhaseTime = 4

numStates = numPhasesForAction*(numQueueSizeBuckets*numwaitingBuckets)**numEdges


"""				COLLECTIONS					"""

QValues = np.random.rand(numStates,2) # all state action pairs
QProbs = np.ones((numStates,2))/2 # initialize randomly
QCounts = np.zeros((numStates, 2))
QAlphas = np.ones((numStates, 2))

# print 'QValues = ', QValues
# print 'QProbs = ', QProbs

# two lanes for each edge
listLanes = ['8949170_0', '8949170_1', \
			'-164126513_0', '-164126513_1',\
			'52016249_0', '52016249_1',\
			'-164126511_0', '-164126511_1']

listEdges = ['8949170', '-164126513', '52016249', '-164126511']
tupEdges = ('8949170', '-164126513', '52016249', '-164126511')
# (south (palm), north (palm), west (arboretum), east (arboretum))


# pick the thresholds from small, medium, long-sized queues
numPhasesForAction = 4 # 8 including the yellow phases
numEdges = 4
numLanes = 8
numQueueSizeBuckets = 3
numwaitingBuckets = 3

laneQueueTracker = {}
laneWaitingTracker = {}
laneNumVehiclesTracker = {}
# laneMeanSpeedTracker = {}
for lane in listLanes:
	laneQueueTracker[lane] = 0
	laneWaitingTracker[lane] = 0
	laneNumVehiclesTracker[lane] = 0
	# laneMeanSpeedTracker[lane] = 0

queueTracker = {}
waitingTracker = {}
numVehiclesTracker = {}
# meanSpeedTracker = {}
for edge in listEdges:
	queueTracker[edge] = 0
	waitingTracker[edge] = 0
	numVehiclesTracker[edge] = 0
	# meanSpeedTracker[edge] = 0

# queueBuckets = [3,6] # actually the boundaries of the buckets
# waitingBuckets = [35,120] # actually the boundaries of the buckets

# stateCols = ('phase', '8949170_q', '8949170_w', '-164126513_q', '-164126513_w',\
# 		 '52016249_q', '52016249_w', '-164126511_q','-164126511_w')

# dfStateMapping = pd.DataFrame(columns=stateCols)
# for p in range(numPhasesForAction):
# 	print 'p = ', p
# 	for e1q in range(numQueueSizeBuckets):
# 		for e1w in range (numwaitingBuckets):
# 			for e2q in range(numQueueSizeBuckets):
# 				for e2w in range (numwaitingBuckets):
# 					for e3q in range(numQueueSizeBuckets):
# 						for e3w in range (numwaitingBuckets):
# 							for e4q in range(numQueueSizeBuckets):
# 								for e4w in range (numwaitingBuckets):
# 									df = pd.DataFrame([[p, e1q, e1w, e2q, e2w, e3q, e3w,e4q, e4w]], columns = stateCols)
# 									dfStateMapping = dfStateMapping.append(df, ignore_index=True)
# dfStateMapping['stateNum'] = dfStateMapping.index
# dfStateMapping.to_csv('dfStateMapping.csv')

dfStateMapping = pd.DataFrame.from_csv('dfStateMapping.csv')

cols = ('8949170_q', '8949170_w', '-164126513_q', '-164126513_w',\
 		 '52016249_q', '52016249_w', '-164126511_q','-164126511_w')
dfQueueSizesWaitingTimes = pd.DataFrame(columns=cols)

"""				HELPER FUNCTIONS 				"""

def computeObjValue(queueTracker, waitingTracker):
	currObjValue = 0
	for key in listEdges:
		currObjValue -= ((1*queueTracker[key])**1.75 + (2*waitingTracker[key])**1.75) #TODO - include waitingTracker
	return currObjValue

# determine Thresholds (bucket boundaries) 0.75 percentiles of ALL queue sizes)
def assignStateNum(phaseNum, queueTracker, waitingTracker, queueBuckets, waitingBuckets):
	# assign each edge queue size to a corresponding bucket number
	for i in queueTracker.keys():
		assignedBucket = False
		for j in range(len(queueBuckets)):
			if not assignedBucket and queueTracker[i] <= queueBuckets[j]:
				queueTracker[i] = j
				assignedBucket = True
		if not assignedBucket:
			queueTracker[i] = len(queueBuckets)

	for i in waitingTracker.keys():
		assignedBucket = False
		for j in range(len(waitingBuckets)):
			if not assignedBucket and waitingTracker[i] < waitingBuckets[j]:
				waitingTracker[i] = j
				assignedBucket = True
		if not assignedBucket:
			waitingTracker[i] = len(waitingBuckets)

	# assign each edge waiting time to a correpsonding bucket number
	p = dfStateMapping['phase'] == phaseNum/2 # only 4 states where we are taking action 
	e1q = dfStateMapping['8949170_q'] == queueTracker['8949170']
	e1w = dfStateMapping['8949170_w'] == waitingTracker['8949170']
	e2q = dfStateMapping['-164126513_q'] == queueTracker['-164126513']
	e2w = dfStateMapping['-164126513_w'] == waitingTracker['-164126513']
	e3q = dfStateMapping['52016249_q'] == queueTracker['52016249']
	e3w = dfStateMapping['52016249_w'] == waitingTracker['52016249']
	e4q = dfStateMapping['-164126511_q'] == queueTracker['-164126511']	
	e4w = dfStateMapping['-164126511_w'] == waitingTracker['-164126511']	
	a = dfStateMapping['stateNum'][p & e1q & e1w & e2q & e2w & e3q & e3w & e4q & e4w]
	# print 'a = ', a
	# return a
	# print 'a.dtype = ', a.dtype
	return int(a.tolist()[0])
	# lookup number in table; return number corresponding to state (not numerically significant)

# zz = assignStateNum(phaseNum, queueTracker, waitingTracker, queueBuckets, waitingBuckets)
# print zz
# print zz + 3
# queueTracker['8949170'] = 10
# print 'assignStateNum = ', assignStateNum(phaseNum, queueTracker, waitingTracker, queueBuckets, waitingBuckets)

# Q-value update
def updateQValues(lastState, lastAction, currState, reward, alpha):
	QCounts[lastState, lastAction] += 1
	QAlphas[lastState, lastAction] = 1/(QCounts[lastState, lastAction])
	QValues[lastState, lastAction] = (1 - alpha)*QValues[lastState, lastAction] + QAlphas[lastState, lastAction]*(reward + gamma*max(QValues[currState,]))


def updateQProbs(lastState, lastAction, epsilon):
	numerator = np.exp(QValues[lastState, ]/epsilon)
	tempSum = np.sum(numerator)
	denominator = np.array([tempSum, tempSum])
	QProbs[lastState, ] = np.divide(numerator, denominator)

dfObjValsMaster = pd.DataFrame()
dfQueueTrackerMaster = pd.DataFrame()
dfWaitingTrackerMaster = pd.DataFrame()
dfActions = pd.DataFrame()


# make a dict of numpy arrays
stateData = {}
actionPhases = [0,2,4,6]
for i in range(24):
	stateData[i] = {}
	for j in range(len(actionPhases)):
		stateData[i][actionPhases[j]] = np.array([])

global dictClusterObjects
dictClusterObjects = {}
global numClustersTracker
numClustersTracker = {}
for i in range(24): #hod
	dictClusterObjects[i] = {}
	numClustersTracker[i] = {}
	for j in actionPhases:
		dictClusterObjects[i][j] = None
		numClustersTracker[i][j] = 0

# print 'dictClusterObjects = ', dictClusterObjects
# print 'numClustersTracker = ', numClustersTracker

global mapDiscreteStates 
mapDiscreteStates = {}

global listMeanObjVals, listMedianObjVals, listMinObjVals
listMeanObjVals = []
listMedianObjVals = []
listMinObjVals = []

def learnDiscretization(daysToTrain):
	# """							SIMULATION 					"""
	dynamic = 0
	day = 0
	totalDays = daysToTrain

	# learning rates and discount factors
	gamma = 0.95
	# epsilons = [1, 0.99, 0.97, 0.93, 0.91, 0.89, 0.80, 0.75, 0.70, 0.6, 0.5, 0.45, 0.3, 0.20, 0.18, 0.17, 0.15, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01, 0.01]

	# print 'len(epsilons) = ', len(epsilons)

	for day in range(totalDays): # range(len(epsilons)+1): #len(alphas)
		
		# generate the random route schedule for the day
		arrivalRateGen.writeRoutes(day+1)

		sumoProcess = subprocess.Popen(['sumo-gui.exe', "-c", "palm.sumocfg", \
			"--remote-port", str(PORT)], stdout=sys.stdout, stderr=sys.stderr)

		# sumoProcess = subprocess.Popen(['sumo.exe', "-c", "palm.sumocfg", "--fcd-output", \
		# 	"out.fcd.xml", "--tripinfo-output", "out.trip.xml", "--summary", "out.summary.xml", "--queue-output", "out.queue.xml", "--remote-port", str(PORT)], stdout=sys.stdout, stderr=sys.stderr)

		traci.init(PORT)

		dfObjVals = pd.DataFrame()
		dfQueueTracker = pd.DataFrame()
		dfWaitingTracker = pd.DataFrame()

		action = 0 # number of seconds over minimum that we decided to take in 5 buckets (0,1,2,4)
		lastAction = 0
		hod = 0
		currSod = 0
		epsilon = 1 # TODO - change epsilon dynamically?
		currPhaseID = 0
		secsThisPhase = 0
		
		while currSod < secondsInDay: 

			if currPhaseID == int(traci.trafficlights.getPhase(SL)) and currSod != 0: # if phase HAS NOT changed
				secsThisPhase += 1 # increase the seconds in the currentPhase	
			else: # IF THE PHASE HAS CHANGED
				secsThisPhase = 0
				currPhaseID = int(traci.trafficlights.getPhase(SL)) 

			# STORE INFORMATION TO DETERMINE IF ITS TIME TO MAKE A DECISION

			# ARRAY TO MAP STATE:
			# (2) Hour of day (24)
			# (1) Light phase for decision (4) getPhase
			# (2) Num stopped cars X 4 getLastStepHaltingNumber
			# (3) Num vehicles in lane getLastStepVehicleNumber
			# (4) Cum waiting time x 4 getWaitingTime
			# (5) Last step mean speed X 4 getLastStepMeanSpeed

			if currPhaseID%2 == 0 and secsThisPhase == 0: # only collecting data when we come to the end of a yellow phase

				#============  HOD
				if hod != currSod/secondsInHour:
					hod = int(currSod/secondsInHour)
					print 'observation day = ', day
					print 'hod = ', hod
					# print 'len(stateData[h][1]) = ', len(stateData[hod][1])
					# print 'int(sum(np.std(stateData[h][a], axis = 0))) = ', int(sum(np.std(stateData[hod][1], axis = 0)))
					# print 'len(stateData[h][3]) = ', len(stateData[hod][3])
					# print 'int(sum(np.std(stateData[h][a], axis = 0))) = ', int(sum(np.std(stateData[hod][3], axis = 0)))
					# print 'len(stateData[h][5]) = ', len(stateData[hod][5])
					# print 'int(sum(np.std(stateData[h][a], axis = 0))) = ', int(sum(np.std(stateData[hod][5], axis = 0)))
					# print 'len(stateData[h][7]) = ', len(stateData[hod][7])
					# print 'int(sum(np.std(stateData[h][a], axis = 0))) = ', int(sum(np.std(stateData[hod][7], axis = 0)))

				#============ currPhaseID

				#================= count halted vehicles (4 elements)

				for lane in listLanes:
					laneQueueTracker[lane] = traci.lane.getLastStepHaltingNumber(str(lane))
					# laneQueueTracker[lane] = traci.lane.getLastStepVehicleNumber(str(lane))

				for edge in queueTracker.keys():
					queueTracker[edge] = laneQueueTracker[str(edge) + '_' + str(0)] + laneQueueTracker[str(edge) + '_' + str(1)]
					# inherently, we assume balancing here
					# TODO - later chage this to only track 

				# df = pd.DataFrame([[currSod,  queueTracker['8949170'], queueTracker['-164126513'], queueTracker['52016249'], queueTracker['-164126511']]])
				# dfQueueTracker = dfQueueTracker.append(df, ignore_index = True)

				# ================ count vehicles in lane

				for lane in listLanes:
					laneNumVehiclesTracker[lane] = traci.lane.getLastStepVehicleNumber(str(lane))

				for edge in numVehiclesTracker.keys():
					numVehiclesTracker[edge] = laneNumVehiclesTracker[str(edge) + '_' + str(0)] + laneNumVehiclesTracker[str(edge) + '_' + str(1)]

				# ================ cum waiting time in minutes

				for lane in listLanes:
					laneWaitingTracker[lane] = traci.lane.getWaitingTime(str(lane))/60
				for edge in waitingTracker.keys():
					waitingTracker[edge] = laneWaitingTracker[str(edge) + '_' + str(0)] + laneWaitingTracker[str(edge) + '_' + str(1)]

				# df = pd.DataFrame([[currSod,  waitingTracker['8949170'], waitingTracker['-164126513'], waitingTracker['52016249'], waitingTracker['-164126511']]])
				# dfWaitingTracker = dfWaitingTracker.append(df, ignore_index = True)

				# ================ mean speed

				# for lane in listLanes:
				# 	laneMeanSpeedTracker[lane] = traci.lane.getLastStepMeanSpeed(str(lane))
				# for edge in meanSpeedTracker.keys():
				# 	meanSpeedTracker[edge] = (laneMeanSpeedTracker[str(edge) + '_' + str(0)] + laneMeanSpeedTracker[str(edge) + '_' + str(1)])/2

				# ============== CREATE A NEW ENTRY FOR OUR STATE TRACKER

				stateDataEntry = []
				for edge in listEdges:
					stateDataEntry.append(queueTracker[edge])
				for edge in listEdges:
					stateDataEntry.append(numVehiclesTracker[edge])
				for edge in listEdges:
					stateDataEntry.append(waitingTracker[edge])
				# for edge in listEdges:
				# 	stateDataEntry.append(meanSpeedTracker[edge])

				if len(stateData[hod][currPhaseID]) == 0:
					stateData[hod][currPhaseID] = np.array(stateDataEntry)
				else:
					stateData[hod][currPhaseID] = np.vstack([stateData[hod][currPhaseID], stateDataEntry])


				# TRACK OBJECTIVE FUNCTION

				currObjValue = computeObjValue(queueTracker, waitingTracker)
				df = pd.DataFrame([[currSod, currObjValue]]) # todo - fix so plot shows the second of the day
				dfObjVals = dfObjVals.append(df, ignore_index=True)

				# print 'currPhaseID = ', currPhaseID
				# print 'secsThisPhase = ', secsThisPhase
				# print 'currSod = ', currSod
				# print 'hod = ', hod
				# print 'queueTracker = ', queueTracker
				# print 'waitingTracker = ', waitingTracker

			# # we can make a decision
			# if currPhaseID%2 == 0 and secsThisPhase%secsPerInterval == 0 and secsThisPhase >=4: # and currSod > 20000:
			# 	# print 'currPhaseID = ', currPhaseID
			# 	# print 'secsThisPhase = ', secsThisPhase

			# 	# arrayObjVals = np.append(arrayObjVals, currObjValue)
			# 	# arrayQueueSizes = np.append(arrayQueueSizes, queueTracker.values())
			# 	# if day > 0:
			# 	# 	dynamic = 1
			# 	if dynamic:
			# 	# CONTROL ACTION
			# 		phaseNum = traci.trafficlights.getPhase(SL)
			# 		currState = assignStateNum(phaseNum, queueTracker, waitingTracker, queueBuckets, waitingBuckets) 

					# # reward = objective value; we want it to be as close to zero as possible (will always be negative)
					# reward = currObjValue - lastObjValue
					# lastObjValue = currObjValue

					# updateQValues(int(lastState), int(lastAction), int(currState), reward, alpha) # alpha controls whether we explore or exploit
					# updateQProbs(int(lastState), int(lastAction), epsilon)

					# # pick action
					# unigen = random.random() 
					# if QProbs[currState,0] < unigen or secsThisPhase == 0:
					# 	action = 0 # stay in this phase
					# else:
					# 	action = 1 # change phases; transition to the next yellow phase immediately
					# # TODO - totally change the actions to be the number of seconds (# of 4-second time chunks we are choosing for the cycle)

					# 	traci.trafficlights.setPhase(SL, (int(currPhaseID) + 1)%8)
					# 	#TODO - tell the light how long to run for; not just whether or not to change

					# 	# print 'int(traci.trafficlights.getPhase(SL)) = ', int(traci.trafficlights.getPhase(SL)) 
					# df = pd.DataFrame([[currSod, secsThisPhase, currPhaseID, currState, lastState, action, currObjValue, lastObjValue, reward]]) # todo - fix so plot shows the second of the day
					# dfActions = dfActions.append(df, ignore_index=True)

					# lastState = currState
					# lastAction = action
			currSod += 1
			# print traci.vehicle.getIDList()
			traci.simulationStep()
		# print stateData
		# totalSize = 0
		# for i in actionPhases:
		# 	totalSize += len(stateData[0][i])
		# print 'totalSize = ', totalSize

		# for i in actionPhases:
		# 	print 'np.std(stateData[0][i], axis = 0) = ', np.std(stateData[0][i], axis = 0)
		# 	print 'sumvariability = ', sum(np.std(stateData[0][i], axis = 0))
		# 	print 'np.std(stateData[0][i], axis = 0) = ', np.std(stateData[0][i], axis = 0)
		# 	print 'sumstd = ', sum(np.std(stateData[0][i], axis = 0))
		# 	totalSize += len(stateData[0][i])

		traci.close() # TODO - fix; need to figure out how to plot multiple with different x-axes

		print 'dfObjVals = ', dfObjVals

		dfMean = dfObjVals.mean(axis = 0)
		meanObjVal = dfMean[1]

		dfMedian = dfObjVals.median(axis = 0)
		medianObjVal = dfMedian[1]
		# vMedian = dfMedian[1]

		dfMin = dfObjVals.min(axis=0)
		minObjVal = dfMin[1]

		listMeanObjVals.append(meanObjVal)
		listMedianObjVals.append(medianObjVal)
		listMinObjVals.append(minObjVal)

	for h in range(hoursInDay): #TODo - change to hoursInDay
		for a in actionPhases:
			numClustersTracker[h][a] = int(sum(np.std(stateData[h][a], axis = 0))) #
			print 'h = ', h
			print 'a = ', a
			print 'numClustersTracker[h][a] = ', numClustersTracker[h][a]

			dictClusterObjects[h][a] = KMeans(n_clusters = numClustersTracker[h][a])
			dictClusterObjects[h][a].fit(stateData[h][a])
			# result = dictClusterObjects[0][a].predict(stateData[0][a])
			# print 'result = ', result
			# print 'max(result) = ', max(result)

	# print 'hod = ', hod
	# print 'dictClusterObjects = ', dictClusterObjects
	print 'numClustersTracker = ', numClustersTracker
	totalClusters = 0
	for h in range(hoursInDay): #TODO - change to hoursInDay
		for a in actionPhases:
			totalClusters += numClustersTracker[h][a]

	print 'totalClusters = ', totalClusters

	stateCounter = 0
	for h in range(hoursInDay): #TODO - change to hoursInDay
		mapDiscreteStates[h] = {}
		for a in actionPhases:
			mapDiscreteStates[h][a] = {}
			for c in range(numClustersTracker[h][a]):
				mapDiscreteStates[h][a][c] = stateCounter
				stateCounter += 1
	print 'stateCounter = ', stateCounter

def getMapDiscreteStates():
	return mapDiscreteStates

def getInvMapDiscreteStates():
	invMapDiscreteStates = {}
	for h in range(hoursInDay):
		for a in actionPhases:
			for c in range(numClustersTracker[h][a]):
				invMapDiscreteStates[mapDiscreteStates[h][a][c]] = {'hod':h, 'phase':a, 'num':c}
	print getInvMapDiscreteStates
	return invMapDiscreteStates


def getDictClusterObjects():
	return dictClusterObjects

def getNumClustersTracker():
	return numClustersTracker

def plotClusterHistograms():
	dfClusters = pd.DataFrame.from_dict(numClustersTracker, orient = 'index')
	dfClusters.columns = ['phase 0', 'phase 2', 'phase 4', 'phase 6']
	print dfClusters
	dfClusters.plot(kind = 'bar', stacked = True)
	plt.xlabel('hour of day')
	plt.ylabel('number discrete states chosen')
	plt.title('Discrete States Selected By K-Means Clustering for each (hour, phase)')
	plt.show()

def plotQueueSizes():
	pass

def plotWaitingTimes():
	pass

def getBaselineMean():
	return np.mean(listMeanObjVals)

def getBaselineMedian():
	return np.mean(listMedianObjVals)

def getBaselineMin():
	return np.mean(listMinObjVals)




# 	# print 'dfObjValsMaster = ', dfObjValsMaster
# 	# dfObjVals.columns = ['second', 'day ' + str(day) + '; eps = ' + str(epsilon)]
# 	# dfObjVals['day ' + str(day) + '; eps = ' + str(epsilon)] = \
# 	# 	pd.ewma(dfObjVals['day ' + str(day) + '; eps = ' + str(epsilon)], span=600)
# 	# dfObjVals.columns = ['second', 'day ' + str(day) + '; eps = ' + str(epsilon)]

# 	if day < 0:
# 		dfActions.columns = ['currSod', 'secsThisPhase', 'currPhaseID', 'currState', 'lastState', 'action', 'currObjValue', 'lastObjValue', 'reward']
# 		dfActions.to_csv('dfActions' + str(day) + '.csv')

# 	dfQueueTracker.columns = ['hour', 'south', 'north', 'west', 'east']
# 	dfQueueTracker['hour'] = dfQueueTracker['hour']/secondsInHour
# 	dfQueueTracker.to_csv('dfQueueTracker' + str(day) + '.csv')

# 	dfWaitingTracker.columns = ['hour','south', 'north', 'west', 'east']
# 	dfWaitingTracker['hour'] = dfWaitingTracker['hour']/secondsInHour
# 	dfWaitingTracker.to_csv('dfWaitingTracker' + str(day) + '.csv')

# 	dfObjVals.columns = ['second', 'day ' + str(day)]
# 	dfObjVals['day ' + str(day)] = \
# 		pd.ewma(dfObjVals['day ' + str(day)], span=1200)
# 	dfObjVals.columns = ['second', 'day ' + str(day)]
# 	dfObjVals.to_csv('dfObjVals' + str(day) + '.csv')

# 	# print 'dfObjVals = ', dfObjVals
# 	if day == 0:
# 		dfObjValsMaster = dfObjVals
# 		dfObjVals.columns = ['second', 'static policy']
# 	else:
# 		dfObjValsMaster = dfObjValsMaster.merge(dfObjVals, on = 'second')


# 	print 'QValues = ', QValues
# 	np.savetxt('QValues.txt',QValues)
# 	print 'QProbs = ', QProbs
# 	np.savetxt('QProbs.txt',QProbs)
# 	print 'QAlphas = ', QAlphas
# 	np.savetxt('QAlphas.txt', QAlphas)
# 	print 'QCounts = ', QCounts
# 	np.savetxt( 'QCounts.txt', QCounts)

# listEdges = ['south', 'north', 'west', 'east']

# dfQueueTracker.columns = ['hour', 'south', 'north', 'west', 'east']
# dfQueueTracker['hour'] = dfQueueTracker['hour']/secondsInHour
# dfQueueTracker.to_csv('dfQueueTracker.csv')
# for i in listEdges:
# 	dfQueueTracker[i] = pd.ewma(dfQueueTracker[i], span = 1200)
# dfQueueTracker.to_csv('dfWaitingTracker.csv')

# dfWaitingTracker.columns = ['hour','south', 'north', 'west', 'east']
# dfWaitingTracker['hour'] = dfWaitingTracker['hour']/secondsInHour
# for i in listEdges:
# 	dfWaitingTracker[i] = pd.ewma(dfWaitingTracker[i], span = 1200)
# dfWaitingTracker.to_csv('dfWaitingTracker.csv')

# dfObjValsMaster['second'] = dfObjValsMaster['second']/secondsInHour
# dfObjValsMaster.to_csv('dfObjValsMaster.csv')

# dfObjValsMaster.plot(x = 'second')
# plt.xlabel('hour')
# plt.title('Moving Average of Objective Function Value (Static Policy)')

# dfQueueTracker.plot(x = 'hour')
# plt.xlabel('hour')
# plt.title('Moving Average of Queue Tracker (Static Policy)')

# dfWaitingTracker.plot(x = 'hour')
# plt.xlabel('hour')
# plt.title('Moving Average Waiting Time By Edge (Static Policy)')

# plt.show()