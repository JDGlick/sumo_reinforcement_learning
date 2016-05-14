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
import math

import pandas as pd
import numpy as np
from numpy import random
import numpy.matlib
import matplotlib.pyplot as plt
import ggplot as gg

import xml.etree.ElementTree as ET
from xml.dom import minidom

import heapq

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import arrivalRateGen
import getDiscreteStates

"""				PARAMETERS 					"""

secondsInHour = 60*60
hoursInDay = 24
secondsInDay = hoursInDay*secondsInHour

SL = "65546898" # ID of stoplight

# discretization parameters
numPhasesForAction = 4 # 8 including the yellow phases
actionPhases = [0,2,4,6] # even phases are all yellow phases (0,2,4,6)

numEdges = 4
numLanes = 8

minPhaseTime = 4
maxPhaseTime = 36
yellowPhaseTime = 4

actionsForStraightPhase = [10, 14, 18, 22, 26, 30, 34, 38] # number of seconds to run an actionPhase
actionsForLeftPhase = [3,4,5,6,7,8,9,10]

global numActions
numActions = len(actionsForStraightPhase)
numActionsForStraightPhase = len(actionsForStraightPhase) #10
numActionsForLeftPhase = len(actionsForLeftPhase)


"""				STATE DISCRETIZATION / INITIAL LEARNING 		"""

getDiscreteStates.learnDiscretization(3)
getDiscreteStates.plotClusterHistograms()
dictClusterObjects = getDiscreteStates.getDictClusterObjects() # IN: hour (int), phase (int); OUT: stateSubID
mapDiscreteStates = getDiscreteStates.getMapDiscreteStates() # IN, hour (int), phase (int), stateSubID (int); OUT: stateID
invMapDicreteStates = getDiscreteStates.getInvMapDiscreteStates()
numClustersTracker = getDiscreteStates.getNumClustersTracker()
numStates = 0
for h in range(hoursInDay):
	for a in actionPhases:
		numStates += numClustersTracker[h][a]
print 'stateCounter = ', numStates

global baselineMean, baselineMedian, baselineMin 

baselineMean = getDiscreteStates.getBaselineMean()
print 'baselineMean = ', baselineMean
baselineMedian = getDiscreteStates.getBaselineMedian()
print 'baselineMedian = ', baselineMedian
baselineMin = getDiscreteStates.getBaselineMin()
print 'baselineMin = ', baselineMin 

"""				COLLECTIONS					"""

global QValues, QCounts, QProbs, QAlphas
QValues = np.zeros((numStates,numActions)) # all state action pairs
QCounts = np.zeros((numStates, numActions)) # some of these will always be null
QProbs = np.ones((numStates,numActions))/numActions # given state s, what is the probability that I should take the next action
# TODO - we need to apply some priors here by preventing certain actions from being taken
QAlphas = np.ones((numStates, numActions))
# same thing here; 

# two lanes for each edge
listLanes = ['8949170_0', '8949170_1', \
			'-164126513_0', '-164126513_1',\
			'52016249_0', '52016249_1',\
			'-164126511_0', '-164126511_1']

listEdges = ['8949170', '-164126513', '52016249', '-164126511']
tupEdges = ('8949170', '-164126513', '52016249', '-164126511')
# (south (palm), north (palm), west (arboretum), east (arboretum))

laneQueueTracker = {}
laneWaitingTracker = {}
laneNumVehiclesTracker = {}
for lane in listLanes:
	laneQueueTracker[lane] = 0
	laneWaitingTracker[lane] = 0
	laneNumVehiclesTracker[lane] = 0

queueTracker = {}
waitingTracker = {}
numVehiclesTracker = {}
for edge in listEdges:
	queueTracker[edge] = 0
	waitingTracker[edge] = 0
	numVehiclesTracker[edge] = 0

stateCols = ('phase', '8949170_q', '8949170_w', '-164126513_q', '-164126513_w',\
		 '52016249_q', '52016249_w', '-164126511_q','-164126511_w')

"""				HELPER FUNCTIONS 				"""

def computeObjValue(queueTracker, waitingTracker):
	currObjValue = 0
	for key in listEdges:
		currObjValue -= ((1*queueTracker[key])**1.75 + (2*waitingTracker[key])**1.75) #TODO - include waitingTracker
	return currObjValue

# TODO - these are the values for beta and theta that we need to select for the objective function
# plot this on the 3D plot and see if it makes sense to the decision maker

def getStateID(currHod, phase, queueTracker, numVehiclesTracker, waitingTracker):
	stateData = []
	for edge in listEdges:
		stateData.append(queueTracker[edge])
	for edge in listEdges:
		stateData.append(numVehiclesTracker[edge])
	for edge in listEdges:
		stateData.append(waitingTracker[edge])
	stateData = np.array(stateData)
	# print 'stateData = ', stateData
	stateSubID = int(dictClusterObjects[currHod][phase].predict(stateData))
	# print 'subStateID = ', stateSubID
	stateID = mapDiscreteStates[currHod][phase][stateSubID]
	# print 'stateID = ', stateID
	return stateID

# Q-value update
def updateQValues(lastStateID, lastAction, currStateID, reward):
	QValues[lastStateID, lastAction] = (1 - QAlphas[lastStateID, lastAction])*QValues[lastStateID, lastAction] + QAlphas[lastStateID, lastAction]*(reward + gamma*max(QValues[currStateID,]))
	QCounts[lastStateID, lastAction] += 1
	# print 'QAlphas[lastStateID, lastAction] before = ',  QAlphas[lastStateID, lastAction]
	QAlphas[lastStateID, lastAction] = 1/(QCounts[lastStateID, lastAction])
	# print 'QAlphas[lastStateID, lastAction] after = ',  QAlphas[lastStateID, lastAction]

def updateQProbs(lastStateID, lastAction):
	# print 'np.sum(QCounts[lastStateID,]) = ', np.sum(QCounts[lastStateID,])
	# print 'np.sum(QCounts[lastStateID,]) = ', np.sum(QCounts[lastStateID,])
	# print 'np.sum(QValues[lastStateID,]) = ', np.sum(QValues[lastStateID,])
	if np.sum(QCounts[lastStateID,]) == 0 or np.sum(QValues[lastStateID,]) == 0:
		tau = 1
	else:
		# print '(-(np.mean(QValues[lastStateID,]))) = ', (-(np.mean(QValues[lastStateID,])))
		# print '(np.mean(QCounts[lastStateID,])) = ', (np.mean(QCounts[lastStateID,]))
		tau = (-(np.mean(QValues[lastStateID,])))/(np.mean(QCounts[lastStateID,]))
	# print 'tau = ', tau
	numerator = np.exp(QValues[lastStateID, ]/tau)
	tempSum = np.sum(numerator)
	denominator = np.array([tempSum, tempSum, tempSum, tempSum, tempSum, tempSum, tempSum, tempSum])
	QProbs[lastStateID, ] = np.divide(numerator, denominator)

# initial dataframes which will be able to store performance data over different days
dfObjValsMaster = pd.DataFrame()
dfObjValsSummaryMaster = pd.DataFrame
dfQueueTrackerMaster = pd.DataFrame()
dfWaitingTrackerMaster = pd.DataFrame()

# check to see what the actions are
dfActions = pd.DataFrame()


"""							SIMULATION 					"""

currSod = 0
currPhaseID = 0
secsThisPhase = 0

# state objects and boolean helpers
phaseNum = 0
lastObjValue = 0
lastAction = 0
arrivalTracker = 0
waitingTime = 0
currStateID = 0
lastStateID = 0

dynamic = 1
totalDays = 60

# learning rates and discount factors
gamma = 0.95 # to do - drop gamma down a little bit?

for day in range(totalDays): 

	# generate the random route schedule for the day
	arrivalRateGen.writeRoutes(day+1)

	sumoProcess = subprocess.Popen(['sumo-gui.exe', "-c", "palm.sumocfg", \
		"--remote-port", str(PORT)], stdout=sys.stdout, stderr=sys.stderr)

# 	# sumoProcess = subprocess.Popen(['sumo.exe', "-c", "palm.sumocfg", "--fcd-output", \
# 	# 	"out.fcd.xml", "--tripinfo-output", "out.trip.xml", "--summary", "out.summary.xml", "--queue-output", "out.queue.xml", "--remote-port", str(PORT)], stdout=sys.stdout, stderr=sys.stderr)

	traci.init(PORT)

	dfObjVals = pd.DataFrame()
	dfObjValsMasterSummary = pd.DataFrame
	dfQueueTracker = pd.DataFrame()
	dfWaitingTracker = pd.DataFrame()

	dfNumVehiclesTracker = pd.DataFrame()
	dfActions = pd.DataFrame()

 	lastAction = 0
	currHod = 0
	currSod = 0
	
	while currSod < secondsInDay: 

		# update currHod
		if currHod != currSod/secondsInHour:
			currHod = int(currSod/secondsInHour)
			print 'training day = ', day
			print 'currHod = ', currHod

		# DETERMINE IF ITS TIME TO MAKE A DECISION
		if currPhaseID == int(traci.trafficlights.getPhase(SL)) and currSod != 0: # if phase HAS NOT changed
			secsThisPhase += 1 # increase the seconds in the currentPhase	
		else: # IF THE PHASE HAS CHANGED
			secsThisPhase = 0
			currPhaseID = int(traci.trafficlights.getPhase(SL)) 

		# just came out of yellow and about to start our new phase;
		# we need to collect our reward from the last decision
		# and make a new decision about how long we want this phase to be

		if currPhaseID%2 == 0 and secsThisPhase == 0: # only collecting data when we come to the end of a yellow phase

			# update our trackers so we know what state the environment is in
			# TODO - later come back and have the updates detect the state only according to what we think the environment is; we would also have to update the getDiscreteStates.py file

			#================= COUNT HALTED VEHICLES (I.E. QUEUE SIZE) (4 elements)

			for lane in listLanes:
				laneQueueTracker[lane] = traci.lane.getLastStepHaltingNumber(str(lane))

			for edge in queueTracker.keys():
				queueTracker[edge] = laneQueueTracker[str(edge) + '_' + str(0)] + laneQueueTracker[str(edge) + '_' + str(1)]
				# TOTAL CARS IN QUEUE; DOESN'T REALLY COMPUTE THE LENGTH

			df = pd.DataFrame([[currSod,  queueTracker['8949170'], queueTracker['-164126513'], queueTracker['52016249'], queueTracker['-164126511']]])
			dfQueueTracker = dfQueueTracker.append(df, ignore_index = True)

			# ================ count vehicles in lane

			for lane in listLanes:
				laneNumVehiclesTracker[lane] = traci.lane.getLastStepVehicleNumber(str(lane))

			for edge in numVehiclesTracker.keys():
				numVehiclesTracker[edge] = laneNumVehiclesTracker[str(edge) + '_' + str(0)] + laneNumVehiclesTracker[str(edge) + '_' + str(1)]

			df = pd.DataFrame([[currSod,  numVehiclesTracker['8949170'], numVehiclesTracker['-164126513'], numVehiclesTracker['52016249'], numVehiclesTracker['-164126511']]])
			dfNumVehiclesTracker = dfNumVehiclesTracker.append(df, ignore_index = True)

			# ================ cum waiting time in minutes

			for lane in listLanes:
				laneWaitingTracker[lane] = traci.lane.getWaitingTime(str(lane))/60

			for edge in waitingTracker.keys():
				waitingTracker[edge] = laneWaitingTracker[str(edge) + '_' + str(0)] + laneWaitingTracker[str(edge) + '_' + str(1)]

			df = pd.DataFrame([[currSod,  waitingTracker['8949170'], waitingTracker['-164126513'], waitingTracker['52016249'], waitingTracker['-164126511']]])
			dfWaitingTracker = dfWaitingTracker.append(df, ignore_index = True)

			# GET THE stateID we are in right now

			currStateID = getStateID(currHod, currPhaseID, queueTracker, numVehiclesTracker, waitingTracker)

			# collect the currObjValue (reward) based on the current state; larger is better

			currObjValue = computeObjValue(queueTracker, waitingTracker)

			df = pd.DataFrame([[currSod, currObjValue]]) 
			dfObjVals = dfObjVals.append(df, ignore_index=True)

			if dynamic and currSod != 0:
			# CONTROL ACTION
				# print 'currPhaseID = ', currPhaseID
				# print 'currStateID = ', currStateID
				# print 'secsThisPhase = ', secsThisPhase
				# print 'currHod = ', currHod
				# print 'currSod = ', currSod
				# print 'queueTracker = ', queueTracker
				# print 'numVehiclesTracker = ', numVehiclesTracker
				# print 'waitingTracker = ', waitingTracker
				# print 'lastObjValue = ', lastObjValue
				# print 'lastAction = ', lastAction

				# record the state -> action -> state combination which led to a particular reward in the QTable
				# print 'QCounts[lastStateID, ] before = ', QCounts[lastStateID, ]
				# print 'QAlphas[lastStateID, ] before = ', QAlphas[lastStateID, ]
				# print 'QValues[lastStateID, ] before = ', QValues[lastStateID, ]

				updateQValues(lastStateID, lastAction, currStateID, lastObjValue) 

				# print 'QCounts[lastStateID, ] after = ', QCounts[lastStateID, ]
				# print 'QAlphas[lastStateID, ] after = ', QAlphas[lastStateID, ]
				# print 'QValues[lastStateID, ] after = ', QValues[lastStateID, ]

				# with QValues updated, update our probability distribution for picking certain actions given a particular state

				# print 'QProbs[lastStateID, ] before = ', QProbs[lastStateID,]

				updateQProbs(int(lastStateID), int(lastAction))
				
				# print 'QProbs[lastStateID, ] after = ', QProbs[lastStateID,]

# 				# pick action
				unigen = random.random() 
				# print 'unigen = ', unigen
				probsActions =  np.cumsum(QProbs[currStateID,])
				# print 'probsActions = ', probsActions

				for i in range(len(probsActions)):
					if unigen <= probsActions[i]:
						action = i # i maps to some second value that we want to run this phase for
						# # print 'action = ', action
						# if currPhaseID == 0 or currPhaseID == 4:
						# 	# print 'actionsForLeftPhase[action] = ', actionsForLeftPhase[action]
						# else:
						# 	# print 'actionsForStraightPhase[action] = ', actionsForStraightPhase[action]
						break


				# print what the current lenth of the phase (this is the default in palm.new.net.xml)
				# print 'currPhaseID = ', int(traci.trafficlights.getPhase(SL)) # we should still be in the same phase
				# print 'currPhaseDuration = ', int(traci.trafficlights.getPhaseDuration(SL)) # we should still be in the same phase

				# tell the light how long to run the currentPhase
				if currPhaseID == 0 or currPhaseID == 4:
					traci.trafficlights.setPhaseDuration(SL, actionsForLeftPhase[action])
				else:
					traci.trafficlights.setPhaseDuration(SL, actionsForStraightPhase[action])


				# as a check, make sure that the traffic light listened to us
				# print 'currPhaseID after action= ', int(traci.trafficlights.getPhase(SL)) # we should still be in the same phase
				# print 'currPhaseDuration after action = ', int(traci.trafficlights.getPhaseDuration(SL)) # we should still be in the same phase

				# record what decision was just made here for post-simulation diagnostics
				df = pd.DataFrame([[currHod, currSod, lastStateID, lastObjValue, lastAction, currStateID, currObjValue, action]]) 
				# todo - fix so plot shows the second of the day
				dfActions = dfActions.append(df, ignore_index=True)

				lastStateID = currStateID
				lastAction = action
				lastObjValue = currObjValue
		currSod += 1
		traci.simulationStep()
	traci.close() # TODO - fix; need to figure out how to plot multiple with different x-axes

	# MERGE NEW OBJECTIVE FUNCTION DATA

	dfObjVals.columns = ['hour', 'day ' + str(day)]
	dfObjVals['hour'] = dfObjVals['hour']/(1.0*secondsInHour)
	dfObjVals.to_csv('dfObjVals' + str(day) + '.csv')

	dfMean = dfObjVals.mean(axis=0)
	print 'dfMean = ', dfMean
	print 'dfMean[1] = ', dfMean[1]
	# vMean = dfMean[1]

	dfMedian = dfObjVals.median(axis = 0)
	print 'dfMedian = ', dfMedian
	print 'dfMedian[1] = ', dfMedian[1]
	# vMedian = dfMedian[1]

	dfMin = dfObjVals.min(axis=0)
	print 'dfMin = ', dfMin
	print 'dfMin[1] = ', dfMin[1]
	# vMin = dfMin[1]

	df = pd.DataFrame([[str(day), dfMean[1], dfMedian[1], dfMin[1]]])
	df.columns = ['day', 'mean', 'median', 'min']

	if day == 0:
		dfObjValsSummaryMaster = df
	else:
		dfObjValsSummaryMaster = dfObjValsSummaryMaster.append(df, ignore_index=True)	
		dfObjValsSummaryMaster.columns = ['day', 'mean', 'median', 'min']
	
	dfObjValsSummaryMaster.to_csv('dfObjValsSummaryMaster' + str(day) + '.csv')

	plt.subplot(3,1,1)
	plt.plot(dfObjValsSummaryMaster['day'], dfObjValsSummaryMaster['mean'], color = 'green', label = 'mean')
	plt.axhline(y = baselineMean, color = 'black', label = 'static')
	plt.xlabel('day')
	plt.ylabel('obj val')
	plt.title('Objective value mean')
	plt.tight_layout()

	plt.subplot(3,1,2)
	plt.plot(dfObjValsSummaryMaster['day'], dfObjValsSummaryMaster['median'], color = 'blue', label = 'median')
	plt.axhline(y = baselineMedian, label = 'static', color = 'black')
	plt.xlabel('day')
	plt.ylabel('obj val')
	plt.title('Objective value median')
	plt.tight_layout()

	plt.subplot(3,1,3)
	plt.plot(dfObjValsSummaryMaster['day'], dfObjValsSummaryMaster['min'], color = 'red', label = 'min')
	plt.axhline(y = baselineMin, label = 'static', color = 'black')
	plt.xlabel('day')
	plt.ylabel('obj val')
	plt.title('Objective value min')
	plt.tight_layout()


	dfObjVals['day ' + str(day)] = pd.ewma(dfObjVals['day ' + str(day)], span = 2400)
	if day == 0:
		dfObjValsMaster = dfObjVals
	else:
		dfObjValsMaster = dfObjValsMaster.merge(dfObjVals, on = 'hour', how = 'outer')


	dfObjValsMaster.to_csv('dfObjValsMaster' + str(day) + '.csv')

	dfObjValsMaster.plot(x = 'hour')
	plt.xlabel('hour')
	plt.title('Objective Function Values')

	if dynamic:
		dfActions.columns = ['currHod', 'currSod', 'lastStateID', 'lastObjValue', 'lastAction', 'currStateID', 'currObjValue', 'action']
		dfActions.to_csv('dfActions' + str(day) + '.csv')

	dfQueueTracker.columns = ['hour', 'south', 'north', 'west', 'east']
	dfQueueTracker['hour'] = dfQueueTracker['hour']/(1.0*secondsInHour)
	dfQueueTracker.to_csv('dfQueueTracker' + str(day) + '.csv')

	# for i in ['south', 'north', 'west', 'east']:
	# 	dfQueueTracker[i] = pd.ewma(dfQueueTracker[i], span = 2400)
	# dfQueueTracker.plot(x = 'hour')
	# plt.xlabel('hour')
	# plt.title('Queue Tracker on day '+ str(day))

	dfNumVehiclesTracker.columns = ['hour', 'south', 'north', 'west', 'east']
	dfNumVehiclesTracker['hour'] = dfNumVehiclesTracker['hour']/(1.0*secondsInHour)
	dfNumVehiclesTracker.to_csv('dfQueueTracker' + str(day) + '.csv')

	# for i in ['south', 'north', 'west', 'east']:
	# 	dfNumVehiclesTracker[i] = pd.ewma(dfNumVehiclesTracker[i], span = 2400)
	# dfNumVehiclesTracker.plot(x = 'hour')
	# plt.xlabel('hour')
	# plt.title('Number Vehicles on Edge Tracker on day '+ str(day))

	dfWaitingTracker.columns = ['hour','south', 'north', 'west', 'east']
	dfWaitingTracker['hour'] = dfWaitingTracker['hour']/(1.0*secondsInHour)
	dfWaitingTracker.to_csv('dfWaitingTracker' + str(day) + '.csv')

	# for i in ['south', 'north', 'west', 'east']:
	# 	dfWaitingTracker[i] = pd.ewma(dfWaitingTracker[i], span = 2400)
	# dfWaitingTracker.plot(x = 'hour')
	# plt.xlabel('hour')
	# plt.title('Cumulative Minutes Waited For Stopped Vehicles on day '+ str(day))


# 	print 'QValues = ', QValues
	np.savetxt('QValues' + str(day) +'.txt', QValues)
# 	print 'QProbs = ', QProbs
	np.savetxt('QProbs' + str(day) +'.txt', QProbs)
# 	print 'QAlphas = ', QAlphas
	np.savetxt('QAlphas' + str(day) + '.txt', QAlphas)
# 	print 'QCounts = ', QCounts
	np.savetxt('QCounts' + str(day) + '.txt', QCounts)

	plt.show()

# TODO - show the learning curve; number of days of training

# def assignStateNum(phaseNum, queueTracker, waitingTracker, queueBuckets, waitingBuckets):
# 	# assign each edge queue size to a corresponding bucket number
# 	for i in queueTracker.keys():
# 		assignedBucket = False
# 		for j in range(len(queueBuckets)):
# 			if not assignedBucket and queueTracker[i] <= queueBuckets[j]:
# 				queueTracker[i] = j
# 				assignedBucket = True
# 		if not assignedBucket:
# 			queueTracker[i] = len(queueBuckets)

# 	for i in waitingTracker.keys():
# 		assignedBucket = False
# 		for j in range(len(waitingBuckets)):
# 			if not assignedBucket and waitingTracker[i] < waitingBuckets[j]:
# 				waitingTracker[i] = j
# 				assignedBucket = True
# 		if not assignedBucket:
# 			waitingTracker[i] = len(waitingBuckets)

# 	# assign each edge waiting time to a correpsonding bucket number
# 	p = dfStateMapping['phase'] == phaseNum/2 # only 4 states where we are taking action 
# 	e1q = dfStateMapping['8949170_q'] == queueTracker['8949170']
# 	e1w = dfStateMapping['8949170_w'] == waitingTracker['8949170']
# 	e2q = dfStateMapping['-164126513_q'] == queueTracker['-164126513']
# 	e2w = dfStateMapping['-164126513_w'] == waitingTracker['-164126513']
# 	e3q = dfStateMapping['52016249_q'] == queueTracker['52016249']
# 	e3w = dfStateMapping['52016249_w'] == waitingTracker['52016249']
# 	e4q = dfStateMapping['-164126511_q'] == queueTracker['-164126511']	
# 	e4w = dfStateMapping['-164126511_w'] == waitingTracker['-164126511']	
# 	a = dfStateMapping['stateNum'][p & e1q & e1w & e2q & e2w & e3q & e3w & e4q & e4w]
# 	# print 'a = ', a
# 	# return a
# 	# print 'a.dtype = ', a.dtype
# 	return int(a.tolist()[0])
# 	# lookup number in table; return number corresponding to state (not numerically significant)

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

# dfStateMapping = pd.DataFrame.from_csv('dfStateMapping.csv')

# cols = ('8949170_q', '8949170_w', '-164126513_q', '-164126513_w',\
#  		 '52016249_q', '52016249_w', '-164126511_q','-164126511_w')
# dfQueueSizesWaitingTimes = pd.DataFrame(columns=cols)