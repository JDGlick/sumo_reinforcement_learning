"""
FILENAME: arrivalRateGen.py
INPUT: arrivalRates.csv (synthetically generated data of arrival rates for each of the four roads)
OUTPUT: arrivalRates.
"""
#==============IMPORT LIBRARIES

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn import datasets, linear_model
import pandas as pd
from pandas import *
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import random
import xml.etree.ElementTree as ET
from datetime import datetime

#==============PARAMETERS

secondsInDay = 24*60*60
secondsInHour = 60.0*60.0
sPhoneTrackingRate = 0.35
arrivalScaling = 0.475

xls = pd.ExcelFile('simInput.xlsx')
# print xls.sheet_names
dfArrivalData = xls.parse('randArrTurnRates')


# print dfArrivalData
# dfArrivalData.columns = ['hour', 'Palm_NBound_TurnL', 'Palm_NBound_Staight', 'Palm_NBound_TurnR', \
# 							'Palm_SBound_TurnL', 'Palm_SBound_Staight', 'Palm_SBound_TurnR',\
# 							'Arboretum_WBound_TurnL', 'Arboretum_WBound_Staight', 'Arboretum_WBound_TurnR',\
# 							'Arboretum_EBound_TurnL', 'Arboretum_EBound_Staight', 'Arboretum_EBound_TurnR']
edges = ['8949170 -52016249', '8949170 164126513', '8949170 164126511', \
							'-164126513 164126511', '-164126513 -8949170', '-164126513 -52016249',\
							'52016249 164126513', '52016249 164126511', '52016249 -8949170',\
							'-164126511 -8949170', '-164126511 -52016249', '-164126511 164126513']
dfArrivalData.columns = ['hour', '0', '1', '2', \
							'3', '4', '5',\
							'6', '7', '8',\
							'9', '10', '11']
# print dfArrivalData.columns
colNames = list(dfArrivalData.columns.values)
x = dfArrivalData['hour']
functions = []
# print 'arrivals = ', arrivals
for j in range(0,len(colNames)-1):
	# print "j = ", j
	newCol = str(colNames[j+1])
	# print 'newCol = ', newCol
	y = dfArrivalData[newCol]
	# print 'y = ', y
	p = np.polyfit(x, y, 12)
	# print p
	functions.append(np.poly1d(p))

	# print 'z = ', z
	# Once we have the generating function
def writeRoutes(seed):
	phoneON = 0
	arrivals = []
	random.seed(seed)
	for z in range(len(functions)):
		sod = 0.0
		while sod < secondsInDay:
			# print 'newCol = ', newCol
			# print 'sod = ', sod
			hod = sod/(60*60.0)
			# print 'hod = ', hod
			arrivalRate = max(1,functions[z](hod)*arrivalScaling)
			# print 'arrivalRate = ', arrivalRate
			iaTimeSeconds = random.expovariate(arrivalRate)*60.0
			# print 'iaTimeSeconds = ', iaTimeSeconds
			unif = random.random()
			if unif > (1 - sPhoneTrackingRate):
				phoneON = 1
			else:
				phoneON = 0
			if sod + iaTimeSeconds > secondsInDay:
				break	
			arrivals.append([sod, colNames[z+1], phoneON])
			sod += iaTimeSeconds

	arrivals.sort()

	# print 'arrivals[1:30] =', arrivals[1:30]

	# Open the file
	with open('palm.rand.rou.xml', 'w') as routes:
		routes.write("""<?xml version="1.0"?>""" + '\n' + '\n')
		routes.write("""<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">""" + '\n')
		routes.write('\n')
		routes.write("""<vType id="type1" color="255,105,180"/>""" + '\n')
		routes.write("""<vType id="type0" color="0,255,255"/>""" + '\n')
		routes.write('\n')	
		for i in range(12):
			routes.write("""<route id=\"""" + str(i) + """\"""" + """ edges=\"""" + edges[i] + """\"/> """ + '\n')
			#    <route id="route0" color="1,1,0" edges="beg middle end rend"/>

		routes.write('\n')
		idCounter = 0
		for i in arrivals:
			if (i[2]==1):
				color = """ color=\"255,105,180\""""
				vType = """\" type=\"type1"""
			else:
				color = """ color=\"0,255,255\""""
				vType = """\" type=\"type0"""
			routes.write("""<vehicle id=\"""" + str(idCounter) + """\" depart=\"""" + str(round(i[0],2)) + """\" route=\"""" + str(i[1]) + vType + """\"/>""" + '\n')
			# routes.write("""     <route edges=\"""" + str(i[1]) + """\"""" + """/>""" + '\n')
			# routes.write("""</vehicle>""" + '\n')
			idCounter += 1
		routes.write("""</routes>""")	
 #    <vehicle id="0" depart="0.00">


# <routes>
#    <vType id="type1" accel="0.8" decel="4.5" sigma="0.5" length="5" maxSpeed="70"/>

#    <route id="route0" color="1,1,0" edges="beg middle end rend"/>

#    <vehicle id="0" type="type1" route="route0" depart="0" color="1,0,0"/>
#    <vehicle id="1" type="type1" route="route0" depart="0" color="0,1,0"/>

# </routes>