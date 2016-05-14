"""
FILENAME: plotArrivalFittingFunction.py
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

#==============PARAMETERS

hoursInDay = 24

xls = pd.ExcelFile('simInput.xlsx')
print xls.sheet_names
dfArrivalData = xls.parse('randArrTurnRates')
print dfArrivalData
dfArrivalData.columns = ['hour', 'Palm_NBound_TurnL', 'Palm_NBound_Staight', 'Palm_NBound_TurnR', \
							'Palm_SBound_TurnL', 'Palm_SBound_Staight', 'Palm_SBound_TurnR',\
							'Arboretum_WBound_TurnL', 'Arboretum_WBound_Staight', 'Arboretum_WBound_TurnR',\
							'Arboretum_EBound_TurnL', 'Arboretum_EBound_Staight', 'Arboretum_EBound_TurnR']
print dfArrivalData.columns
colNames = list(dfArrivalData.columns.values)
x = dfArrivalData['hour']
x = x.reshape(hoursInDay, 1)
# print "x = ", x

for j in range(0,len(colNames)-1):
	print "j = ", j
	newCol = str(colNames[j+1])
	print 'newCol = ', newCol
	y = dfArrivalData[newCol]
	print "y =", y
	y = y.reshape(hoursInDay, 1)

	poly = PolynomialFeatures(degree=12)
	hoursPoly = poly.fit_transform(x)
	# print hoursPoly

	regr = linear_model.LinearRegression()
	regr.fit(hoursPoly, y)

	# plot it as in the example at http://scikit-learn.org/
	plt.figure(j/6+1)
	plotNum = 230 + j%6 + 1
	plt.subplot(plotNum)
	plt.scatter(x, y,  color='black')
	plt.plot(x, regr.predict(hoursPoly), color='blue', linewidth=3)
	plt.xlabel('Hour of Day')
	plt.ylabel('Cars Per Minute')
	plt.title(newCol)
	plt.xticks()
	plt.yticks()
	plt.ylim(0,30)
	plt.xlim(0,24)
	if (j+1)%6 == 0:
		plt.tight_layout()
		plt.show()

plt.tight_layout()
plt.show()