#!/usr/bin/env python
# linear_model.py
#Author : saimadhu
#Date: 25-sept-2014
#About: Linear Regression for brain-body weight predictions

# Required Packages
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

#converting txt files to csv files
def convert_txt_csv(txtfile_name,csvfile_name):
	with open(txtfile_name, "rb") as infile, open(csvfile_name, 'wb') as outfile:
		in_txt = csv.reader(infile, delimiter = '\t')
		out_csv = csv.writer(outfile)
		out_csv.writerows(in_txt)
	return out_csv

# Getting X_values and Y_values	
def convert_test_train_data(csv_file):	
	X,Y=[],[]
	with open(csv_file) as fil:
	    for line in fil.readlines():
	        temp=line.split()
	        if temp:
	        	# converting string values to float
	            X.append([float(temp[1])])
	            Y.append(float(temp[2]))
	return X,Y

# linear model main calling function
def linearmodel_main():
	train_txt_file = r"train_data.txt"
	test_txt_file = r"test_data.txt"
	train_csv_name = r"train_csv_file.csv"
	test_csv_name = r"test_csv_file.csv"
	train_csv_file = convert_txt_csv(train_txt_file,train_csv_name)
	test_csv_file = convert_txt_csv(test_txt_file,test_csv_name)
	train_X,train_Y,test_X,test_Y = [], [] , [], []
	train_X , train_Y = convert_test_train_data(train_csv_name)
	test_X,test_Y = convert_test_train_data(test_csv_name)
	#print train_X
	#print train_Y
	#print test_X
	#print test_Y

	# Create linear regression object
	regr = linear_model.LinearRegression()
	regr.fit(train_X, train_Y)
	print('Coefficients: \n', regr.coef_)
	#plt.scatter(test_X, test_Y,  color='red')
	#plt.plot(test_X, regr.predict(test_X), color='blue',linewidth=3)
	plt.scatter(train_X,train_Y,color='blue')
	plt.plot(train_X,regr.predict(train_X),color='red',linewidth=4)
	plt.xticks(())
	plt.yticks(())
	plt.show()
	print "successfully Running Completed............."

linearmodel_main()		#calling main function
