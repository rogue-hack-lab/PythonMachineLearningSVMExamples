#!/usr/bin/env python
# Example of Skikitlearn SVM with Congress data from https://github.com/unitedstates/congress
# Setup (if numpy/pandas not installed):
# sudo pip3 install numpy
# sudo pip3 install pandas
# -*- coding: utf-8 -*-
from __future__ import division #or it does odd stuff python3 and any other language doesn't do.
import os
import pandas

def main():
	datapath = 'data/115/votes/2018'
	persons = {} #To look up by name...
	for item in os.listdir( datapath ):
		if item[0] == 'h':
			path = os.path.join(datapath, item, 'data.json')
			print(path)
			data = pandas.read_json(path)
			if 'Nay' in data.votes:
				print('-%d' % (len(data.votes.Nay)) )
				for voter in data.votes.Nay:
					if not voter['display_name'] in persons:
						voter['record'] = []
						persons[ voter['display_name'] ] = voter

			if 'Yea' in data.votes:
				print('+%d' % (len(data.votes.Yea)) )
				for voter in data.votes.Yea:
					if not voter['display_name'] in persons:
						voter['record'] = []
						persons[ voter['display_name'] ] = voter
						

			print( 'Loaded %d names.' % (len(persons)))
			#Get all yea/Nay votes' voters.. and that's a row.
			
			#Each person [h1,h2,h3]...[d/r] category.
	#Now read each into record of the person...
	print('gathering record for each...')
	for item in os.listdir( datapath ):
		if item[0] == 'h':
			path = os.path.join(datapath, item, 'data.json')
			print(path)
			data = pandas.read_json(path)
			for person in persons:
				thisvote = 0
				if 'Nay' in data.votes:
					for voter in data.votes.Nay:
						if voter['display_name'] == person:
							thisvote = -1
				if 'No' in data.votes:
					for voter in data.votes.No:
						if voter['display_name'] == person:
							thisvote = -1

				if 'Yea' in data.votes:
					for voter in data.votes.Yea:
						if voter['display_name'] == person:
							thisvote = 1
				if 'Aye' in data.votes:
					for voter in data.votes.Aye:
						if voter['display_name'] == person:
							thisvote = 1
				#print( person + str(thisvote))
				persons[person]['record'].append( thisvote )

			#Each person [h1,h2,h3]...[d/r].
			
	#Get in the format SVM learns:
	data = []
	category = []
	for person in persons:
		print( persons[person]['display_name'], persons[person]['record'], persons[person]['party'] )
		data.append(persons[person]['record'])
		if persons[person]['party'] == 'R':
			category.append( 1 )
		else:
			category.append( -1 )
	#Exactly like the example... more dimensions			
	import numpy
	from sklearn.svm import SVC

	clf = SVC(kernel='linear')
	clf.fit(data, category)

	#Get m coefficients:
	coef = clf.coef_[0]
	b = clf.intercept_[0]

	print('This is the M*X+b=0 equation...')
	print('M=%s' % (coef))
	print('b=%s' % (b))
	
	vertmatrix = [[x] for x in coef]
	good = 0
	bad = 0
	for i, d in enumerate(data):
		#i-th element, d in data:
		calculatedValue = numpy.dot(d, vertmatrix)[0] + b
		if calculatedValue > 0 and category[i] > 0:
			good += 1
		elif calculatedValue < 0 and category[i] < 0:
			good += 1
		else:
			bad +=1 #they should have matched category.
			print('Wrong calculated value:')
			print( 'Mx+b for x=%s calculates to %s realval %s' % (d, calculatedValue, category[i]) )
			print( '\n')
		if calculatedValue < 1 and calculatedValue > -1:
			#The middle zone.
			print( 'Mx+b for x=%s calculates to %s realval %s' % (d, calculatedValue, category[i]) )
		
			
	print('accuracy=%f' % (good/(good+bad)) )
	#The same as the builtin "score" accuracy:
	print('accuracy=%f' % clf.score(data, category) )
	return 0

if __name__ == '__main__':
	main()

