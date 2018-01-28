from __future__ import division
# data points [coffee, cream]:
data = [[ 0,0 ], [ 0,1 ], [ 1,0 ], [ 1,1 ] ]

#Just last one is a positive experience
category = [ -1,  -1,  -1, 1 ]

import numpy
from sklearn.svm import SVC

clf = SVC(kernel='linear', C=3)
clf.fit(data, category)

#Get m coefficients:
coef = clf.coef_[0]
b = clf.intercept_[0]

print('This is the M*X+b=0 equation...')
print('M=%s' % (coef))
print('b=%s' % (b))
print('So the equation of the separating line in this 2d svm is:')
print('%f*x + %f*y + %f = 0' % (coef[0],coef[1],b))
print('The support vector limit lines are:')
print('%f*x + %f*y + %f = -1' % (coef[0],coef[1],b))
print('%f*x + %f*y + %f = 1' % (coef[0],coef[1],b))

vertmatrix = [[x] for x in coef]

good = 0
bad = 0
for i, d in enumerate(data):
	#i-th element, d in data:
	calculatedValue = numpy.dot(d, vertmatrix)[0] + b
	print( 'Mx+b for x=%s calculates to %s' % (d, calculatedValue) )
	if calculatedValue > 0 and category[i] > 0:
		good += 1
	elif calculatedValue < 0 and category[i] < 0:
		good += 1
	else:
		bad +=1 #they should have matched category.
		
print('accuracy=%f' % (good/(good+bad)) )
#The same as the builtin "score" accuracy:
print('accuracy=%f' % clf.score(data, category) )

while True:
	#Try other points
	d = input('vector=')
	calculatedValue = numpy.dot(d, vertmatrix)[0] + b
	print(calculatedValue)
