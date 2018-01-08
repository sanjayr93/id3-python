"""
Author - Sanjay Ramachandran

AIMA Decision Tree Learning Algorithm with Chi-squared pruning

- Does boolean classification as of now
- Target/Goal attribute should be the last column
"""

#!/usr/bin/python3
import os
import copy
import csv
from math import log2, fsum, inf

try:
    from scipy.stats import chisquare  # for chi-square calculation
except ImportError:  # try to install requests module if not present
    print ("Trying to Install required module: scipy\n")
    os.system('python -m pip install --user scipy')
from scipy.stats import chisquare

class Utilities:
	'Utility functions'

	#chi-square table values at 5%level for degrees 1 to 8
	#if value of delta is less than this, then attribute is irrelevant
	chiSquare = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.07, 6: 12.59, 7: 14.07, 8: 15.51}

	#Entropy of a boolean random variable that is true with probability q
	B = lambda q : q if q == 0.0 else -1 * (q * log2(q) + (((1 - q) * log2(1 - q)) if q < 1 else 0))

	#Remainder(A) will be the sum of this computation for each subset.
	#counts - counts tuple (p, n, pk, nk)
	ExpectedEntropy = lambda counts : ((counts[2] + counts[3])/(counts[0] + counts[1])) * Utilities.B((counts[2]/(counts[2] + counts[3])) if (counts[2] + counts[3] > 0) else 0)

	#floating point sum of the elements in list
	Sum = lambda list : fsum(list)

	#Remainder(A) calculation
	#subsets has the list of count arrays of each subset of example on splitting with Attribute A
	Remainder = lambda subsets : Utilities.Sum(map(Utilities.ExpectedEntropy, subsets))

	#Gain calculation
	Gain = lambda p, n, subsets : Utilities.B(p/(p + n)) - Utilities.Remainder(subsets)

	#p cap and n cap calculation. counts tuple - (p, n, pk, nk)
	PCap = lambda counts : counts[0] * ((counts[2] + counts[3])/(counts[0] + counts[1]))
	NCap = lambda counts : counts[1] * ((counts[2] + counts[3])/(counts[0] + counts[1]))

	def deltaValue(node):
		#degree is number of branches - 1
		#positive and negetive examples
		pk = []
		nk = []
		#expected number of positive and negative examples
		pkcap = []
		nkcap = []

		#calculating the observed and expected frequencies for all subsets
		for key in node.branches:
			branch = node.branches[key]
			pk.append(branch.p)
			nk.append(branch.n)
			pkcap.append(Utilities.PCap((node.p, node.n, branch.p, branch.n)))
			nkcap.append(Utilities.NCap((node.p, node.n, branch.p, branch.n)))

		#calculating the delta value
		return chisquare(pk, f_exp = pkcap)[0] + chisquare(nk, f_exp = nkcap)[0]


class DTNode:
	'Node of the Decision Tree'

	def __init__(self, parent, pk, nk, nodeType, attributeName=None, attributeIndex=-1, classification=None):
		self.parent = parent
		self.p = pk
		self.n = nk
		self.type = nodeType
		if(self.type == 'testNode'):
			self.attributeName = attributeName
			self.attributeIndex = attributeIndex
		else:
			self.classification = classification
		self.branches = {} #key of branches will be the different values for the attribute


class DecisionTree:
	'Implementation of DTL'

	def __init__(self, dataset):
		i = 0
		self.examples = []
		for row in dataset:
			if(i == 0):
				i = 1
				self.attributes = copy.deepcopy(row)
			else:
				self.examples.append(copy.deepcopy(row))
		
		self.attributeValues = self.getAttributeValues(self.attributes, self.examples)
		targetVals = self.attributeValues[self.attributes[len(self.attributes) - 1]]
		self.pValue, self.nValue = targetVals[0], targetVals[1]
		self.p, self.n = self.getClassCount(self.examples)
		self.takenAttributes = []


	def DTL(self, examples, attributes, parent, parentExamples):
		'''
		Implementation of the DTL algorithm
		'''
		if(len(examples) == 0):
			#return a leaf node with the majority class value in parent examples
			return self.pluralityValue(parent, parentExamples)
		elif(self.hasSameClass(examples)):
			#return a leaf node with the class value
			p, n = self.getClassCount(examples)
			return DTNode(parent, p, n, 'leafNode', classification = self.pValue if p > 0 else self.nValue)
		elif((len(attributes) - len(self.takenAttributes)) == 0):
			#return a leaf node with the majority class value in examples
			return self.pluralityValue(parent, examples)
		else:
			#find the attribute that has max information gain
			attrIndex = self.importantAttrIndex(attributes, examples)
			attribute = attributes[attrIndex]
			p, n = self.getClassCount(examples)

			#create a root node
			root = DTNode(parent, p, n, 'testNode', attributeName = attribute, attributeIndex = attrIndex)
			#to track the attributes in inner nodes
			self.takenAttributes.append(attribute)

			#divide the examples and recursively call DTL to create child nodes
			for value in self.attributeValues[attribute]:
				newExample = []
				for row in examples:
					if(row[attrIndex] == value):
						newExample.append(copy.deepcopy(row))
				childNode = self.DTL(newExample, attributes, root, examples)

				#add the sub tree to the main tree
				root.branches[value] = childNode

		return root



	def importantAttrIndex(self, attributes, examples):
		'''
		Calculate the Importance value or the information gain for all attributes
		Return the attribute with max gain
		'''
		maxVal = -inf
		maxValInd = -1
		
		for index, a in enumerate(attributes[:len(attributes) - 1]):
			if(a not in self.takenAttributes):
				gain = self.importance(a, index, examples)
				if(gain > maxVal):
					maxVal = gain
					maxValInd = index

		return maxValInd


	def importance(self, attribute, index, examples):
		'''
		Calculate the gain for a given attribute
		'''
		subsets = []

		for value in self.attributeValues[attribute]:
			pk = nk = 0
			for row in examples:
				if(row[index] == value):
					if(row[len(row) - 1] == self.pValue):
						pk += 1
					else:
						nk += 1

			subsets.append((self.p, self.n, pk, nk))

		return Utilities.Gain(self.p, self.n, subsets)


	def getAttributeValues(self, attributes, examples):
		'''
		To find the domain values for each attribute
		'''
		values = {}

		for index, a in enumerate(attributes):
			temp = []
			for row in examples:
				if(row[index] not in temp):
					temp.append(row[index])
			values[a] = temp

		return values


	def hasSameClass(self, examples):
		'''
		Checks if the examples have the same target variable value
		'''
		prevValue = examples[0][len(examples[0]) - 1]
		
		for row in examples[1:]:
			if(row[len(row) - 1] != prevValue):
				return False

		return True

	def pluralityValue(self, parent, examples):
		'''
		Returns a leaf node with majority class value
		'''
		p, n = self.getClassCount(examples)
		return DTNode(parent, p, n, 'leafNode', classification = self.pValue if p > n else self.nValue)

	def getClassCount(self, examples):
		'''
		Returns the number of examples in positive and negative classes
		'''
		p = n = 0

		for row in examples:
			if(row[len(row) - 1] == self.pValue):
				p += 1
			elif(row[len(row) - 1] == self.nValue):
				n += 1

		return p, n

	def printDTree(self, node, value=None):
		'''
		Recursively prints the DTree in a flattened structure
		'''
		print('If parent - ', node.parent.attributeName if node.parent else 'This is Root Node', ' = ', value if value else 'This is Root Node',
																											 ' | test node - ', node.attributeName)

		for branch in node.branches:
			if(node.branches[branch].type == 'leafNode'):
				print('If parent - ', node.branches[branch].parent.attributeName, ' = ', branch if branch else 'This is Root Node', ' | leaf node - ', 
																													node.branches[branch].classification)

		for branch in node.branches:
			if(node.branches[branch].type == 'testNode'):
				self.printDTree(node.branches[branch], branch)

	def traverseTree(self, test, node):
		'''
		Traverses the tree to classify the test data
		'''
		attributeValue = test[node.attributeName]
		if(node.branches[attributeValue].type == 'leafNode'):
			return node.branches[attributeValue].classification
		else:
			return self.traverseTree(test, node.branches[attributeValue])


	def predict(self, testSet):
		'''
		Returns the prediction for all the test data
		'''
		predictions = []
		for index, row in enumerate(testSet):
			if (index == 0):
				continue
			test = {}

			for index, data in enumerate(row):
				test[self.attributes[index]] = data

			predictions.append(self.traverseTree(test, self.root))

		return predictions

	def chiSquarePrune(self, node):
		'''
		- Finds the test node with only leaf descendants
		- calculates the delta value
		- if its less than the 5% value, then creates a leaf node with the majority class
		- replaces the test node with this leaf node
		- recursively does this till no more nodes can be pruned
		'''
		flag = 0
		for key in node.branches:
			if(node.branches[key].type == 'leafNode'):
				flag += 1
				continue
			else:
				pruned = self.chiSquarePrune(node.branches[key])
				if(pruned):
					flag += 1

		#calculating the delta value and deciding to prune or not
		if(flag == len(node.branches)):
			if(Utilities.deltaValue(node) < Utilities.chiSquare[len(node.branches) - 1]):
				if(node.parent):
					newLeafNode = DTNode(node.parent, node.p, node.n, 'leafNode', classification = self.pValue if node.p > node.n else self.nValue)
					tAttrValue = None
					for key in node.parent.branches:
						if(node.parent.branches[key].type == 'testNode' and node.parent.branches[key].attributeName == node.attributeName):
							tAttrValue = key
							break

					parent = node.parent
					parent.branches[tAttrValue] = newLeafNode
					del node
				return True
			else:
				return False



if (__name__ == '__main__'):

	decisionTree = None
	with open('trainingdata.csv') as csvFile:
		dataset = csv.reader(csvFile, delimiter=',')
		decisionTree = DecisionTree(dataset)

		print('### Building the Decision Tree from the training data in trainingdata.csv...')
		decisionTree.root = decisionTree.DTL(decisionTree.examples, decisionTree.attributes, None, decisionTree.examples)
		print('### Decision Tree model built as below...')
		decisionTree.printDTree(decisionTree.root)

		flag = input('### Do you want to recursively prune ? (y/n) - Caution: At 5% level, Sometimes pruning makes the decision tree very shallow.')
		if(flag == 'y'):
			decisionTree.chiSquarePrune(decisionTree.root)
			print('### After pruning at 5% level: ')
			decisionTree.printDTree(decisionTree.root)

	print('### Testing the decision tree with test set from testdata.csv file')
	with open('testdata.csv') as csvTest:
		testSet = csv.reader(csvTest, delimiter=',')
		predictions = decisionTree.predict(testSet)
		for index, prediction in enumerate(predictions):
			print('For example -', index + 1, ', the prediction is ', decisionTree.attributes[len(decisionTree.attributes) - 1], '=', prediction)