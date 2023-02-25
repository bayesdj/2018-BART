import numpy as np
from numpy import random, exp, log, ones, zeros, sqrt, array, arange
from treelib import Node, Tree

class Cart():
	def __init__(self,df):
		tree = Tree()
		root = tree.create_node(0,0,data=df)
		self.tree = tree

# for t in range(T):
	# tree1 = sampleTree(tree)
	