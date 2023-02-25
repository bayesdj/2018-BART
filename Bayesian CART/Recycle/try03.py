import numpy as np
from numpy import random, exp, log, ones, zeros, sqrt, array, arange
from treelib import Node, Tree
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats
from itertools import chain, combinations
#%%
def f1(x):
    x1,x2 = x
    if np.isin(x2,['a','b']):
        return 8 if x1 <= 5 else 2
    else:
        if x1 <= 3: return 1
        elif x1 <= 7: return 5
        else: return 8        

def genData(n=800,seed=None):
    if seed is not None:
        random.seed(seed)
    x1val = np.arange(1,11)
    x2val = ['a','b','c','d']    
    x1 = random.choice(x1val,size=n)
    x2 = random.choice(x2val,size=n)
    y = array(list(map(f1,tuple(zip(x1,x2))))) + 2*random.normal(size=n)
    return pd.DataFrame({'y':y,'x1':x1,'x2':x2})

def powerset(s):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    #s = set(iterable)
    g = chain.from_iterable(combinations(s, r) for r in range(1,len(s)))
    return list(g)

def getLegalXVal(df,xvar):
#    dType = dataType[xvar]
    xVal = [set(df.iloc[:,i]) for i in xvar]
    n = array([len(v) for v in xVal])
    legal = n>=2
    if sum(legal)==len(xvar): 
        return xvar, xVal
    else:
        xVal = [i for (i, v) in zip(xVal, legal) if v]
        return xvar[legal], xVal
    
def get_st(df):
    y = df['y']
    n = len(y)
    s = (n-1)*y.var()
    t = n*a/(n+a)*(y.mean()-mu)**2
    return s+t 

def get_pSplit(alpha,beta,depth):
    p = alpha/(1+depth)**beta
    return p/sum(p)

#%%
xiList = []
sList = []
terminalAry = array([0])
depthAry = array([0])

df = genData(seed=98)
df['x3'] = 22
s = df['y'].var()
mu = df['y'].mean()
n,p = df.shape

v = 12; lamda = 4
vlamda = v*lamda
ig1 = v/2; ig2 = lamda*ig1
a = 1/3
lna = log(a)
npv = n+v

alpha = 0.95
beta = 1
#%%
tree = Tree()
root = tree.create_node(0,0,data=df)
DataTypes = df.dtypes.map(lambda x:x.kind)

ST = [get_st(df)]
STsum = sum(ST)+vlamda
#actionSelect
pGrow = 1
action = 'grow'
leaf = tree.leaves()
depth = array([len(i) for i in tree.paths_to_leaves()])-1
pSplit = get_pSplit(alpha,beta,depth)
#node select
node_select = random.choice(range(len(leaf)),p=pSplit) 
xvar,xVal = getLegalXVal(df,arange(1,p))
# variable select
xi = 1#random.choice(xvar) 
x = xvar[xi]
dtype = DataTypes[x]
# split value select
if dtype == 'O':
    S = powerset(xVal[xi])
    s = random.choice(S)
    idx = df.iloc[:,x].isin(s)
elif (dtype == 'i') | (dtype == 'f'):
    S = list(xVal[xi])[:-1] #leave the last element out
    s = random.choice(S)
    idx = (df.iloc[:,x] <= s)
nXS = len(S)*len(xvar)
parent = tree.root
Child = [df.loc[idx],df.loc[~idx]] 
tree1 = Tree(tree=tree,deep=True)
for i in [0,1]:
    tree1.create_node(i+1,i+1,parent=parent,data=Child[i])
tree1[parent].data=None
pPrune = 0.5

stChild = [get_st(df) for df in Child]
p1 = a*(a+len(df))/(a+array([len(df) for df in Child])).prod()
STsum1 = STsum-ST[0]+sum(stChild)
p2 = (STsum/STsum1)**npv
rLike = sqrt(p1*p2)
leaf1 = tree1.leaves()
tree1InternalNodes = set([node.bpointer for node in leaf1])
w2 = len(tree1InternalNodes)
rTransit = pPrune/pGrow/pSplit[node_select]*nXS/w2
d = depth[node_select]+1
rStruct = alpha*(1-alpha/(1+d)**beta)**2/(d**beta-alpha)/nXS
r = rLike*rTransit*rStruct



