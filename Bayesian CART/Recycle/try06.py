import numpy as np
from numpy import random, exp, log, ones, zeros, sqrt, array, arange
from treelib import Node, Tree
import pandas as pd
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
    S = chain.from_iterable(combinations(s, r) for r in range(1,len(s)))
    return list(S)

def get_xvar(df):
    n = df.iloc[:,1:].apply(lambda s:len(set(s)))
    return arange(1,df.shape[1])[n>1]
    
def get_st(df):
    y = df['y']
    n = len(y)
    s = (n-1)*y.var()
    t = n*a/(n+a)*(y.mean()-mu)**2
    return s+t 

def get_pSplit(alpha,beta,depth):
    p = alpha/(1+depth)**beta
    return p/p.sum()

def grow(tree):    
    global j
    #random.seed(98)
    leaf = list(filter(lambda node:len(node.xvar)>0,tree.leaves()))
    b = len(leaf)
    if b<1: raise ValueError('cannot grow anymore')
    pNode = random.choice(leaf) 
    pid = pNode.identifier
    x = random.choice(pNode.xvar)   
    dtype = DataTypes[x]
    df = pNode.data
    xVal = set(df.iloc[:,x])
    if dtype == 'O':
        S = powerset(xVal)
        s = S[random.choice(range(len(S)))]
        idx = df.iloc[:,x].isin(s)
    elif (dtype == 'i') | (dtype == 'f'):
        S = sorted(xVal)[:-1] #leave the last element out
        s = random.choice(S)
        idx = (df.iloc[:,x] <= s)
    nXS = len(pNode.xvar)*len(S)
    childDF = [df.loc[idx],df.loc[~idx]] 
    tree1 = Tree(tree=tree,deep=True)    
    node1 = tree1.create_node(str(j),j,parent=pid,data=childDF[0])
    node2 = tree1.create_node(str(j+1),j+1,parent=pid,data=childDF[1])
    st1 = [get_st(df) for df in childDF]
    tree1.stSum = tree.stSum-pNode.st+sum(st1)
    
    p1 = a*(a+len(df))/(a+array([len(df) for df in childDF])).prod()
    p2 = (tree.stSum/tree1.stSum)**npv    
    rLike = sqrt(p1)*p2
    w2 = len(set([node.bpointer for node in tree1.leaves()]))
    rTransit = pPrune/pGrow*b*nXS/w2
    d = tree.level(pid)+1
    rStruct = alpha*(1-alpha/(1+d)**beta)**2/(d**beta-alpha)/nXS
    r = rLike*rTransit*rStruct
    if random.uniform() < r:
        j+=2
        pNode = tree1.get_node(pid)
        pNode.tag = str(pid)+'-'+str(x)+'-'+str(s)
#        pNode.data = None
        nodes = [node1,node2]
        for i in [0,1]:
            df = childDF[i]
            nodes[i].xvar = get_xvar(df)
            nodes[i].st = st1[i]
            nodes[i].data = df
        return tree1
    else:
        return tree

#%%
df = genData()
df['x3'] = 22
s = df['y'].var()
mu = df['y'].mean()
n,p = df.shape

v = 12; lamda = 4
vlamda = v*lamda
ig1 = v/2; ig2 = lamda*ig1
a = 1/3
lna = log(a)
npv = (n+v)/2

alpha = 0.95
beta = 1
DataTypes = df.dtypes.map(lambda x:x.kind)
#%%
tree = Tree()
root = tree.create_node('0',0,data=df)
root.st = get_st(root.data)
root.xvar = get_xvar(df)
tree.stSum = root.st+vlamda
j = 1
#%%
#actionSelect
pGrow = 1
pPrune = 0.5
action = 'grow'
leaf = tree.leaves()
#depth = array([len(i) for i in tree.paths_to_leaves()])-1

for t in range(100):
    tree = grow(tree)
    tree.show()
    


