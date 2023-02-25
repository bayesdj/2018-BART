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

def prune(tree):
    leaf0 = tree.leaves()
    # only pairs of terminal nodes with no children
    leaf = filter(lambda n:len(tree.siblings(n.identifier)[0].fpointer)==0,leaf0)
    leaf2 = list(set([node.bpointer for node in leaf])) 
    w2 = len(leaf2)
    pid = random.choice(leaf2)
    pNode = tree[pid]
    leaf = [n.identifier for n in leaf0 if (n.identifier not in pNode.fpointer)
                    and len(n.xvar)>0 ] # leaf for grow
    b = len(leaf)+1
    st1 = [tree[c].st for c in pNode.fpointer]
    nChild = array([tree[c].data.shape[0] for c in pNode.fpointer])
    nXS = len(pNode.xvar)*len(pNode.S)    
    stSum = tree.stSum-sum(st1)+pNode.st
    d = tree.level(pid)+1
    p1 = (a+nChild).prod()/(a*(a+pNode.data.shape[0]))
    p2 = (tree.stSum/stSum)**npv
    pGrow = 1 if pid == 0 else (1-pPrune)
    rLike = sqrt(p1)*p2
    rTransit = pGrow/pPrune*w2/(b*nXS)
    rStruct = (d**beta-alpha)*nXS/(alpha*(1-alpha/(1+d)**beta)**2)
    r = rLike*rTransit*rStruct
    if random.uniform() < r:        
        for node in pNode.fpointer.copy():
            tree.remove_node(node)
        pNode.tag = str(pid)
        tree.stSum = stSum
        try:
            if len(tree.siblings(pid)[0].fpointer)==0: tree.w2 -= 1
        except: # root node does not have sibling
            tree.w2 = 0
        tree.dep = tree.depth()
        tree.show()
    return tree

def grow(tree):    
    global j
    #random.seed(98)
    leaf = [n.identifier for n in tree.leaves() if len(n.xvar)>0]
    b = len(leaf)
    if b<1: raise ValueError('cannot grow anymore')
    pid = random.choice(leaf) 
    pNode = tree[pid]
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
    nChild = array([df.shape[0] for df in childDF])
    st1 = [get_st(df) for df in childDF]
    stSum1 = tree.stSum-pNode.st+sum(st1)
    d = tree.level(pid)+1
    w2 = tree.w2
    newLevel = d>tree.dep
    if newLevel or len(tree.siblings(pid)[0].fpointer)==0: 
        w2+=1
    p1 = a*(a+len(df))/(a+nChild).prod()
    p2 = (tree.stSum/stSum1)**npv    
    pGrow = 1 if pid == 0 else (1-pPrune)
    rLike = sqrt(p1)*p2
    rTransit = pPrune/pGrow*b*nXS/w2    
    rStruct = alpha*(1-alpha/(1+d)**beta)**2/(d**beta-alpha)/nXS
    r = rLike*rTransit*rStruct
    if random.uniform() < r:            
        node1 = tree.create_node(str(j),j,parent=pid,data=childDF[0])
        node2 = tree.create_node(str(j+1),j+1,parent=pid,data=childDF[1])        
        pNode.tag = str(pid)+'-'+str(x)+'-'+str(s)
        pNode.S = S
        nodes = [node1,node2]
        if newLevel: tree.dep+=1
        tree.stSum = stSum1
        tree.w2 = w2
        for i in [0,1]:
            df = childDF[i]
            nodes[i].xvar = get_xvar(df)
            nodes[i].st = st1[i]
            nodes[i].data = df
        j+=2
        tree.show()
    return tree

def drawTree(tree):
    if tree.dep > 0:
        pGrow = 0.5
        if random.uniform() < pGrow:
            print('grow'+str(t))
            tree = grow(tree)
        else:
            print('prune'+str(t))
            tree = prune(tree)    
    else:    
        pGrow = 1
        tree = grow(tree)
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
tree.dep = 0
tree.w2 = 0
j = 1
p = array([0.5,0.5])
p0 = array([1,0.5])
pPrune = 0.5
for t in range(2000):
    tree = drawTree(tree)
    #tree.show()