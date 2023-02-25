import numpy as np
from numpy import random, exp, log, ones, zeros, sqrt, array
from binarytree import Node, tree
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats
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

#%%
df = genData(seed=98)
s = df['y'].var()
mu = df['y'].mean()
n = len(df)

v = 12; lamda = 4
vlamda = v*lamda
ig1 = v/2; ig2 = lamda*ig1
a = 1/3
lna = log(a)
nv2 = -0.5*(n+v)

#xx = np.linspace(0,20,1000)
#fx = stats.invgamma.pdf(xx,ig1,scale=ig2)
#plt.plot(xx,fx)
#plt.axvline(x=ig2/(ig1+1),color='red')

t = Node(0,Node(1),Node(2))
t.left.left = Node(3)
t.left.right = Node(4)
print(t)
#%%
t = Node(0)
t.left = Node(1)
t.right = Node(2)
leaves = t.leaves
b = len(leaves)

DF = []
i1 = df['x2'].isin(['c','d'])
df1 = df.loc[i1]
df2 = df.loc[~i1]
DF.append(df1)
DF.append(df2)

def mvl(df):
    y = df['y']
    return y.mean(), y.var(), len(y)


mat = array(list(map(mvl,DF)))
N = mat[:,-1]

lnp = 0.5*b*lna-0.5*sum(log(N+a))
s = (N-1)*mat[:,1]
t = N*a/(N+a)*(mat[:,0]-mu)**2
lnp += nv2*log(sum(s+t)+vlamda)