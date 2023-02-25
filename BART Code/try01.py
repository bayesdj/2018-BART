import numpy as np
from numpy import exp, log, ones, zeros, sqrt, array, arange
import random
from treelib import Node, Tree
import pandas as pd
from itertools import chain, combinations
from copy import deepcopy
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
        np.random.seed(seed)
    x1val = np.arange(1,11)
    x2val = ['a','b','c','d']    
    x1 = np.random.choice(x1val,size=n)
    x2 = np.random.choice(x2val,size=n)
    y = array(list(map(f1,tuple(zip(x1,x2))))) + 2*np.random.normal(size=n)
    return pd.DataFrame({'y':y,'x1':x1,'x2':x2})

def powerset(s):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    n = len(s)
    S = chain.from_iterable(combinations(s, r) for r in range(1,n))
    return [set(i) for i in S]

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
    leaf = filter(lambda n:tree.siblings(n.identifier)[0].var==None,leaf0)
    leaf2 = list(set([node.bpointer for node in leaf])) 
    w2 = len(leaf2)
    pid = random.choice(leaf2)
    pNode = tree[pid]
    leaf = [n.identifier for n in leaf0 if (n.identifier not in pNode.fpointer)
                    and len(n.xvar)>0 ] # leaf for grow
    b = len(leaf)+1
    nChild = array([tree[c].data.shape[0] for c in pNode.fpointer])
    nXS = len(pNode.xvar)*len(pNode.S)    
    d = tree.level(pid)+1
    df = pNode.data
    R0 = tree.R[df.index].sum()**2
    R1 = array([tree.R[tree[c].data.index].sum()**2 for c in pNode.fpointer])
    vp = var+len(df)*var_mu
    vc = var+nChild*var_mu
    p1 = vc.prod()/(var*vp)
    p2 = R0/vp - (R1/vc).sum()
    pGrow = 1 if pid == 0 else Prob[0]
    pPrune = Prob[1]-Prob[0]
    rLike = sqrt(p1)*exp(0.5*var_mu/var*p2)
    rTransit = pGrow/pPrune*w2/(b*nXS)
    rStruct = (d**beta-alpha)*nXS/(alpha*(1-alpha/(1+d)**beta)**2)
    r = rLike*rTransit*rStruct
    print('prune '+str(t)+': '+pNode.tag+'; r='+str(r.round(4)))
    if random.uniform(0,1) < r:        
        for node in pNode.fpointer.copy():
            tree.remove_node(node)
        pNode.tag = str(pid)
        pNode.var = None
        pNode.S = None
        try:
            if len(tree.siblings(pid)[0].fpointer)==0: tree.w2 -= 1
        except: # root node does not have sibling
            tree.w2 = 0
        tree.dep = tree.depth()
        tree.show()
    return tree

def grow(tree):    
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
        s = random.choice(S)
        idx = df.iloc[:,x].isin(s)
    elif (dtype == 'i') | (dtype == 'f'):
        S = sorted(xVal)[:-1] #leave the last element out
        s = random.choice(S)
        idx = (df.iloc[:,x] <= s)
    nXS = len(pNode.xvar)*len(S)
    #if nXS <= 0: raise ValueError('nXS <= 0')
    childDF = [df.loc[idx],df.loc[~idx]] 
    R0 = tree.R[df.index].sum()**2
    R1 = array([tree.R[df.index].sum()**2 for df in childDF])
    nChild = array([df.shape[0] for df in childDF])
    d = tree.level(pid)+1
    w2 = tree.w2
    newLevel = d>tree.dep
    if newLevel or len(tree.siblings(pid)[0].fpointer)==0: 
        w2+=1
    vp = var+len(df)*var_mu
    vc = var+nChild*var_mu
    p1 = var*vp/vc.prod()
    p2 = (R1/vc).sum()-R0/vp
    pGrow = Prob[0]; pPrune = ProbDefault[1]-ProbDefault[0]
    rLike = sqrt(p1)*exp(0.5*var_mu/var*p2)
    rTransit = pPrune/pGrow*b*nXS/w2    
    rStruct = alpha*(1-alpha/(1+d)**beta)**2/(d**beta-alpha)/nXS
    r = rLike*rTransit*rStruct
    print('grow '+str(t)+': '+str((pid,x,s))+'; r='+str(r.round(4)))
    if random.uniform(0,1) < r:      
        cid = array([pid,pid])*2+array([1,2])
        nodes = [None,None]
        for i in range(2):
            df = childDF[i]
            if len(df)<=0: raise Exception('no data in leaf')
            nodes[i] = tree.create_node(str(cid[i]),cid[i],parent=pid,data=childDF[i])
            nodes[i].xvar = get_xvar(df)
            nodes[i].var = None        
        pNode.tag = str(pid)+'-'+str(x)+'-'+str(s)
        pNode.var = x
        pNode.split = s
        pNode.S = S
        if newLevel: tree.dep+=1
        tree.w2 = w2
        tree.show()
    return tree

def genTree(node,sList=[]):    
    nodeType = type(node)
    tree = Tree()
    if nodeType == pd.core.frame.DataFrame:        
        df = node
        root = tree.create_node(tag='0',identifier=0,data=df)
        root.xvar = get_xvar(df)
    elif nodeType == Node:
        root = deepcopy(node)
        root.fpointer = []
        df = root.data
        tree.add_node(root)
    tree.dep = 0
    tree.w2 = 0
        
    for split in sList:
        pid,x,s = split
        pNode = tree[pid]
        df = pNode.data
        dtype = DataTypes[x]
        xVal = set(df.iloc[:,x])
        if dtype == 'O':
            S = powerset(xVal)
            idx = df.iloc[:,x].isin(s)
        elif (dtype == 'i') | (dtype == 'f'):
            S = sorted(xVal)[:-1] #leave the last element out
            idx = (df.iloc[:,x] <= s)
        childDF = [df.loc[idx],df.loc[~idx]] 
        nodes = [None,None]
        cid = array([pid,pid])*2+array([1,2])
        for i in range(2):
            if len(childDF[i]) <= 0: raise Exception('no data in leaf')
            nodes[i] = tree.create_node(str(cid[i]),cid[i],parent=pid,data=childDF[i])
            nodes[i].var = None
            nodes[i].xvar = get_xvar(childDF[i])             
        pNode.tag = str(pid)+'-'+str(x)+'-'+str(s)
        pNode.S = S
        pNode.var = x
        pNode.split = s
        newLevel = tree.level(pid)>=tree.dep
        if newLevel:
            tree.w2+=1
            tree.dep+=1
        elif len(tree.siblings(pid)[0].fpointer)==0: 
            tree.w2+=1
    return tree         

def getChoice(tree,nid):
    df = tree[nid].data
    p = df.shape[1]
    choices = [sorted(set(df.iloc[:,i])) for i in range(1,p)]
    childID = tree[nid].fpointer
    ileft = [[] for i in range(p)]
    iright = [[] for i in range(p)] # must be deep copy
    for r in recurTag(tree,childID[0]):
        ileft[r[1]].append(r[2])
    for r in recurTag(tree,childID[1]):
        iright[r[1]].append(r[2])
    for i in range(1,p):
        type_i = DataTypes[i]
        choice_i = choices[i-1]
        llegit = set(choice_i)
        rlegit = llegit
        if len(ileft[i]) > 0:            
            if type_i == 'f' or type_i == 'i':
                minValue = max(ileft[i])                
                j = choice_i.index(minValue)+1
                llegit = set(choice_i[j:])
            elif type_i == 'O':
                choices[i-1] = []
        if len(iright[i]) > 0:            
            if type_i == 'f' or type_i == 'i':
                maxValue = min(iright[i])
                j = choice_i.index(maxValue)+1
                rlegit = set(choice_i[:j])
            elif type_i == 'O':
                choices[i-1] = []
        if type_i == 'f' or type_i == 'i':
            # must leave one value out
            choices[i-1] = sorted(llegit.intersection(rlegit))[:-1] 
        elif type_i == 'O' and choices[i-1] != []:
            choices[i-1] = powerset(choices[i-1])
    return choices

# make sure obj data type's ancestor are not in selection
def nidValid(tree):
    internalNodes = [n for n in tree.all_nodes_itr() if n.var != None]
    nObj = (n for n in internalNodes if DataTypes[n.var]=='O') 
    noChange = set()
    for node in nObj:
        current = node
        while current.identifier != tree.root:
            ancestor = tree[current.bpointer]
            if ancestor.var == node.var:
                noChange.add(ancestor.identifier)
                break
            current = ancestor
    nInternal = [n.identifier for n in 
                 filter(lambda n:n.identifier not in noChange, internalNodes)]
    return nInternal
       
def change(tree):    
    nidInternal = nidValid(tree)
    choices = [getChoice(tree,n) for n in nidInternal]
    n_choices = map(lambda L: sum([len(i) for i in L]),choices)
    choiceDic = {a:b for (a,b,c) in zip(nidInternal,choices,n_choices) if c > 1}
    choices1 = list(choiceDic.keys())
    nid = random.choice(choices1)    
    p = tree[nid].data.shape[1]
    x0 = tree[nid].var; s0 = tree[nid].split
    choices = choiceDic[nid] # choose nid to split
    choices[x0-1].remove(s0) # remove original split option
    choices2 = [i+1 for i in range(p-1) if len(choices[i])>0] # choose var to split
    x = random.choice(choices2)
    choices3 = choices[x-1] # choose value to split
    s = random.choice(choices3)
    tree1 = Tree(tree,deep=True)
    pid = tree1[nid].bpointer
    sub = tree1.remove_subtree(nid)
    tags = recurTag(sub,nid)
    tags[0] = (nid,x,s)
    sub1 = genTree(sub[nid],tags)
    if pid is not None:
        tree1.paste(pid,sub1)
        tree1[pid].fpointer = sorted(tree1[pid].fpointer)
    else:
        tree1 = sub1
    nidInternal1 = set(nidValid(tree1))
    choices1 = set(choices1)
    choices11 = nidInternal1.intersection(choices1)
    extra = nidInternal1-choices1
    n_choices = map(lambda L: sum([len(i) for i in L]),
                    [getChoice(tree,n) for n in extra])
    choices11 = list(choices11)+[a for (a,b) in zip(extra,n_choices) if b > 1]
    choices31 = getChoice(tree1,nid)[x0-1]
    rTransit = len(choices1)*len(choices3)/(len(choices11)*(len(choices31)-1))
    leaf = sub.leaves(); leaf1 = sub1.leaves()
    stSum1 = tree.stSum-sum((i.st for i in leaf))+sum((i.st for i in leaf1))    
    p1 = 0.5*((len(leaf1)-len(leaf))*log(a) + 
              log(a+array([len(n.data) for n in leaf])).sum() -
              log(a+array([len(n.data) for n in leaf1])).sum())
    p2 = npv*log(tree.stSum/stSum1)
    rLike = exp(p1+p2)
    nXS = [len(n.xvar)*len(n.S) for n in sub.all_nodes_itr() if n.var is not None]
    nXS1 = [len(n.xvar)*len(n.S) for n in sub1.all_nodes_itr() if n.var is not None]
    rStruct = array(nXS).prod()/array(nXS1).prod()
    r = rLike*rTransit*rStruct
    print('change '+str(t)+': '+str(tags[0])+'; r='+str(r.round(4)))
    if random.uniform(0,1) < r:    
        tree1.stSum = stSum1
        tree1.dep = tree.dep
        tree1.w2 = tree.w2        
        tree1.show()
        return tree1
    return tree

def str2num(x):
    try: return float(x)
    except: return x
    
def recurTag(tree,i):
    node = tree[i]
    childID = node.fpointer
    if childID != []:
        tag = (node.identifier,node.var,node.split)
        return [tag] + recurTag(tree,childID[0]) + recurTag(tree,childID[1])
    return []
        
def drawTree(tree):
    global Prob
    if tree.dep > 0:        
        #print(Prob)
        u = random.uniform(0,1)
        Prob = ProbDefault
        if  u < Prob[0]:
            tree = grow(tree)
        elif u < Prob[1]:
            tree = prune(tree)    
        else:
            tree = change(tree)
    else:   
        #print(Prob)
        Prob = [1,1]
        tree = grow(tree)    
    return tree
#%%
v = 12; lamda = 4
vlamda = v*lamda
ig1 = v/2; ig2 = lamda*ig1

df0 = genData()
df0['x3'] = 22
y = df0['y']
var = y.var()
lamda = 1/var
mu = y.mean()
n0,p = df0.shape
a = ig1+n0*0.5-1

k = 2; m = 1 # m is here
mumu = (y.min()+y.max())*0.5/m
sigma_mu = (y.max()-m*mumu)/(k*sqrt(m))
var_mu = sigma_mu**2
lamda_mu = 1/var_mu


alpha = 0.95
beta = 1
DataTypes = df0.dtypes.map(lambda x:x.kind)
ProbDefault = [0.5,1]

sList = [(0,2,{'a','b'}),(1,1,5),(2,1,3),(6,1,7)]
trueTree = genTree(df0,sList)
#trueTree.show()
#%%
tree = Tree()
root = tree.create_node('0',0,data=df0)
root.xvar = get_xvar(df0)
root.var = None
tree.dep = 0
tree.w2 = 0
tree = trueTree

T = 100
trees = [deepcopy(tree) for i in range(m)]
MM = pd.DataFrame(index=df0.index,columns=range(m))

for i in range(m):
    for l in trees[i].leaves():
        idx = l.data.index
        MM.loc[idx,i] = l.data['y'].mean()/m
    #M[i] = [l.data['y'].mean()/m for l in leaves]
#%%
for t in range(T):
    for i in range(m):
        trees[i].R = df0['y']-(MM.sum(axis=1)-MM.iloc[:,i])
        tree = drawTree(trees[i])
        for l in tree.leaves():
            n = l.data.shape[0]
            idx = l.data.index
            lamda_mu_n = n*lamda + lamda_mu
            mu = (n*lamda*tree.R[idx].mean()+lamda_mu*mumu)/lamda_mu_n
            M = np.random.normal(mu,sqrt(1/lamda_mu_n))
            MM.loc[idx,i] = M
        trees[i] = tree
        b = ig2+0.5*sum((df0['y']-MM.sum(axis=1))**2)
        lamda = np.random.gamma(a,1/b)
        var = 1/lamda
        #tree = drawTree(trees[i])

#print(tag)
#print(z)