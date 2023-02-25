   p1 = a**(len(leaf1)-len(leaf))*(array([len(n.data) for n in leaf])+a).prod() \
       /(array([len(n.data) for n in leaf1])+a).prod() 
   p2 = (tree.stSum/stSum1)**npv

def genTree0(sList,df):
    tree = Tree()
    root = tree.create_node(tag='0',identifier=0,data=df)
    root.st = get_st(root.data)
    root.xvar = get_xvar(df)
    tree.stSum = root.st+vlamda
    tree.dep = 0
    tree.w2 = 0
    j = 1
    for split in sList:
        pid,x,s = split
        dtype = DataTypes[x]
        xVal = set(df.iloc[:,x])
        pNode = tree[pid]
        if dtype == 'O':
            S = powerset(xVal)
            idx = df.iloc[:,x].isin(s)
        elif (dtype == 'i') | (dtype == 'f'):
            S = sorted(xVal)[:-1] #leave the last element out
            idx = (df.iloc[:,x] <= s)
        childDF = [df.loc[idx],df.loc[~idx]] 
        st1 = [get_st(df) for df in childDF]
        node1 = tree.create_node(str(j),j,parent=pid,data=childDF[0])
        node2 = tree.create_node(str(j+1),j+1,parent=pid,data=childDF[1])  
        pNode.tag = str(pid)+'-'+str(x)+'-'+str(s)
        pNode.S = S
        nodes = [node1,node2]
        newLevel = tree.level(pid)>=tree.dep
        if newLevel or len(tree.siblings(pid)[0].fpointer)==0: 
            tree.w2+=1
        if newLevel: tree.dep+=1
        tree.stSum = tree.stSum-pNode.st+sum(st1)
        for i in [0,1]:
            df = childDF[i]
            nodes[i].xvar = get_xvar(df)
            nodes[i].st = st1[i]
            nodes[i].data = df
        j+=2
    return tree,j  

def recurTree(tree,parent,d,a,b,j):
    u = random.uniform()
    if u > a/d**b:
        return 
    else:
        n1 = tree.create_node(j,j,parent=parent)
        n2 = tree.create_node(j+1,j+1,parent=parent)
        for node in [n1,n2]:
            recurTree(tree,node,d+1,a,b,node.identifier*2+1)

def getRandomTree(alpha,beta,root=0,seed=None):
    tree = Tree()
    root = tree.create_node(root,root)
    if seed is not None: random.seed(seed)
    recurTree(tree,root,1,alpha,beta,1)
    return tree
        
		