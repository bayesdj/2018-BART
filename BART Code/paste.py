
sList = [(0,2,{'a','b'}),(1,1,5),(2,1,3),(6,1,7)]
trueTree = genTree(df0,sList)
trueTree.leaf = [n.identifier for n in trueTree.leaves() if len(n.xvar)>0]
trueTree.show()
#%%

def get_likelihood(node,R):
    df = node.data
    idx = df.index
    Ri = R[idx]
    mu = Ri.mean()
    n = len(Ri)
    nmumu = n*mu*mu
    x = ((Ri-mu)**2).sum()-nmumu*n/(n+var_ratio)+nmumu
    lnP = n/2*log(lamda/(2*pi))+0.5*log(var_ratio/(var+var_ratio))
    return lnP-0.5*lamda*x

def sumLike2(i):
    tree = trees[i]
    R = RR.iloc[:,i]
    leaf = tree.leaves()
    like = map(lambda n:get_likelihood(n,R),leaf)
    return sum(like)
	
def sumLike(tree,R):
    leaf = tree.leaves()
    x = map(lambda n:get_likelihood(n,R),leaf)
    return sum(x)
	
#        for l in tree.leaves():
#            n = l.data.shape[0]
#            idx = l.data.index
#            lamda_mu_n = n*lamda + lamda_mu
#            mu = (n*lamda*tree.R[idx].mean()+lamda_mu*mumu)/lamda_mu_n
#            M = np.random.normal(mu,sqrt(1/lamda_mu_n))
#            MM.loc[idx,mi] = M