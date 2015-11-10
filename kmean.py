import numpy as np
import random


#### Version that maintains IDs
def new_cluster_points(X, mu):
    clusters  = {}
    
    # this is for excluding IDs from the calculation
    tmp_mu = []
    for point in mu:
        tmp_mu.append(point[1:13])
 
    for x in X:
        tmp_x = x[1:13]
        # norm calculates the distance of a vector 
        # In this formula, it cacluates the distance between the sample vectors and all the other vectors, and select the min value as the best mean
        bestmukey = min([(i[0], np.linalg.norm(tmp_x-tmp_mu[i[0]])) for i in enumerate(tmp_mu)], key=lambda t:t[1])[0]

        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
  
 
def new_reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())

    for k in keys:
        tmp_mu = []
    
        for point in clusters[k]:
            tmp_mu.append(point[1:13])

        newmean = np.mean(tmp_mu, axis = 0)
        newmean = np.insert(newmean, 0, 0)
        newmu.append(newmean)
    return newmu
 
def new_has_converged(mu, oldmu):
    tmp_mu = []
    tmp_oldmu = []
    for point in mu:
        tmp_mu.append(point[1:13])
        
    for point in oldmu:
        tmp_oldmu.append(point[1:13])
        
    return (set([tuple(a) for a in tmp_mu]) == set([tuple(a) for a in tmp_oldmu]))

def new_find_centers(X, K):
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not new_has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = new_cluster_points(X, mu)
        # Reevaluate centers
        mu = new_reevaluate_centers(oldmu, clusters)
        
    try:
        clusters
    except: 
        clusters = new_cluster_points(X, mu)  # added to avoid null cluster
    return(mu, clusters)

### Original clustering functions without maintaining IDs (allowing multiple dimensions)
def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
 
def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
        
    try:
        clusters
    except: 
        clusters = cluster_points(X, mu)  # added to avoid null cluster
        
    return(mu, clusters)


def Wk(mu, clusters):
    K = len(mu)     
    try:
        r = sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) for i in range(K) for c in clusters[i]])
    except:
        r = 1
        print("index error")
    return r 

def bounding_box(X):
    
    size = len(X[0])
    xmins = [0 for x in range(size)]
    xmaxs = [0 for x in range(size)]
    
    for i in range(0, size):        
        xmins[i], xmaxs[i] = min(X,key=lambda a:a[i])[i], max(X,key=lambda a:a[i])[i]        
    return (xmins,xmaxs)
 
def gap_statistic(X, num_k):
    (xmins,xmaxs) = bounding_box(X)

    # Dispersion for real distribution
    ks = range(1,num_k)
    Wks = np.zeros(len(ks))
    Wkbs = np.zeros(len(ks))
    sk = np.zeros(len(ks))
    
    
    for indk, k in enumerate(ks):
        print("K:" + str(k))
        mu, clusters = find_centers(X,k)
        Wks[indk] = np.log(Wk(mu, clusters))
        # Create B reference datasets
        B = 10
        
        BWkbs = np.zeros(B)
        for i in range(B):
#             print("B: " + str(i))
            Xb = []
            for n in range(len(X)):                             
                randomvalues = []
                for index in range(len(xmins)):
                    randomvalues.insert(0, random.uniform(xmins[index], xmaxs[index]))
                Xb.append(randomvalues)
            Xb = np.array(Xb)
            mu, clusters = find_centers(Xb,k)
        
            BWkbs[i] = np.log(Wk(mu, clusters))            
        Wkbs[indk] = sum(BWkbs)/B
        sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)

    sk = sk*np.sqrt(1+1/B)
    return(ks, Wks, Wkbs, sk)


#example 
input_list = np.array([[1, 2], [4, 5], [4, 3], [4, 5], [3, 3], [1, 3], [7, 8]])
num_k=3

# to start the gap analysis to determin K
ks, logWks, logWkbs, sk = gap_statistic(input_list, num_k)
print (ks, logWks, logWkbs, sk)
