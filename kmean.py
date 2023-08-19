

##  K- Mean --- four steps  1. Decide n clusters ,2. random centroid ,3.Assign clusters , 4. move centroid ,5. finish.



import random 
import numpy as np

class Kmeans:

    def __init__(self,n_clusters=2,max_iter=100):
        self.n_clusters = n_clusters # number of clusters
        self.max_iter   = max_iter  # maximum iterations
        self.centroids = None


    def fit_predict(self,X):
        random_point_index = random.sample(range(0,X.shape[0]),self.n_clusters)    ##--step 2 random centroid
        self.centroids = X[random_point_index]

        for i in range(self.max_iter):

            cluster_group =  self.assign_clusters(X)    ## Assign Cluster


            old_centroids = self.centroids              ## move centroid 

            self.centroids = self.move_centroids(X,cluster_group)    
                                                          

            if (old_centroids==self.centroids).all():        ## finish
                break                                              


        return cluster_group


    def assign_clusters(self,X):
        cluster_group = []
        distances = []

        for rows in  X:
            for centroid in self.centroids:              ## distance btw point and centroid

                distances.append(np.sqrt(np.dot(rows-centroid,rows-centroid)))
        
            
            min_distances = min(distances)
            index_positioin = distances.index(min_distances)
            cluster_group.append(index_positioin)

            distances.clear()

                  


        return np.array(cluster_group)
    


    def move_centroids(self,X,cluster_group):
        new_centroids = []

        cluster_type = np.unique(cluster_group)

        for type in cluster_type:
            new_centroids.append(X[cluster_group==type].mean(axis=0))

        return np.array(new_centroids)