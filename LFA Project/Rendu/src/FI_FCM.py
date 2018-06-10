#############################################
# HEIG-VD - LFA - Projet
# Full Iterative Fuzzy C-Mean Clustering
# Arthur Passuelo & Francois Quellec
# Mai-Juin 2018
#############################################

import numpy as np
from random import randint
from cvi import * # fcm
import operator
import math
import copy

class FI_FCM:

    # Initializator
    # default parameters: fuzzy_param = 2.0; membership_threshold= 0.95
    def __init__(self):
        self.fuzzy_param = 2.0
        self.membership_threshold = 0.95

    # Change the default parameters
    def setParams(self, fuzzy_param, membership_threshold):
        self.fuzzy_param = fuzzy_param
        self.membership_threshold = membership_threshold

    # Update the membership grades of each data for every cluster
    def fcm_get_u(self, x, v, m):
        distances = pairwise_squared_distances(x, v)
        nonzero_distances = np.fmax(distances, np.finfo(np.float64).eps)
        inv_distances = np.reciprocal(nonzero_distances)**(1/(m - 1))
        return inv_distances.T/np.sum(inv_distances, axis=1)

    # Basic FCM algorithm return the membership grades 
    # matrix and the centers of every cluster
    def fcm(self, x, c, m=2.0, v=None, max_iter=100, error=0.05):
        if v is None: v = x[np.random.randint(x.shape[0], size=c)]
        u = self.fcm_get_u(x, v, m)
        for iteration in range(max_iter):
            u_old = u
            um = u**m
            v = np.dot(um, x)/np.sum(um, axis=1, keepdims=True)
            u = self.fcm_get_u(x, v, m)
            if np.linalg.norm(u - u_old) < error: break
        return u, v

    # Check how much known label of each class there is in each cluster
    # return the percentage of known entries of each class in each cluster
    def checkKnownEntries(self, dataset, labels, c, numberOfLabels):
        # First we count the number of known entries
        res = np.zeros((c, numberOfLabels))
        for cluster in range(c) :
            for i in range(len(dataset)):
                if(dataset[i][-1] != 0 and labels[i] == cluster):
                    res[cluster][int(dataset[i][-1] - 1)] += 1
        
        # Then we compute the proportion
        for cluster in range(c):
            c_total = 0
            for j in range(numberOfLabels):
                c_total += res[cluster][j]
            for label in range(numberOfLabels):
                if(c_total == 0):
                    res[cluster][label] = 0
                else:
                    res[cluster][label] = res[cluster][label]/c_total

        return res

    # Get the output class of each cluster
    # Take the most represented class from the known entries
    # If there is no known entries in a cluster or there is an equal number of each class
    # the class become 0 -> unknown 
    def getClass(self, dataset, labels, k, labels_names):
        count_class = np.zeros((k, len(labels_names)))
        newLabels = copy.copy(labels)

        # Count the number of known of each class in each cluster entries
        for i in range(len(dataset)):
            if(dataset[i][-1] != 0):
                count_class[labels[i]][int(dataset[i][-1] - 1)] += 1

        # Take the most represented class to classify the cluster
        for c in range(len(count_class)):  
            indexPredominantClass = np.argmax(count_class[c])
            if np.max(count_class[c]) == 0:
                cl = 0
            else :
                cl = labels_names[indexPredominantClass]
                
            for i in range(len(labels)):
                if labels[i] == c:
                    newLabels[i] = cl 

        return newLabels

    # Get the cluster of each data depending on their grade in the membership matrix
    def getClusters(self, dataset, membership_mat):
        cluster_labels = list()
        for i in range(len(dataset)):
            max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
            cluster_labels.append(idx)
        return cluster_labels

    # Main function, compute the FI-FCM algorithm
    # Return the new labels, the centers of each cluster and the membership matrix 
    def clusterize(self, dataset, labels_names):
        result_labels = [[]]
        result_mb     = [[]]
        temp_data = copy.deepcopy(dataset)
        c = 2
        done = False
        while(not done and c < math.sqrt(len(dataset))): 
        	#Compute fcm with the current parameters and clusters number
            mb, centers = self.fcm(dataset[:,:-1], c, m=self.fuzzy_param)
            labels = self.getClusters(dataset[:,:-1], mb.T)

            #Check the produced cluster
            sup_verif = self.checkKnownEntries(dataset, labels, c, len(labels_names))
            cluster_ok = []
            for i in range(c):
                if(sup_verif[i][0] < (1-self.membership_threshold) or sup_verif[i][0] >= self.membership_threshold):
                    cluster_ok.append(i)

            # if all clusters are good stop otherwise rerun with one more cluster
            if(len(cluster_ok) < c):
                c =c + 1
            else:
                done = True
               
        labels = self.getClass(dataset, labels, len(centers), labels_names)
        return labels, centers, mb