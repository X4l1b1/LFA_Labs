#############################################
# HEIG-VD - LFA - Projet
# Selective Fuzzy C-Mean Clustering
# Arthur Passuelo & Francois Quellec
# Mai-Juin 2018
#############################################

import numpy as np
from cvi import * # fcm
from random import randint
import operator
import math
import scipy.spatial
import copy

class S_FCM:

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
                if(dataset[i][-2] != 0 and labels[i] == cluster):
                    res[cluster][int(dataset[i][-2] - 1)] += 1

        # Then we compute the proportion
        for cluster in range(c):
            c_total = 0
            for j in range(numberOfLabels):
                c_total = c_total + res[cluster][j]
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
            if(dataset[i][-2] != 0):
                count_class[int(labels[i])][int(dataset[i][-2] - 1)] += 1

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

            
    # Get the indices of the differents clusters we have in labels
    def extractCluster(self, c, dataset, labels):
        if(len(dataset) == 0):
            return [],[]
        cluster = np.empty((0, len(dataset[0])), float)
        indices = []
        for i in range(len(dataset)):
            if(labels[i] == c):
                cluster = np.vstack((cluster, dataset[i]))
                indices.append(i)
        return cluster, indices

    # Calculate the covariance
    def calculate_covariances(self,x, u, v, m):
        c, n = np.array(u).shape
        d = np.array(v).shape[1]

        um = np.array(u)**m

        covariances = np.zeros((c, d, d))

        for i in range(c):
            xv = x - v[i]
            uxv = um[i, :, np.newaxis]*xv
            covariances[i] = np.einsum('ni,nj->ij', uxv, xv)/np.sum(um[i])

        return covariances

    # Calculate the pairwise squared distances
    def pairwise_squared_distances(self, A, B):
        return scipy.spatial.distance.cdist(A, B)**2

    # Partition Coefficient
    def pc(self, x, u, v, m):
        c, n = np.array(u).shape
        return np.square(np.array(u)).sum()/n

    # Fuzzy Hyperbolic Volume
    def fhv(self, x, u, v, m):
        covariances = self.calculate_covariances(x, u, v, m)
        return sum(np.sqrt(np.linalg.det(cov)) for cov in covariances)

    # Xie-Beni Index
    def xb(self, x, u, v, m):
        n = np.array(x).shape[0]
        c = np.array(v).shape[0]

        um = np.array(u)**m

        d2 = pairwise_squared_distances(x, v)
        v2 = pairwise_squared_distances(v, v)

        v2[v2 == 0.0] = np.inf

        return np.sum(um.T*d2)/(n*np.min(v2))


    # Main function, compute the FI-FCM algorithm
    # Return the new labels, the centers of each cluster and the membership matrix 
    def clusterize(self, dataset, labels_names):
        result_labels  = np.zeros((len(dataset), len(dataset[0])), float)
        res_labels     = []
        result_mb      = [[]]
        result_centers = []
        temp_data      = copy.deepcopy(dataset)

        c    = len(labels_names)
        done = False

        found_clusters = 0
        result_index   = 0

        # We loop until the number of cluster is too big (sqrt(number of elements))
        # or the cluster are good enough
        while(not done and (found_clusters + c) < math.sqrt(len(dataset))):
        	#Compute fcm with the current parameters and clusters number
            if(len(temp_data) == 0):
                done = True
                break
            mb, centers = self.fcm(temp_data[:,:-2], c, m=self.fuzzy_param)
            labels = self.getClusters(temp_data[:,:-2], mb.T)

            #Check the produced cluster
            sup_verif = self.checkKnownEntries(temp_data, labels, c, len(labels_names))
            cluster_ok = []
            for i in range(c):
                ok = False
                for j in range(len(sup_verif[i])):
                    if(sup_verif[i][j] > self.membership_threshold):
                        ok = True
                    if(ok == True):
                        cluster_ok.append(i)
   
            # if there is no cluster with membership threshold of one class we
            # run again the fcm with one more cluster
            if(len(cluster_ok) == 0):
                c =c + 1
            else:
                # Else we check if all the cluster respect the 3 following indicators
                fhv_s = self.fhv(x = temp_data[:,:-2], v = centers, u = mb, m =self.fuzzy_param)
                pc_s  = self.pc(temp_data[:,:-2], mb, centers, self.fuzzy_param)
                xb_s  = self.xb(x = temp_data[:,:-2], u = mb, v = centers, m = self.fuzzy_param)

                # If it's the case we stop
                if(pc_s > 0.82 and xb_s < 0.25 and fhv_s < 3):
                    for i in range(len(temp_data)):
                            result_labels[result_index] = temp_data[i]
                            res_labels.append(int(found_clusters + labels[i]))
                            result_index += 1
                    for i in range(len(centers)):
                        result_centers += [centers[i]]
                    done = True

                # Otherwise we keep the good clusters and continue to loop
                else:
                    for c in cluster_ok :
                        cluster, indices = self.extractCluster(c, temp_data, labels)

                        for i in range(len(cluster)):
                            result_labels[result_index] = cluster[i]
                            res_labels.append(found_clusters)
                            result_index += 1

                        result_centers += [centers[int(c)]]
                        found_clusters += 1
                        temp_data = np.delete(temp_data, indices, 0)

                c = c - len(cluster_ok) + 2
                if (c < 2):
                    c = 2
                continue

        # Make sure that we have the good number of clusters and prepare the data to
        # extract the class
        if(len(res_labels) < len(result_labels)):
            for i in range(max(labels) + 1):
                if(len(temp_data) == 0):
                    break
                cluster, indices = self.extractCluster(i, temp_data, labels)
                for j in range(len(cluster)):
                    result_labels[result_index] = cluster[j]
                    res_labels.append(found_clusters)
                    result_index += 1

                result_centers += [centers[i]]
                found_clusters += 1

                temp_data = np.delete(temp_data, indices, 0)
                labels    = np.delete(labels, indices, 0)

        label =  self.getClass(result_labels, res_labels, len(result_centers), labels_names)
        done = False
        return result_labels, label, result_centers, mb
