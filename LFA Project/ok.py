import numpy as np
from cvi import * # fcm
import matplotlib.pyplot as plt
from numpy import genfromtxt
from random import randint
import operator
import math
from sklearn import metrics
import scipy.spatial
import copy
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import _tree

class FCM_SS_2:

    def __init__(self):
        self.fuzzy_param = 2.0
        self.membership_threshold = 0.95

    def fcm_get_u(self, x, v, m):
        distances = pairwise_squared_distances(x, v)
        nonzero_distances = np.fmax(distances, np.finfo(np.float64).eps)
        inv_distances = np.reciprocal(nonzero_distances)**(1/(m - 1))
        return inv_distances.T/np.sum(inv_distances, axis=1)

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

    def checkKnownEntries(self, dataset, labels, c, numberOfLabels):
        res = np.zeros((c, numberOfLabels))
        for cluster in range(c) :
            for i in range(len(dataset)):
                if(dataset[i][-1] != 0 and labels[i] == cluster):
                    res[cluster][int(dataset[i][-1] - 1)] += 1

        for cluster in range(c):
            c_total = res[cluster][0] + res[cluster][1]
            res[cluster][0] /= c_total
            res[cluster][1] /= c_total

        return res

    def getClass(self, dataset, labels, k, labels_names):
        count_class = np.zeros((k, len(labels_names)))
        newLabels = copy.copy(labels)

        for i in range(len(dataset)):
            if(dataset[i][-1] != 0):
                count_class[labels[i]][int(dataset[i][-1] - 1)] += 1

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

    def getClusters(self, dataset, membership_mat):
        cluster_labels = list()
        for i in range(len(dataset)):
            max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
            cluster_labels.append(idx)
        return cluster_labels


    def extractCluster(self, c, dataset, labels):
        cluster = np.empty((0, len(dataset[0])), float)
        indices = []
        for i in range(len(dataset)):
            if(labels[i] == c):
                cluster = np.vstack((cluster, dataset[i]))
                indices.append(i)
        return cluster, indices

    def pairwise_squared_distances(self, A, B):
        return scipy.spatial.distance.cdist(A, B)**2

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

    def setParams(self, fuzzy_param, membership_threshold):
    	self.fuzzy_param = fuzzy_param
    	self.membership_threshold = membership_threshold

    def clusterize(self, dataset, labels_names):
        result_labels  = np.zeros((len(dataset), len(dataset[0])), float)
        res_labels     = []
        result_mb      = [[]]
        result_centers = []
        temp_data      = copy.deepcopy(dataset)

        c    = 2
        done = False

        found_clusters = 1.
        result_index   = 0

        while(not done and (found_clusters + c) < math.sqrt(len(dataset))):
        	#Compute fcm with the current parameters and clusters number
            mb, centers = self.fcm(temp_data[:,:-1], c, m=self.fuzzy_param)
            labels = self.getClusters(temp_data[:,:-1], mb.T)

            #Check the produced cluster
            sup_verif = self.checkKnownEntries(temp_data, labels, c, len(labels_names))
            cluster_ok = []
            for i in range(c):
                if(sup_verif[i][0] < (1-self.membership_threshold) or sup_verif[i][0] >= self.membership_threshold):
                    cluster_ok.append(i)

            # if all clusters are good stop otherwise rerun with one more cluster
            if(len(cluster_ok) == 0):
                c =c + 1
            else:
                fhv_s = self.fhv(x = temp_data[:,:-1], v = centers, u = mb, m =2)
                pc_s  = self.pc(temp_data[:,:-1], mb, centers, 2)
                xb_s  = self.xb(x = temp_data[:,:-1], u = mb, v = centers, m = 2)

                if(pc_s > 0.7 and xb_s < 0.4 and fhv_s < 30):#fhv_s > 0.1 or
                    for i in range(len(temp_data)):
                            result_labels[result_index] = temp_data[i]
                            res_labels.append(int(found_clusters + labels[i]))
                            result_index += 1
                    for i in range(len(centers)):
                        result_centers += [centers[i]]
                    done = 1
                    break
                else:
                    for c in cluster_ok :
                        cluster, indices = self.extractCluster(c, temp_data, labels)

                    for i in range(len(cluster)):
                        result_labels[result_index] = cluster[i]
                        res_labels.append(found_clusters)
                        print(res_labels)
                        result_index += 1

                    result_centers += [centers[int(c)]]
                    found_clusters += 1
                    temp_data = np.delete(temp_data, indices, 0)

                c = c - len(cluster_ok) + 2
                if (c < 2):
                    c = 2

                continue
        print(res_labels)
        label =  self.getClass(result_labels, res_labels, len(result_centers), labels_names)

        print(label)
        return label, result_centers, mb
