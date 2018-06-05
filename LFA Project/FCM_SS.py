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

class FCM_SS:

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

    def setParams(self, fuzzy_param, membership_threshold):
    	self.fuzzy_param = fuzzy_param
    	self.membership_threshold = membership_threshold

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