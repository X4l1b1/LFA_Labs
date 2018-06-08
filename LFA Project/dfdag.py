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
                count_class[int(labels[i])][int(dataset[i][-1] - 1)] += 1

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

    def extractCluster(self, c, dataset, labels):
        cluster = np.empty((0, len(dataset[0])), float)
        indices = []
        for i in range(len(dataset)):
            if(labels[i] == c):
                cluster = np.vstack((cluster, dataset[i]))
                indices.append(i)
        return cluster, indices

    def clusterize(self, dataset, labels_names):

      result_labels  = np.zeros((len(dataset), len(dataset[0])), float)
      res_labels     = []
      result_mb      = [[]]
      result_centers = []
      temp_data      = copy.deepcopy(dataset)
      c    = 2
      done = False
      print('datasetl',len(dataset))
      found_clusters = 0
      result_index   = 0

      while(not done and (found_clusters) < math.sqrt(len(dataset))):
            #Compute fcm with the current parameters and clusters number
        mb, centers = self.fcm(temp_data[:,:-1], c, m=self.fuzzy_param)
        labels = self.getClusters(temp_data[:,:-1], mb.T)
        print('labels', len(res_labels))

        #Check the produced cluster
        sup_verif = self.checkKnownEntries(temp_data, labels, c, len(labels_names))
        all_zeros = 1
        for val in sup_verif:
            for v in val :
                if(v != 0):
                    all_zeros = 0
        cluster_ok = []
        for i in range(c):
            if(sup_verif[i][0] < (1-self.membership_threshold) or sup_verif[i][0] >= self.membership_threshold):
                cluster_ok.append(i)

        # if all clusters are good stop otherwise rerun with one more cluster
        if(len(cluster_ok) == 0):
            c =c + 1
        elif(len(cluster_ok) == c or all_zeros == 1):
            print('coucou labels', len(res_labels))
            done = 1
            for i in range(len(temp_data)):
                    result_labels[result_index] = temp_data[i]
                    res_labels.append(int(found_clusters + labels[i]))
                    result_index += 1
            for i in range(len(centers)):
                result_centers += [centers[i]]
            print('coucou labels', len(res_labels))

        else:
            for c in cluster_ok :
                print('ecoucou labels', len(res_labels))
                cluster, indices = self.extractCluster(c, temp_data, labels)

                for i in range(len(cluster)):
                    result_labels[result_index] = cluster[i]
                    res_labels.append(found_clusters)
        #        print(res_labels)
                    result_index += 1
                result_centers += [centers[int(c)]]
                found_clusters += 1
                temp_data = np.delete(temp_data, indices, 0)
            if (c < 2):
                c = 2

            continue
        label =  self.getClass(result_labels, res_labels, len(result_centers), labels_names)
        print('labels', len(res_labels))
        print(done)
        print(label)
        print(len(label))
        return label, result_centers, mb
