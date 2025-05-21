__authors__ = [1630717,1631990, 1632068, 1638180]
__group__ = 12

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist
from scipy.spatial.distance import mahalanobis


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################
        self.P
        self.D

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        train_data = train_data.astype("float64")
        self.P = train_data.shape[0]
        self.D = train_data.shape[1]*train_data.shape[2]
        self.train_data = train_data.reshape(self.P,self.D)

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        test_data = test_data.reshape(test_data.shape[0],test_data.shape[1]*test_data.shape[2])
        dist = cdist(test_data,self.train_data)
        indices_fila = np.argsort(dist, axis=1)[:, :k]
        self.neighbors = self.labels[indices_fila]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        repeated = np.empty(self.neighbors.shape[0],dtype=np.dtype("U16"))
        for i, row in enumerate(self.neighbors):
            values, repetitions = np.unique(row, return_counts=True)
            if len(np.where(repetitions == np.max(repetitions))[0])>1:
                new_values = values[np.where(repetitions == np.max(repetitions))[0]]
                commons, idx = np.unique(row[np.isin(row,new_values)],return_index=True)
                new_values = commons[np.argsort(idx)]
                repeated[i] = new_values[0]
            else:
                repeated[i] = values[(np.argmax(repetitions))]

        return repeated

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """
        self.get_k_neighbours(test_data, k)
        return self.get_class()
