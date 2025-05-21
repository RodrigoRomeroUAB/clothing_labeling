__authors__ = [1630717,1631990, 1632068, 1638180]
__group__ = 12



import numpy as np
import utils

class KMeans:

    def __init__(self, X, K, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options


    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################
        self.N
        self.D

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        X = X.astype("float64")
        if len(X.shape)==3:
            X = X.reshape((X.shape[0]*X.shape[1],3))
        self.X = X
        self.N = X.shape[0]
        self.D = X.shape[1]

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 20
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.
        if 'bss_tol' not in options:
            options['bss_tol'] = 0.1

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def _init_centroids(self):
        """
        Initialization of centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        if self.options['km_init'].lower() == 'first':
            self.centroids = np.array([self.X[0]])
            for x in self.X:
                if not any((self.centroids[:]==x).all(1)) and len(self.centroids)<self.K:
                    self.centroids = np.append(self.centroids,[x],axis=0)
        elif self.options['km_init'].lower() == 'random':
            idx = np.random.randint(self.X.shape[0], size= self.K)
            self.centroids = self.X[idx,:]
            unicos = np.unique(self.centroids,axis=0)
            while len(unicos) != len(self.centroids):
                idx = np.random.randint(self.X.shape[0], size=self.K)
                self.centroids = self.X[idx, :]
                unicos = np.unique(self.centroids,axis=0)
        elif self.options['km_init'].lower() == 'equal_segments':
            h = int(self.X.shape[0]/(self.K))
            idx = np.array([i for i in range(0,self.X.shape[0],h)])
            idx = idx[:self.K]
            self.centroids = self.X[idx]
            unicos, indices = np.unique(self.centroids, axis=0,return_index=True)
            while len(unicos) != len(self.centroids):
                for x in indices:
                    self.centroids[x] = self.centroids[x]-1
                unicos, indices = np.unique(self.centroids, axis=0,return_index=True)
        self.old_centroids = np.random.rand(self.K, self.D)

    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.labels = np.empty((self.N),dtype=int)
        dist_matrix = distance(self.X,self.centroids)
        #if self.options['km_init'] != 'first':
        #    print(dist_matrix)
        self.labels = np.argmin(dist_matrix,axis=1)



    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.old_centroids = np.copy(self.centroids)
        for i in range(self.K):
            idx_matrix = self.X[np.where(self.labels == i)[0]]
            avgrs = np.mean(idx_matrix, axis=0)
            for j in range(3):
                self.centroids[i][j]=avgrs[j]


    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        if (self.centroids == self.old_centroids).all():
            return True
        else:
            return False


    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.num_iter = 0
        self._init_centroids()
        while(self.converges() == False and self.num_iter<=self.options["max_iter"]):
            self.get_labels()
            self.get_centroids()
            self.num_iter+=1

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        wcd = 1 #/ self.X.shape[0]
        for i in range(self.centroids.shape[0]):
            idxs = np.where(self.labels == i)[0]
            idx_matrix = self.X[idxs]
            dist_array = distance(idx_matrix, np.array([self.centroids[i]]))
            wcd = wcd + np.sum(np.array(list(map(lambda x: x ** 2, list(dist_array)))))
        return wcd / self.X.shape[0]

    def bSS_tSS(self):
        glob_centroid = np.mean(self.X,axis=0)
        dist = distance(self.centroids,glob_centroid)
        bss = 0
        for i, row in enumerate(dist):
            bss += (row[0]**2) * (np.where(self.labels == i)[0]).shape[0]
        dist = distance(self.X,glob_centroid)
        tss = 0
        for row in dist:
            tss += row**2
        return bss/tss[0]

    def fisherCoeficient(self):
        glob_centroid = np.mean(self.X, axis=0)
        dist = distance(self.centroids, glob_centroid)
        ssb = 0
        for i in range(self.centroids.shape[0]):
            ssb += np.sum(np.array(list(map(lambda x: x ** 2, list(dist)))))
        nom = ssb/(self.K-1)
        ssw = 0
        for i in range(self.centroids.shape[0]):
            idxs = np.where(self.labels == i)[0]
            idx_matrix = self.X[idxs]
            dist = distance(idx_matrix, np.array([self.centroids[i]]))
            ssw = ssw + np.sum(np.array(list(map(lambda x: x ** 2, list(dist)))))
        denom = ssw/(self.X.shape[0]/self.K)
        return nom/denom



    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        if self.options['fitting'] == 'WCD':
            tol = self.options["tolerance"]
        elif self.options['fitting'] == 'BSS':
            tol = self.options['bss_tol']
        elif self.options['fitting'] == 'FC':
            fcs = np.array([])
            kas = np.array([])
        for k in range(2, max_K + 1):
            self.K = k
            self.fit()
            if self.options['fitting'] == 'WCD':
                heuristics = self.withinClassDistance()
                if k != 2:
                    dec_pctg = 100 * (1 - (heuristics / heuristics_aux))
                    if dec_pctg <= tol:
                        self.K = k - 1
                        break
                heuristics_aux = heuristics
            elif self.options['fitting'] == 'BSS':
                heuristics = self.bSS_tSS()
                if k != 2:
                    t_change = heuristics-heuristics_aux
                    if t_change <= tol:
                        self.K = k
                        break
                heuristics_aux = heuristics
            elif self.options['fitting'] == 'FC':
                heuristics = self.fisherCoeficient()
                fcs = np.append(fcs,heuristics)
                kas = np.append(kas,np.argmax(fcs))
                if len(kas)>=2 and kas[-2] == kas[-1]:
                    self.K = np.argmax(fcs) + 2
                    break
            if self.options['fitting'] == 'FC':
                self.K = np.argmax(fcs)+2

def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    dist =  np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
    return dist


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    color_matrix = utils.get_color_prob(centroids)
    color_list = []
    for c in color_matrix:
        color_list.append(utils.colors[np.argmax(c)])
    return color_list
