
import numpy as np

class KMeans(object):

    def __init__(self, points, k, init='random', max_iters=10000, rel_tol=1e-5):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria w.r.t relative change of loss
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == 'random':
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):
        """
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """
        center_indices = np.random.choice(self.points.shape[0], self.K, replace= False) #choose random indices
        
        self.centers = self.points[center_indices] #slicing
        
        return self.centers

    def kmpp_init(self):# [3 pts]
        """
            Use the intuition that points further away from each other will probably be better initial centers
        Return:
            self.centers : K x D numpy array, the centers.
        """
        #1. Sample 1% of the points from the dataset, uniformly at random (UAR) and without replacement. This sample will be the dataset the remainder of the algorithm uses to minimize initialization overhead.
        # 2. From the above sample, select only one random point to be the first cluster center.
        # 3. For each point in the sampled dataset, find the nearest currently established cluster center and record the squared distance to get there.
        # 4. Examine all the squared distances and take the point with the maximum squared distance as a new cluster center. In other words, we will choose the next center based on the maximum of the minimum calculated distance instead of sampling randomly like in step 2. You may break ties arbitrarily.
        # 5. Repeat 3-4 until all k-centers have been assigned. You may use a loop over K to keep track of the data in each cluster. 
        
        center_indices = np.random.choice(self.points.shape[0], int(0.01 * self.points.shape[0]), replace= False) #choose random indices
        
        sample_dataset = self.points[center_indices]
        
        cluster_center_index = np.random.choice(center_indices, 1, replace= False)
                
        k_centers = self.points[cluster_center_index]
        
        for i in range(0, self.K - 1):
            dist_matrix  = pairwise_dist(sample_dataset, k_centers)
            dist_matrix = np.min(dist_matrix, axis = 1)
            max_dist_index = np.argmax(dist_matrix)
            k_centers = np.vstack([k_centers, sample_dataset[max_dist_index]])
        
        self.centers = k_centers
        return self.centers
        

    def update_assignment(self):
        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: You could call pairwise_dist() function
        """
        
        dist_matrix = pairwise_dist(self.points, self.centers)
        self.assignments = np.argmin(dist_matrix, axis = 1)   
        return self.assignments     


    def update_centers(self):
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        """
        
        for i in range(self.K):
            if self.points[self.assignments == i].size == 0:
                continue
            self.centers[i] = np.mean(self.points[self.assignments == i], axis= 0)
        
        return self.centers

    def get_loss(self):
        """
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """
        self.loss = np.sum(np.square((self.points - self.centers[self.assignments])))
        return self.loss

    def train(self): 
        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster, 
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned, 
                     pick a random point in the dataset to be the new center and 
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference 
                     in loss compared to the previous iteration is less than the given 
                     relative tolerance threshold (self.rel_tol). 
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!
                
        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.
        """
        loss_prev_iter = pow(2, 60)
        for i in range(0, self.max_iters):
            self.update_assignment()
            
            self.update_centers()
            
            #account for cluster center with no points
            for j in range(self.K):
                if self.points[self.assignments == j].size == 0:
                    self.centers[i] = self.points[np.random.randint(0, self.points.size)]
                
            
            loss_curr = self.get_loss()
            if (i == 0):
                loss_prev_iter = loss_curr
                continue

            percent_diff = np.abs(loss_curr - loss_prev_iter) / loss_prev_iter
            loss_prev_iter = loss_curr
            if percent_diff < self.rel_tol:
                break
        
        return self.centers, self.assignments, self.loss
            

def pairwise_dist(x, y):
        np.random.seed(1)
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
        """
 
        #construct x's dot product matrix:
        n = x.shape[0]
        m = y.shape[0]
        
        #construct x's dot product matrix:
        x_dot = np.sum(x*x , axis = 1, keepdims= True)  # shape of n, 1
        y_dot = np.sum(y*y, axis = 1, keepdims= True) # shape of m, 1
        
        #construct 2ab matrix
        xy_dot = x.dot(y.T)
        dist = np.sqrt(np.abs(x_dot + y_dot.T - (2*xy_dot)))
        
        return dist
