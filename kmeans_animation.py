import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate random data
np.random.seed(0)
num_points = 100
num_clusters = 3

data_points = np.random.randn(num_points, 2)
centroids = np.random.randn(num_clusters, 2)

# Initialize figure and axes
fig, ax = plt.subplots()
colors = ['r', 'g', 'b']  # Color for each cluster

# Initialization function
def init():
    ax.scatter(data_points[:, 0], data_points[:, 1], c='black')
    ax.scatter(centroids[:, 0], centroids[:, 1], c=colors, marker='x')
    ax.set_title('K-Means Clustering')
    return ax

# Update function for each frame
def update(frame):
    # Perform K-Means iteration
    distances = np.sqrt(np.sum((data_points - centroids[:, np.newaxis]) ** 2, axis=2))
    labels = np.argmin(distances, axis=0)
    
    # Update centroids
    for i in range(num_clusters):
        centroids[i] = np.mean(data_points[labels == i], axis=0)
    
    # Clear previous plot
    ax.cla()
    
    # Plot data points, centroids, and connecting lines
    for i in range(num_clusters):
        cluster_points = data_points[labels == i]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i])
        ax.scatter(centroids[:, 0], centroids[:, 1], c=colors, marker='x')
        for point in cluster_points:
            ax.plot([point[0], centroids[i, 0]], [point[1], centroids[i, 1]], c='gray', linestyle='dotted')
    
    ax.set_title('K-Means Clustering (Iteration {}/{})'.format(frame + 1, num_iterations))
    
# Number of iterations
num_iterations = 10

# Create the animation
animation = FuncAnimation(fig, update, frames=num_iterations, init_func=init, blit=False)

# Set the save path and save the animation
save_path = 'kmeans_animation.gif'
animation.save(save_path, writer='pillow', fps=2)

plt.show()