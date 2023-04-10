import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import threading
import warnings
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore")
# Define the path to the folder containing the image folders
folder_path = 'C:/Users/kirk/Desktop/artbench-10-imagefolder-split/test/'

# Define a function to process an image and return its dominant colors
def process_image(file_path):
    # Open the image and convert it to a numpy array
    image = Image.open(file_path)
    image_array = np.array(image)
    # Reshape the array to be a list of pixels
    pixel_list = image_array.reshape(-1, image_array.shape[-1])
    # Calculate the dominant colors using k-means clustering with k=5
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=0).fit(pixel_list)
    # Get the RGB values of the most dominant color
    most_common_color = kmeans.cluster_centers_[np.argmax(np.unique(kmeans.labels_, return_counts=True)[1])]
    return most_common_color

# Define a function to process a folder of images using threads
def process_folder(folder_name, results):
    # Create a list to store the dominant colors for each image in the folder
    dominant_colors = []
    # Loop through each file in the folder and append the dominant colors to the list
    file_paths = [folder_path + folder_name + '/' + file_name for file_name in os.listdir(folder_path + folder_name) if file_name.endswith('.jpg')]
    for file_path in tqdm(file_paths, desc=folder_name):
        color = process_image(file_path)
        dominant_colors.append(color)
    # If any dominant colors were found, concatenate them into a single array
    if len(dominant_colors) > 0:
        dominant_colors = np.array(dominant_colors)
    results.append((folder_name, dominant_colors))


# Process each folder using threads
results = []
threads = []
for folder_name in os.listdir(folder_path):
    if not os.path.isdir(folder_path + folder_name):
        continue
    thread = threading.Thread(target=process_folder, args=(folder_name, results))
    thread.start()
    threads.append(thread)

# Wait for all the threads to finish
for thread in threads:
    thread.join()

# Create a 3D scatter plot of the dominant colors for each folder
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for folder_name, dominant_colors in results:
    if len(dominant_colors) > 0:
        # Extract the R, G, and B values from the dominant colors array
        R = dominant_colors[:, 0]
        G = dominant_colors[:, 1]
        B = dominant_colors[:, 2]
        ax.scatter(R, G, B, label=folder_name)
# Add labels and a legend to the plot
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.legend()

# Set up the animation
def rotate(angle):
    ax.view_init(azim=angle)
    return fig,

angles = np.linspace(0, 360, 360)
rot_animation = animation.FuncAnimation(fig, rotate, frames=angles, interval=50)

# Save the animation as a mp4 video
rot_animation.save('3d_scatter_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
