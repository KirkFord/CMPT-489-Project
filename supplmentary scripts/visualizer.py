import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
from tqdm import tqdm

directory = 'C:/Users/kirk/Desktop/artbench-10-imagefolder-split/train/'

# Define the paths to the image folders
folder_paths = [os.path.join(directory, folder) for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

# Initialize empty lists for images and labels
images = []
labels = []


# Create a dictionary to map folder names to integer labels
folder_labels = {'art_nouveau': 0, 'baroque': 1, 'expressionism': 2, 'impressionism': 3,
                 'post_impressionism': 4, 'realism': 5, 'renaissance': 6, 'romanticism': 7,
                 'surrealism': 8, 'ukiyo_e': 9}

# Loop through each folder and load the images
for folder_path in folder_paths:
    # Get the label for this folder
    folder_name = os.path.basename(folder_path)
    label = folder_labels[folder_name]
    for filename in tqdm(os.listdir(folder_path), desc=f'Loading images from {folder_name}', unit='image'):
        # Load the image as a numpy array
        img = np.array(Image.open(os.path.join(folder_path, filename)))
        # Add the image to the list
        images.append(img)
        # Add the label to the list
        labels.append(label)




# Convert the lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Reshape the images to 1D arrays
images = images.reshape((images.shape[0], -1))

# Initialize t-SNE with default parameters
tsne = TSNE()

# Fit and transform the data
embeddings = tsne.fit_transform(images)

# Plot the embeddings with different colors for each cluster
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels)
plt.show()
