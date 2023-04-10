import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def extract_dominant_colors(img):
    # Convert the image to a 2D array of RGB values
    rgb_img = img.reshape(-1, 3)

    # Use KMeans clustering to extract dominant colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 5
    ret, label, center = cv2.kmeans(np.float32(rgb_img), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Get the dominant colors and their percentages
    unique, counts = np.unique(label, return_counts=True)
    dominant_colors = center[unique].tolist()
    percentages = counts / sum(counts) * 100

    # Create a palette list with the dominant colors and their hex codes
    palette = []
    for color in dominant_colors:
        hex_code = "#{:02x}{:02x}{:02x}".format(*map(int, color))
        palette.append((color, hex_code))

    return palette, percentages

# Get a list of all image files in the current directory
images = [f for f in os.listdir() if f.endswith(".jpg")]

# Extract the dominant colors from each image
output = []
for image_file in images:
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    palette, percentages = extract_dominant_colors(img)

    output.append((image_file, palette, percentages))

# Write the output to a file
with open("colour_output.txt", "w") as f:
    for image_file, palette, percentages in output:
        f.write("Image: " + image_file + "\n")
        f.write("Palette:\n")
        for i, (color, hex_code) in enumerate(palette):
            f.write("  Color: " + str(color) + " Hex code: " + hex_code + " Percentage: {:.2f}%\n".format(percentages[i]))
        f.write("\n")
