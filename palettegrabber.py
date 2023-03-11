import numpy as np
import cv2
import matplotlib.pyplot as plt

# Step 1: Load the image
img = cv2.imread("IMG_0704.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 2: Convert the image to a 2D array of RGB values
rgb_img = img.reshape(-1, 3)

# Step 3: Use KMeans clustering to extract dominant colors
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 6
ret, label, center = cv2.kmeans(np.float32(rgb_img), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Step 4: Plot the dominant colors
center = np.uint8(center)
res = center[label.flatten()]
output_image = res.reshape((img.shape))

# Plot the image
plt.imshow(output_image)
plt.axis("off")
plt.show()

# Plot a pie chart
unique, counts = np.unique(label, return_counts=True)
dominant_colors = center[unique].tolist()
percentages = counts / sum(counts) * 100
normalized_colors = [list(map(lambda x: x/255, color)) for color in dominant_colors]
patches, texts = plt.pie(percentages, colors=normalized_colors, startangle=90)
plt.axis("equal")
plt.tight_layout()
plt.show()

# Print a palette list
palette = []
for color in dominant_colors:
    hex_code = "#{:02x}{:02x}{:02x}".format(*map(int, color))
    palette.append((color, hex_code))

print("Palette:")
for i, (color, hex_code) in enumerate(palette):
    print("  Color:", color, " Hex code:", hex_code, " Percentage: {:.2f}%".format(percentages[i]))
