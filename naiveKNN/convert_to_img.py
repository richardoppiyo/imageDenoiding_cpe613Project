import numpy as np
import matplotlib.pyplot as plt

# Load grayscale values from the text file and replace bad values with 0
try:
    grayscale_channel = np.genfromtxt('blurred_values.txt', dtype=int, delimiter=",", filling_values=0)
except ValueError as e:
    print(f"Error loading grayscale values: {e}")
    # Handle the error by setting bad values to 0
    grayscale_channel = np.zeros((600 * 600,), dtype=int)

# Reshape the grayscale values into a 2D array (handling size mismatch)
height, width = 600, 600
gray_image = np.resize(grayscale_channel, (height, width))

# Display the grayscale image
plt.imshow(gray_image, cmap='gray')
plt.title('Blurred Image')
plt.show()
