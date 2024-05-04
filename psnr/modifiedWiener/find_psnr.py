from PIL import Image
import numpy as np
import cv2  # For denoising functions

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Infinite PSNR means no difference
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Load the original and noisy images
original = np.array(Image.open("naiveKNN_image.png").convert('L'))  # Convert to grayscale
noisy = np.array(Image.open("naiveKNN_noisy_image.png").convert('L'))

# Apply denoising methods
# denoised_median = cv2.medianBlur(noisy, 5)  # Median filter
# denoised_gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)  # Gaussian filter

# Calculate PSNR
psnr_median = calculate_psnr(original, noisy)
# psnr_gaussian = calculate_psnr(original, denoised_gaussian)

print(f"PSNR for Median Filter: {psnr_median}")
# print(f"PSNR for Gaussian Filter: {psnr_gaussian}")
