#%%
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import matplotlib.pyplot as plt

def super_resolve_image(image_path, downscaling, iterations):
    ssim_data = pd.DataFrame(columns=['Iteration', 'SSIM'])

    # Read the original image
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for SSIM calculation

    # Define the current image that will be processed in each iteration
    current_image = original_image.copy()

    for i in range(iterations):
        height, width = current_image.shape[:2]
        new_height, new_width = int(height * downscaling), int(width * downscaling)
        downsampled_image = cv2.resize(current_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        upscaled_image = cv2.resize(downsampled_image, (width, height), interpolation=cv2.INTER_NEAREST)
        ssim_value = ssim(original_image, upscaled_image)
        new_row = pd.DataFrame({'Iteration': [i + 1], 'SSIM': [ssim_value]})
        ssim_data = pd.concat([ssim_data, new_row], ignore_index=True)
        current_image = upscaled_image.copy()

    return ssim_data

def plot_ssim(df, multiplier):
    plt.figure(figsize=(20, 12))
    plt.plot(df['Iteration'], df['SSIM'], marker='o', linestyle='-', color='b')
    plt.title('SSIM with Downscaling Multiplier: ' + str(multiplier) + '\nusing Nearest Interpolation')
    plt.xlabel('Iteration')
    plt.ylabel('SSIM Value')
    plt.grid(True)
    plt.show()

multiplier = 0.2
a = super_resolve_image("reference.png", multiplier, 100)
plot_ssim(a, multiplier)

# %%
