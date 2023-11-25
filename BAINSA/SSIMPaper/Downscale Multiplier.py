#%%
import cv2
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt



def process_image(image_path):
    # Initialize an empty DataFrame to store the multipliers and SSIM values
    ssim_data = pd.DataFrame(columns=['Multiplier', 'SSIM'])

    # Read the original image
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for SSIM calculation

    # Define the multipliers from 100 to 0.01 in steps of 0.5
    multipliers = np.arange(100, 1, -0.5)

    for multiplier in multipliers:
        # Downscale multiplier to be between 0 and 1
        downscaler = multiplier / 100.0

        # Step 1: Downscale the image
        height, width = original_image.shape[:2]
        new_height, new_width = int(height * downscaler), int(width * downscaler)
        downsampled_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        # Step 2: Upscale the image back to the original size
        upscaled_image = cv2.resize(downsampled_image, (width, height), interpolation=cv2.INTER_NEAREST)

        # Step 3: Calculate the SSIM between the original image and the upscaled image
        ssim_value = ssim(original_image, upscaled_image)

        # Step 4: Add the multiplier and SSIM value to the DataFrame
        new_row = pd.DataFrame({'Multiplier': [multiplier], 'SSIM': [ssim_value]})
        ssim_data = pd.concat([ssim_data, new_row], ignore_index=True)
        
    ssim_data_reversed = ssim_data[ssim_data.columns[::-1]]

    return ssim_data_reversed

def plot_ssim(df, colour):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Multiplier'], df['SSIM'], linestyle='-', color=colour)
    plt.title('SSIM Values over downscaling')
    plt.xlabel('Final resolution (in %)')
    plt.ylabel('SSIM Value')
    plt.grid(True)
    plt.show()

a = process_image("reference.png")
plot_ssim(a, "blue")

# %%
