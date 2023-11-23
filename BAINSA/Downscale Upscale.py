#%%
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def process_image(image_path, downscaler, iterations):
    # Initialize an empty DataFrame to store iteration numbers and SSIM values
    ssim_data = pd.DataFrame(columns=['Iteration', 'SSIM'])

    # Read the original image
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for SSIM calculation

    # Define the current image that will be processed in each iteration
    current_image = original_image.copy()

    for i in range(iterations):
        # Step 1: Downscale the image
        height, width = current_image.shape[:2]
        new_height, new_width = int(height * downscaler), int(width * downscaler)
        downsampled_image = cv2.resize(current_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Step 2: Upscale the image back to original size
        upscaled_image = cv2.resize(downsampled_image, (width, height), interpolation=cv2.INTER_LINEAR)

        # Step 3: Calculate the SSIM between the original image and upscaled image
        ssim_value = ssim(original_image, upscaled_image)
        new_row = pd.DataFrame({'Iteration': [i + 1], 'SSIM': [ssim_value]})
        ssim_data = pd.concat([ssim_data, new_row], ignore_index=True)
        
        # Step 5: Set the upscaled image as the new "original" for the next iteration
        current_image = upscaled_image.copy()

    return ssim_data


def plot_ssim(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Iteration'], df['SSIM'], marker='o', linestyle='-', color='b')
    plt.title('SSIM Values over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('SSIM Value')
    plt.grid(True)
    plt.show()

trial = process_image("reference.png", 0.5, 2)
plot_ssim(trial)
# %%
