# Project Overview
This project focuses on evaluating image super-resolution algorithms using the Structural Similarity Index (SSIM). The objective is to analyze and demonstrate how SSIM values can aid in understanding the performance of various super-resolution techniques.

# Project Structure
1. **Baseline Recursive Image Super-Resolution**
   - Script: `Baseline Recursive.py`
   - Description: This script implements a baseline recursive image super-resolution algorithm. It iteratively downscales an image and then upscales it back to the original size, evaluating SSIM values at each iteration.

2. **Multiplier SSIM Evaluation**
   - Script: `Downscale Multiplier.py`
   - Description: This script evaluates SSIM values at various downscaling multipliers for image super-resolution. It computes SSIM values between the original image and the upscaled version for different multipliers.

3. **Iterative Image Super-Resolution**
   - Script: `Downscale Upscale.py`
   - Description: This script implements an iterative image super-resolution algorithm. It downscales an image by a specified downscaler factor and iteratively upscales it back to the original size, evaluating SSIM values at each iteration.

4. **Mean Squared Error Calculation**
   - Script: `MSE.py`
   - Description: This script calculates the Mean Squared Error (MSE) between two images, providing a quantitative measure of error in image processing tasks.

5. **Remini Image Super-Resolution**
   - Script: `Remini Recursive Downscale.py`
   - Description: This script utilizes the Remini AI API for image super-resolution. It sends downscaled images to the Remini API for enhancement and retrieves the upscaled images for evaluation.

6. **SSIM Data**
   - File: `ssim_data.csv`
   - Description: CSV file containing SSIM values at multiple iterations for evaluating image super-resolution algorithms.

# Libraries Used
  - `cv2`: OpenCV library for image processing tasks.
  - `numpy`: NumPy library for numerical computations and array manipulation.
  - `pandas`: Pandas library for data manipulation and analysis.
  - `scikit-image (skimage)`: Library for image processing algorithms and utilities.
  - `httpx`: HTTP client for sending requests to the Remini AI API.
  - `matplotlib`: Library for creating visualizations in Python.

This project demonstrates how SSIM values can be used to evaluate and compare different image super-resolution techniques. By analyzing SSIM data, researchers can gain insights into the effectiveness of these algorithms in enhancing image quality.
