#%%
import base64
import hashlib
import httpx
import numpy as np
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from time import sleep
import matplotlib.pyplot as plt

API_KEY = "_Uqkw4yeI3cuJWGSnIpVi59kVFQk2-oqM2Wr66eNNB5FZ_Lj"
_TIMEOUT = 60
_BASE_URL = "https://developer.remini.ai/api"
CONTENT_TYPE = "image/png"
OUTPUT_CONTENT_TYPE = "image/png"

def _get_image_md5_content(image_path):
    with open(image_path, "rb") as fp:
        content = fp.read()
        image_md5 = base64.b64encode(hashlib.md5(content).digest()).decode("utf-8")
    return image_md5, content

def remini_image(image_path, downscaling, iterations, df = None, last_image = None, iteration = 0):
    if df is None:
        ssim_data = pd.DataFrame(columns=['Iteration', 'SSIM'])
    else:
        ssim_data = df
    # Read the original image
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for SSIM calculation

    # Define the current image that will be processed in each iteration
    if last_image is None:
        current_image = original_image.copy()
    else:
        last_image_pic = cv2.imread(last_image, cv2.IMREAD_COLOR)
        last_image_pic = cv2.cvtColor(last_image_pic, cv2.COLOR_BGR2GRAY)
        current_image = last_image_pic.copy()

    # Setup an HTTP client with the correct options
    with httpx.Client(
            base_url=_BASE_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
    ) as client:
        for i in range(iterations):
            height, width = current_image.shape[:2]
            new_height, new_width = int(height * downscaling), int(width * downscaling)
            downsampled_image = cv2.resize(current_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            temp_path = 'temp_downsampled.png'
            cv2.imwrite(temp_path, downsampled_image)
            image_md5, content = _get_image_md5_content(temp_path)
            response = client.post(
                "/tasks",
                json={
                    "tools": [
                        {"type": "face_enhance",
                         "mode": "beautify"},
                        {"type": "background_enhance",
                         "mode": "base"}
                    ],
                    "image_md5": image_md5,
                    "image_content_type": CONTENT_TYPE,
                    "output_content_type": OUTPUT_CONTENT_TYPE}
            )
            assert response.status_code == 200
            body = response.json()
            task_id = body["task_id"]
            response = httpx.put(
                body["upload_url"], headers=body["upload_headers"],
                content=content, timeout=_TIMEOUT
            )
            assert response.status_code == 200
            response = client.post(f"/tasks/{task_id}/process")
            assert response.status_code == 202
            for j in range(50):
                response = client.get(f"/tasks/{task_id}")
                assert response.status_code == 200
                if response.json()["status"] == "completed":
                    break
                else:
                    sleep(10)

            upscaled_image_url = response.json()["result"]["output_url"]
            response = httpx.get(upscaled_image_url)
            image_stream = np.asarray(bytearray(response.content), dtype=np.uint8)
            upscaled_image = cv2.imdecode(image_stream, cv2.IMREAD_COLOR)
            upscaled_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
            if upscaled_image.shape != original_image.shape:
                upscaled_image = cv2.resize(upscaled_image, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_LINEAR)
            ssim_value = ssim(original_image, upscaled_image)
            new_row = pd.DataFrame({'Iteration': [iteration + i + 1], 'SSIM': [ssim_value]})
            ssim_data = pd.concat([ssim_data, new_row], ignore_index=True)
            current_image = upscaled_image.copy()
    ssim_data.to_csv('ssim_data.csv', index=False)
    cv2.imwrite('last_current_image.png', current_image)    
    return ssim_data

def plot_ssim(df, multiplier):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Iteration'], df['SSIM'], marker='o', linestyle='-', color='b')
    plt.title('SSIM for Downscaling Multiplier: ' + str(multiplier) + '\nusing Remini')
    plt.xlabel('Iteration')
    plt.ylabel('SSIM Value')
    plt.grid(True)
    plt.show()

multiplier = 0.2
dataframe = pd.read_csv('ssim_data.csv')
a = remini_image("reference.png", multiplier, 4, dataframe, 'last_current_image.png', 96)
plot_ssim(a, multiplier)
    # %%
