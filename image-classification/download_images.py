import os
import requests
from tqdm import tqdm

TRAIN_URL = "http://images.cocodataset.org/train2017/"
VAL_URL = "http://images.cocodataset.org/val2017/"

TRAIN_DIR = "data/train2017"
VAL_DIR = "data/val2017"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)


def download_images(file_list, base_url, save_dir):

    with open(file_list, "r") as f:
        filenames = [line.strip() for line in f]

    for filename in tqdm(filenames):

        save_path = os.path.join(save_dir, filename)

        # Skip if already downloaded
        if os.path.exists(save_path):
            continue

        url = base_url + filename

        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                with open(save_path, "wb") as img:
                    img.write(response.content)
            else:
                print(f"Failed: {filename}")

        except Exception as e:
            print(f"Error downloading {filename}: {e}")


print("Downloading training images...")
download_images("train_files.txt", TRAIN_URL, TRAIN_DIR)

print("\nDownloading validation images...")
download_images("val_files.txt", VAL_URL, VAL_DIR)

print("\nDownload Complete!")