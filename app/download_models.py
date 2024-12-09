import gdown
import os

# Create models path if it doesnt exist yet
if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("models/high-capacity"):
    os.makedirs("models/high-capacity")
if not os.path.exists("models/low-capacity"):
    os.makedirs("models/low-capacity")

# Set Google Drive file ID (extracted from the URL) and path
models_dict = {'1nj8zjdBxX8LK2y4crZS9swEqv3R7UWTo':"models/high-capacity/02691156.keras",
               }

for i, (id, path) in enumerate(models_dict.items()):
    # Download URL format for gdown
    url = f'https://drive.google.com/uc?id={id}'

    # Download the file
    gdown.download(url, path, quiet=False)