import os
from PIL import Image
import numpy as np

def load_and_serialize(data_dir):
    X = []
    y = []

    class_map = {
        "L": 0,
        "T": 1
    }

    for class_name, label in class_map.items():
        folder_path = os.path.join(data_dir, class_name)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            img = Image.open(file_path).convert("L")
            img = img.resize((20, 20))

            img_array = np.array(img) / 255.0
            img_flat = img_array.flatten().tolist()

            X.append(img_flat)
            y.append(label)

    np.save("X.npy", np.array(X))
    np.save("y.npy", np.array(y))

    print("Saved X.npy and y.npy")

# Run it
load_and_serialize("LTdata")
