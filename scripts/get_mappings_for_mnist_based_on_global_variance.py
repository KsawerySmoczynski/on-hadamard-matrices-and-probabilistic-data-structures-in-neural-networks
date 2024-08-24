import json
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

N_CLASSES = 10
ARTIFACTS_DIR = Path("artifacts")
pca_data = "pca_global_data.csv"
data = pd.read_csv(ARTIFACTS_DIR / pca_data)

# Get classes complexity
classes_complexity = data.groupby(["label"]).agg({"rmse": np.mean})
rmse_complexity = classes_complexity.sort_values("rmse").index
classes_mapping = {}
for i in range(0, N_CLASSES, 2):
    classes_mapping[rmse_complexity[i]] = rmse_complexity[i + 1]

# Get indices of imgs
labels_to_indices = {}
for i in range(N_CLASSES):
    indices = data[data["label"] == i]["i"].values
    labels_to_indices[i] = indices

min_imgs = min([len(indices) for indices in labels_to_indices.values()])
for i in range(N_CLASSES):
    indices = labels_to_indices[i]
    which_to_take = np.sort(np.random.choice(np.arange(len(indices)), size=min_imgs, replace=False))
    labels_to_indices[i] = indices[which_to_take].tolist()

labels_to_indices_val = {i: labels_to_indices[i][::5] for i in range(N_CLASSES)}
labels_to_indices_train = {i: [index for index in labels_to_indices[i] if index not in labels_to_indices_val[i]] for i in range(N_CLASSES)}

# Val set
indices_to_indices_mapping_val = {}
for label, mapped_label in classes_mapping.items():
    for index, mapped_index in zip(labels_to_indices_val[label][::2], labels_to_indices_val[mapped_label][::2]):
        indices_to_indices_mapping_val[index] = mapped_index
    for mapped_index, index in zip(labels_to_indices_val[mapped_label][1::2], labels_to_indices_val[label][1::2]):
        indices_to_indices_mapping_val[mapped_index] = index


# Train set
indices_to_indices_mapping_train = {}
for label, mapped_label in classes_mapping.items():
    for index, mapped_index in zip(labels_to_indices_train[label][::2], labels_to_indices_train[mapped_label][::2]):
        indices_to_indices_mapping_train[index] = mapped_index
    for mapped_index, index in zip(labels_to_indices_train[mapped_label][1::2], labels_to_indices_train[label][1::2]):
        indices_to_indices_mapping_train[mapped_index] = index

MNIST_INDICES_MAPPING_TRAIN_FILE = "mnist_indices_mapping_train_file.json"
with open(ARTIFACTS_DIR / MNIST_INDICES_MAPPING_TRAIN_FILE, "w") as f:
    json.dump(indices_to_indices_mapping_train, f, sort_keys=True)


MNIST_INDICES_MAPPING_TRAIN_FILE = "mnist_indices_mapping_val_file.json"
with open(ARTIFACTS_DIR / MNIST_INDICES_MAPPING_TRAIN_FILE, "w") as f:
    json.dump(indices_to_indices_mapping_val, f, sort_keys=True)
