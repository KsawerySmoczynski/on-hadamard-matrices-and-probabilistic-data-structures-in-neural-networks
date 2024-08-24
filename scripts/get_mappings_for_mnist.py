import json
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

N_CLASSES = 10
ARTIFACTS_DIR = Path("artifacts")
pca_data = "pca_data.csv"
data = pd.read_csv(ARTIFACTS_DIR / pca_data)
data["cum_sum_first_5"] = reduce(lambda x, y: x + y, [data[f"component_{i}"] for i in range(1, 5)], data[f"component_0"])
data = data.sort_values(["label", "cum_sum_first_5"]).reset_index(drop=True)
data.loc[data["first_to_100"] < data["first_to_95"], ["first_to_100"]] = 29

# Get classes complexity
classes_complexity = data.groupby(["label"]).agg({"first_to_95": np.mean})
first_to_95_complexity = classes_complexity.sort_values("first_to_95").index
classes_mapping = {}
for i in range(0, N_CLASSES, 2):
    classes_mapping[first_to_95_complexity[i]] = first_to_95_complexity[i + 1]

# Get indices of imgs
labels_to_indices = {}
for i in range(N_CLASSES):
    indices = data[data["label"] == i]["i"].values.tolist()
    labels_to_indices[i] = indices

min_imgs = min([len(indices) for indices in labels_to_indices.values()])
for i in range(N_CLASSES):
    indices = labels_to_indices[i]
    which_to_take = np.arange(len(indices))
    labels_to_indices[i] = indices[np.random.choice(which_to_take, size=min_imgs, replace=False).sort()]

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
    json.dump(indices_to_indices_mapping_train, f)


MNIST_INDICES_MAPPING_TRAIN_FILE = "mnist_indices_mapping_val_file.json"
with open(ARTIFACTS_DIR / MNIST_INDICES_MAPPING_TRAIN_FILE, "w") as f:
    json.dump(indices_to_indices_mapping_val, f)
