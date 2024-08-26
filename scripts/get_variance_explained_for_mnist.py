import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from torch.utils.data import ConcatDataset
from torchvision.datasets import MNIST
from tqdm import tqdm

# https://dataknowsall.com/blog/imagepca.html


mnist_train = MNIST("datasets", train=True, download=True, transform=np.array)

mnist_test = MNIST("datasets", train=False, download=True, transform=np.array)

mnist = ConcatDataset([mnist_train, mnist_test])

# TODO run pca across class and then try to reproduce image by image, and then calculate MSE from first 5 components reproduction

classes = {i: [] for i in range(10)}
for i in tqdm(range(len(mnist)), mininterval=1):
    img, label = mnist[i]
    classes[label].append(i)

dataset_info = []
components_info = {}
for cls, indices in classes.items():
    cls_imgs = np.stack([mnist[i][0].flatten() for i in indices])
    scaler = StandardScaler()
    cls_imgs_standarized = scaler.fit_transform(cls_imgs)
    pca = PCA()
    pca.fit(cls_imgs_standarized)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    first_to_95 = np.argmax(cumsum * 100 >= 95)
    pca = PCA(n_components=first_to_95)
    cls_imgs_reconstructed = pca.inverse_transform(pca.fit_transform(cls_imgs_standarized))
    cls_imgs_reconstructed = scaler.inverse_transform(cls_imgs_reconstructed)
    rmse = np.sqrt(((cls_imgs - cls_imgs_reconstructed) ** 2).sum(1))
    for index, rmse_sample in zip(indices, rmse):
        dataset_info.append((cls, index, rmse_sample))
    components_info[cls] = int(first_to_95)


columns = ["label", "i", "rmse"]
ARTIFACTS_DIR = Path("artifacts")
pca_data = "pca_global_data.csv"
df = pd.DataFrame(dataset_info, columns=columns).sort_values(by=["label", "rmse"])
df.to_csv(ARTIFACTS_DIR / pca_data, index=False)

components_info_file = "components_info.json"
with open(ARTIFACTS_DIR / components_info_file, "w") as f:
    json.dump(components_info, f)
