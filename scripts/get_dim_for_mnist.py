from pathlib import Path
import pandas as pd
from torchvision.datasets import MNIST
from torch.utils.data import ConcatDataset
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from tqdm import tqdm

# https://dataknowsall.com/blog/imagepca.html

def variance_explained_by_first_5_components(x):
    x = np.array(x)
    pca = PCA()
    pca.fit(x)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    first_to_95 = np.argmax(cumsum*100 >= 95)
    first_to_100 = np.argmax(cumsum*100 >= 100)
    return pca.explained_variance_ratio_.tolist(), first_to_95, first_to_100
    


mnist_train = MNIST("datasets",
            train=True,
            download=True,
            transform=variance_explained_by_first_5_components
        )

mnist_test = MNIST("datasets",
            train=False,
            download=True,
            transform=variance_explained_by_first_5_components
        )

mnist = ConcatDataset([mnist_train, mnist_test])

dataset_info = []
for i in tqdm(range(len(mnist)), mininterval=1):
    pca_info, label = mnist[i]
    explained_variance_ratio, first_to_95, first_to_100 = pca_info
    dataset_info.append((i, label, *explained_variance_ratio, first_to_95, first_to_100))

columns = ["i", "label", *[f"component_{i}" for i in range(28)], "first_to_95", "first_to_100"]
ARTIFACTS_DIR = Path("artifacts")
pca_data = "pca_data.csv"
pd.DataFrame(dataset_info, columns=columns).to_csv(ARTIFACTS_DIR / pca_data, index=False)

