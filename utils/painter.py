import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from utils.utils import guarantee_numpy


class Painter:
    def __init__(self, x, y):
        self.x = guarantee_numpy(x)
        self.y = guarantee_numpy(y)

    def t_sne(self, n_components, seed=42, title=None):
        if self.x.ndim <= 2:
            latent_vectors = TSNE(n_components=n_components, random_state=seed).fit_transform(self.x)
        else:
            latent_vectors = TSNE(n_components=n_components, random_state=seed).fit_transform(
                self.x.reshape(len(self.x), -1))
        self.scatter(x=latent_vectors[:, 0], y=latent_vectors[:, 1], labels=self.y, title=title)

    @staticmethod
    def scatter(x, y, labels=None, palette="muted", title=None):
        df = pd.DataFrame(dict(x=x, y=y, labels=labels))
        sns.scatterplot(x='x', y='y', data=df, hue='labels', palette=palette)
        if title is not None:
            plt.title(title)
        plt.show()