#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-21 08:47:55 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/NT/clf/prac_SVC_decision_boundaries.py


"""
This script does XYZ.
"""


"""
Imports
"""
import importlib
import logging
import os
import re
import sys
import warnings
from glob import glob
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from icecream import ic
from natsort import natsorted
from tqdm import tqdm

# sys.path = ["."] + sys.path
# from scripts import utils, load

"""
Warnings
"""
# warnings.simplefilter("ignore", UserWarning)


"""
Config
"""
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def fig3d2gif(fig3d, ax, output_path, frames=60, fps=30, elev=10.0):
    def animate(frame):
        ax.view_init(elev=elev, azim=frame)
        return (fig3d,)

    frames = np.linspace(0, 360, frames)
    anim = animation.FuncAnimation(
        fig3d,
        animate,
        frames=frames,
        interval=50,
        blit=True,
        repeat=False,
    )

    # Use progress_callback with tqdm
    with tqdm(total=frames.size, desc="Saving animation") as pbar:
        anim.save(
            output_path,
            writer="pillow",
            fps=fps,
            progress_callback=lambda i, n: pbar.update(1),
        )

    plt.close(fig3d)

    return anim


def main():
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.datasets import make_classification
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import LinearSVC

    # Generate synthetic dataset with 4 classes
    X, Y = make_classification(
        n_samples=100,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Initialize lists to store coefficients and intercepts
    all_weights = []
    all_biases = []

    # Perform 10-repeated 10-fold CV for 10 sessions
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in tqdm(skf.split(X, Y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # Create and train the LinearSVC model
        model = LinearSVC(multi_class="ovr", random_state=42)
        clf = model.fit(X_train, y_train)

        all_weights.append(clf.coef_)
        all_biases.append(clf.intercept_)

    # Aggregate results
    mean_weights = np.mean(all_weights, axis=0)
    mean_biases = np.mean(all_biases, axis=0)

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50)
    )

    # Plot
    # Create the figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot data points and decision boundaries
    colors = ["b", "r", "g", "c"]
    markers = ["o", "s", "^", "D"]

    for i_cls in range(4):
        ax.scatter(
            X[Y == i_cls, 0],
            X[Y == i_cls, 1],
            X[Y == i_cls, 2],
            c=colors[i_cls],
            marker=markers[i_cls],
            label=f"Class {i_cls}",
        )

        # Plot decision boundary for this class
        w = mean_weights[i_cls]
        b = mean_biases[i_cls]
        z = (-w[0] * xx - w[1] * yy - b) / w[2]
        ax.plot_surface(xx, yy, z, alpha=0.3, color=colors[i_cls])

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    ax.set_title("SVM Decision Boundaries for 4 Classes (CV)")
    ax.legend()

    # # Animation function
    # def animate(frame):
    #     ax.view_init(elev=10.0, azim=frame)
    #     return (fig,)

    # # Create animation with tqdm progress bar
    # frames = np.linspace(0, 360, 60)
    # with tqdm(total=len(frames), desc="Creating animation") as pbar:
    #     anim = animation.FuncAnimation(
    #         fig,
    #         animate,
    #         frames=frames,
    #         interval=50,
    #         blit=True,
    #         repeat=False,  # Ensure the animation only runs once
    #     )

    #     # Save animation
    #     anim.save(
    #         "svm_decision_boundaries.gif",
    #         writer="pillow",
    #         fps=30,
    #         progress_callback=lambda i, n: pbar.update(1),
    #     )
    anim = fig3d2gif(fig, ax, "./anim.gif")

    return fig, anim


if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    fig, anim = main()
    # plt.show()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
