"""
3-D visualiser for ./data/out.txt
Columns: x, y, z_pred, z_true
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def main() -> int:
    data = np.loadtxt("./data/out.txt", delimiter=",")
    if data.shape[1] != 4:
        print("Expected 4 columns (x, y, z_pred, z_true).", file=sys.stderr)
        return 1

    x, y, z_pred, z_true = data.T

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot MLP predictions first so they appear on top
    ax.scatter(x, y, z_pred,
               s=12,
               alpha=0.95,
               c='deepskyblue',
               edgecolors='black',
               linewidths=0.3,
               label="MLP prediction")

    # Draw ground truth as a sparse grid of lines
    N = 5  # draw every Nth line in each dimension for clarity
    unique_x = np.unique(x)
    unique_y = np.unique(y)

    for i in range(0, len(unique_x), N):
        xi = unique_x[i]
        mask = x == xi
        sorted_idx = np.argsort(y[mask])
        ax.plot(y[mask][sorted_idx],
                [xi]*np.sum(mask),
                z_true[mask][sorted_idx],
                color='gray', linewidth=0.8)

    for j in range(0, len(unique_y), N):
        yj = unique_y[j]
        mask = y == yj
        sorted_idx = np.argsort(x[mask])
        ax.plot([yj]*np.sum(mask),
                x[mask][sorted_idx],
                z_true[mask][sorted_idx],
                color='gray', linewidth=0.8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.set_title("MLP Approximation vs Ground Truth")

    # Manual legend
    proxy_pred = plt.Line2D([], [], marker='o', color='w',
                            markerfacecolor='deepskyblue', markeredgecolor='black',
                            markersize=8, label="MLP prediction")
    proxy_true = plt.Line2D([], [], color='gray', linestyle='-',
                            label="Ground truth")
    ax.legend(handles=[proxy_pred, proxy_true], loc="best")

    plt.tight_layout()
    plt.show()
    return 0

if __name__ == "__main__":
    sys.exit(main())
