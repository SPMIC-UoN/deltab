"""
Utility functions used in examples
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

def dscatter(ax, x, y, s=1, **kwargs):
    """
    Density-coloured scatter plot
    """
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax.scatter(x, y, c=z, s=s, **kwargs)

def do_plot(ax, x, y, axlim, title="", horiz=False):
    dscatter(ax, x, y)

    if horiz:
        plt.plot([-axlim, axlim], [0, 0])
    else:
        plt.plot([-axlim, axlim], [-axlim, axlim])

    ax.set_xlim(-axlim, axlim)
    ax.set_ylim(-axlim, axlim)
    ax.set_title(title)

def demean(x, axis=0):
    return x - np.nanmean(x, axis=axis)

def normalize(x, axis=0):
    return demean(x, axis=axis) / np.nanstd(x, axis=axis)
