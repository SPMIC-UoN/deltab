"""
Figure 1 from Smith et al 
"""
import numpy as np
from matplotlib import pyplot as plt

import utils

def main():
    NUM_SUBJS = 20000
    X0=np.random.normal(size=NUM_SUBJS)
    Y0=X0+np.random.normal(size=NUM_SUBJS)*0.3

    utils.do_plot(plt.subplot(2, 3, 1), X0, Y0, axlim=4, title='A.  Gaussian A, B and noise')
    utils.do_plot(plt.subplot(2, 3, 2), X0, Y0+np.random.normal(size=NUM_SUBJS), axlim=4, title='B.  Extra noise on B')
    utils.do_plot(plt.subplot(2, 3, 3), X0[np.abs(X0) < 1], Y0[np.abs(X0) < 1], axlim=2, title='C.  Truncation of A')
    utils.do_plot(plt.subplot(2, 3, 4), X0 - np.mean(X0), Y0 - np.mean(Y0), axlim=4, title='D.  Regularised fit')
    utils.do_plot(plt.subplot(2, 3, 5), X0+np.random.normal(size=NUM_SUBJS), Y0, axlim=4, title='E.  Noise on A (regression dilution)')
    utils.do_plot(plt.subplot(2, 3, 6), X0[np.abs(Y0) < 0.75], Y0[np.abs(Y0) < 0.75], axlim=2, title='F.  Truncation of B')

    plt.show()

if __name__ == "__main__":
    main()
