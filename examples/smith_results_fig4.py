"""
Simulation 1 from Smith et al 
"""
import numpy as np
from matplotlib import pyplot as plt

from deltab.deltab import BrainDelta, Model

import utils

def main():
    # For Simulation 1, we set the number of samples (subjects) to 20000,
    N = 20000

    # set a fairly sharply truncated (non-Gaussian) distribution for the 
    # age range (total range approximately 45-75y)
    Y = np.random.rand(N) - 0.5 + 0.01*np.random.normal(size=N)
    Y = 60+Y*25

    # and added Gaussian δ 
    # with standard deviation 2y to form the gold-standard brain age. We 
    DELTA_TRUE = 2 * np.random.normal(size=N)

    # add quadratic aging effects. 
    Yquad = 0.05
    Y_DEMEAN = utils.demean(Y)
    Y2=utils.demean(Yquad*Y_DEMEAN**2)
    Y2 = Y2-Y_DEMEAN * (np.dot(np.linalg.pinv(Y_DEMEAN[..., np.newaxis]), Y2))
    BRAIN_AGE_TRUE = Y + DELTA_TRUE + Y2

    # then defined 100 underlying components (processes) of subject variation 
    # in “brain imaging” measures, the first being brain age, and the other 
    # 99 being random.
    #
    # NB Matlab code defines 5 features not 100
    NUM_FEATURES = 5
    X = np.random.normal(size=(N, NUM_FEATURES))
    X[:, 0] = BRAIN_AGE_TRUE
    X = utils.normalize(X)

    # We then mixed these ground truth population modes by 
    # a (100x3000) sparse mixing matrix (random Gaussian noise to the fifth power) 
    # to form 3000 imaging variables, resulting in an X of size 20000x3000.
    #
    # NB Matlab code defines mixing matrix of size 100 not 3000
    NUM_MIXING = 100
    X = utils.normalize(np.dot(X, np.random.normal(size=(NUM_FEATURES, NUM_MIXING))**5))

    # Finally, we standardised all columns in X to 1, and added measurement 
    # noise with a standarddeviation of 0.5. (Reducing this noise to the kinder 
    # level of 0.1 does not make a large qualitative difference to the results.) 
    #
    # NB Matlab code defines noise std.dev of 0.2 not 0.5
    X = utils.demean(X + 0.2 * np.random.normal(size=X.shape))
    Y = utils.demean(Y)

    b = BrainDelta()
    b.train(Y, X)
    y1 = b.predict(Y, X, model=Model.SIMPLE, return_delta=False)
    d1 = b.predict(Y, X, model=Model.SIMPLE, return_delta=True)
    y2sq = b.predict(Y, X, model=Model.UNBIASED_QUADRATIC, return_delta=False)
    d2sq = b.predict(Y, X, model=Model.UNBIASED_QUADRATIC, return_delta=True)

    utils.do_plot(plt.subplot(2, 3, 1), Y, y1, axlim=20, title='A.\rm  Predicted age \itY_{B1}\rm vs. age \itY')
    utils.do_plot(plt.subplot(2, 3, 2), Y, d1, axlim=15, horiz=True, title='A.\rm  Predicted age \itY_{B1}\rm vs. age \itY')
    utils.do_plot(plt.subplot(2, 3, 3), DELTA_TRUE, d1, axlim=10, title='A.\rm  Predicted age \itY_{B1}\rm vs. age \itY')

    utils.do_plot(plt.subplot(2, 3, 4), Y, y2sq, axlim=20, title='A.\rm  Predicted age \itY_{B1}\rm vs. age \itY')
    utils.do_plot(plt.subplot(2, 3, 5), Y, d2sq, axlim=15, horiz=True, title='A.\rm  Predicted age \itY_{B1}\rm vs. age \itY')
    utils.do_plot(plt.subplot(2, 3, 6), DELTA_TRUE, d2sq, axlim=10, title='A.\rm  Predicted age \itY_{B1}\rm vs. age \itY')

    plt.show()

if __name__ == "__main__":
    main()
