import numpy as np

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from brain_delta.brain_delta import Model, BrainDelta

import utils

def main():
    NUM_SIMULATIONS = 10                                               # 20 in paper
    NUM_SUBJECTS = 10000                                              # 20000 in paper
    MEAN_Y, RANGE_Y = 60, 25                                          # 60, 25 in MATLAB code
    NOISE_DELTA = 2                                                   # 2y in paper
    NUM_FEATURES_BASE = 100
    NUM_PCA = 10

    print(f"Running simulations: NUM_SUBJECTS={NUM_SUBJECTS}")
    for sim in range(NUM_SIMULATIONS):
        print(f"Simulation: {sim+1} of {NUM_SIMULATIONS}")

        # "a fairly sharply truncated (non-Gaussian) distribution for the age range (total range approximately 45-75y),"
        Y = np.linspace(MEAN_Y-RANGE_Y/2, MEAN_Y+RANGE_Y/2, NUM_SUBJECTS)
        print("Y", np.mean(Y), np.std(Y))
        DELTA_TRUE = list(np.linspace(-NOISE_DELTA, NOISE_DELTA, 10)) * (NUM_SUBJECTS//10)
        print("D", np.mean(DELTA_TRUE), np.std(DELTA_TRUE))

        Yb = Y + DELTA_TRUE # True brain age
        print("Yb", np.mean(Yb), np.std(Yb))

        X = np.random.normal(size=(NUM_SUBJECTS, NUM_FEATURES_BASE))
        X[:, 0] = Yb
        print("X", np.mean(X), np.std(X))
        #pca = PCA(n_components=NUM_PCA)
        #X = pca.fit_transform(X)
        #X = utils.normalize(X)
        #print("X", np.mean(X), np.std(X))
        b = BrainDelta()
        b.train(Y, X, ev_num=NUM_PCA)
        print("gamma", b.gamma.shape, np.mean(b.gamma), np.std(b.gamma))
        delta3 = b.predict(Y, X, model=Model.ALTERNATE, return_delta=True)
        print("delta3", delta3.shape, np.mean(delta3), np.std(delta3))
        d=b.x_norm-np.dot(b.y_demean[..., np.newaxis], b.gamma)
        print("d", d.shape, np.mean(d), np.std(d))
        delta3=np.dot(d, np.linalg.pinv(b.gamma))
        print("delta3", delta3.shape, np.mean(delta3), np.std(delta3))
        
        #print(np.linalg.pinv(b.y2).shape)
        #print(b.x_reduced.shape)
if __name__ == "__main__":
    main()
