"""
Table 1 from Smith et al 
"""
import numpy as np
from matplotlib import pyplot as plt
import sys
import pandas as pd

from brain_delta.brain_delta import Model, BrainDelta

import utils

def DeltaCV(X,Y,J,CONF=None):
    # for slightly improved results, run this many times and average the outputs
    delta1 = np.zeros_like(Y)
    delta2 = np.zeros_like(Y)
    delta3 = np.zeros_like(Y)
    delta2q = np.zeros_like(Y)
    delta3q = np.zeros_like(Y)

    Ncv=10 # number of cross-validation folds
    I = np.random.randint(Ncv, size=len(Y))  # create random cross-validation folds

    if J == 0:
        J = X.shape[1]

    b = BrainDelta()
    for i in range(Ncv):
        OUT, IN = (I==i), np.not_equal(I, i)
        b.train(Y[IN], X[IN,:], ev_num=J)
        
        x_out, y_out = X[OUT,:], Y[OUT]
        delta1[OUT] = b.predict(y_out, x_out, model=Model.SIMPLE, return_delta=True)
        delta2[OUT] = b.predict(y_out, x_out, model=Model.UNBIASED, return_delta=True)
        delta2q[OUT] = b.predict(y_out, x_out, model=Model.UNBIASED_QUADRATIC, return_delta=True)
        delta3[OUT] = b.predict(y_out, x_out, model=Model.ALTERNATE, return_delta=True)
        delta3q[OUT] = b.predict(y_out, x_out, model=Model.ALTERNATE_QUADRATIC, return_delta=True)

    return delta1, delta2, delta3, delta2q, delta3q

def corr(X, *Y):
    return np.corrcoef(np.stack([X] + list(Y), axis=1),rowvar=False)[0, 1:]

def main():

    # linear simulations (Table 1)
    Nsimulations=10
    #Js = [0, 1, 10, 50, 100, 1000, 2990]
    Js = [0, 1, 10, 20]
    N=1000 # N=10000 in paper
    AllResults = np.zeros((len(Js), 11, Nsimulations))
    for sim in range(Nsimulations):
        j=range(sim)
        Y = np.random.rand(N) - 0.5 + 0.02*np.random.normal(size=N)
        Y = 60 + Y*25

        deltaTRUE=2*np.random.normal(size=N)
        minY, maxY = np.min(Y), np.max(Y)
        Y0 = (Y-minY) / (maxY-minY)

        NOISE, ND = 0.5, 100 # SIM 1
        #NOISE, ND = 10.0, 1 # SIM 2
        #NOISE, ND = 0.5, 100
        #deltaTRUE = deltaTRUE*(1 + 0.5*Y0) # SIM 3 non-additive delta

        Yb = Y + deltaTRUE
        X0 = np.random.normal(size=(N, ND))
        X0[:, 0] = Yb
        X0 = utils.normalize(X0)
        Xmix = np.random.normal(size=(ND, 300))**5 # 3000 in paper
        X1 = np.dot(X0, Xmix)
        X2 = utils.normalize(X1)
        X3 = utils.demean(X2+NOISE*np.random.normal(size=X2.shape))
        for j_idx, J in enumerate(Js): # permuting pcaU gives same results as J=1 (ie null model ~ bad model)
            print(J)
            delta1,delta2,delta3,delta2q,delta3q = DeltaCV(X3,Y,J)
            page1, page2, page3 = Y+delta1, Y+delta2, Y+delta3
            agecorr = corr(Y, delta1, page1, page2, page3)
            corrs = corr(deltaTRUE, delta1, delta2, delta3)
            means = np.mean(np.abs([delta1, delta2, delta3]), axis=1)
            AllResults[j_idx, :, sim] = np.concatenate([np.atleast_1d(J), agecorr, means, corrs])
            
    mean_results = np.concatenate([np.mean(AllResults, axis=2), np.std(AllResults, axis=2)], axis=1)
    df = pd.DataFrame(
        mean_results, 
        columns=[
            "J",
            "mean_corr_agedelta1",
            "mean_corr_age1", "mean_corr_age2", "mean_corr_age3",
            "mean_delta1", "mean_delta2", "mean_delta3",
            "mean_corr_delta1", "mean_corr_delta2", "mean_corr_delta3",
            "std_J",
            "std_corr_agedelta1",
            "std_corr_age1", "std_corr_age2", "std_corr_age3",
            "std_delta1", "std_delta2", "std_delta3",
            "std_corr_delta1", "std_corr_delta2", "std_corr_delta3",
        ]
    )
    print(df)

if __name__ == "__main__":
    main()
