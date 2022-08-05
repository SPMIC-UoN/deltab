"""
Table 1 from Smith et al 
"""
import sys

import numpy as np
import pandas as pd

from brain_delta.brain_delta import Model, BrainDelta

import utils

def DeltaCV(X, Y, J, Ncv=10):
    """
    Perform cross-validation

    The data is divided into Ncv parts which are iterated over. In each iteration
    the selected chunk is removed and the model trained on the remainder of the
    data. The trained model is then used to predict age on the left out chunk.

    :param X: Features matrix
    :param Y: Ages matrix
    :param J: Number of feature components to use from PCA reduction (0=no PCA reduction)
    :param Ncv: Number of cross-validation folds

    :return: Tuple of predicted deltas for each subject from model trained on 
             remainder of data when the chunk containing that subject was left out.
             In order: (simple model, corrected model, corrected quadratic model, 
             alternate model, alternate quadratic model)
    """
    # for slightly improved results, run this many times and average the outputs
    delta1 = np.zeros_like(Y)
    delta2 = np.zeros_like(Y)
    delta3 = np.zeros_like(Y)
    delta2q = np.zeros_like(Y)
    delta3q = np.zeros_like(Y)

    # Assign each subject to a random one of Ncv groups
    SUBJECT_GROUPS = np.random.randint(Ncv, size=len(Y))

    b = BrainDelta()
    for i in range(Ncv):
        PREDICTION_SET, TRAINING_SET = (SUBJECT_GROUPS==i), (SUBJECT_GROUPS!=i)

        # Train the model on subjects in the training set
        b.train(Y[TRAINING_SET], X[TRAINING_SET,:], ev_num=J)

        # Get predictions on the subjects not in the training set
        x_out, y_out = X[PREDICTION_SET,:], Y[PREDICTION_SET]
        delta1[PREDICTION_SET] = b.predict(y_out, x_out, model=Model.SIMPLE, return_delta=True)
        delta2[PREDICTION_SET] = b.predict(y_out, x_out, model=Model.UNBIASED, return_delta=True)
        delta2q[PREDICTION_SET] = b.predict(y_out, x_out, model=Model.UNBIASED_QUADRATIC, return_delta=True)
        delta3[PREDICTION_SET] = b.predict(y_out, x_out, model=Model.ALTERNATE, return_delta=True)
        delta3q[PREDICTION_SET] = b.predict(y_out, x_out, model=Model.ALTERNATE_QUADRATIC, return_delta=True)

    return delta1, delta2, delta3, delta2q, delta3q

def corr(X, *Y):
    """
    Calculate correlation coefficient between X and each array in the sequence Y
    """
    return np.corrcoef(np.stack([X] + list(Y), axis=1),rowvar=False)[0, 1:]

def main():
    """
    Quadratic simulations (Table 2)
    """
    NUM_SIMULATIONS = 10                                               # 20 in paper
    NUM_SUBJECTS = 20000                                                # 20000 in paper
    MEAN_Y, RANGE_Y, NOISE_Y_BASE = 60, 25, 0.5                       # 60, 25, 0.5 in MATLAB code
    NOISE_DELTA = 2                                                   # 2y in paper
    NOISE_X, NUM_FEATURES_BASE, NUM_FEATURES_MIXED = 0.5, 100, 300    # SIM 1: 0.5, 100, 3000 in paper
    PCA_NUM_EVALUES = [v for v in  [0, 10, 50, 100, 1000, 2990] if v <= NUM_FEATURES_MIXED]

    print(f"Running simulations: NUM_SUBJECTS={NUM_SUBJECTS}, NUM_FEATURES_BASE={NUM_FEATURES_BASE}, NUM_FEATURES_MIXED={NUM_FEATURES_MIXED}")

    for Yquad in [0, 0.01, 0.025]:
        AllResults = np.zeros((len(PCA_NUM_EVALUES), 15, NUM_SIMULATIONS))
        for sim in range(NUM_SIMULATIONS):
            print(f"Simulation: {sim+1} of {NUM_SIMULATIONS}")

            # "a fairly sharply truncated (non-Gaussian) distribution for the age range (total range approximately 45-75y),"
            Y = MEAN_Y + RANGE_Y *(np.random.rand(NUM_SUBJECTS) - 0.5) + NOISE_Y_BASE*np.random.normal(size=NUM_SUBJECTS)
            DELTA_TRUE = NOISE_DELTA*np.random.normal(size=NUM_SUBJECTS)
            
            # "add quadratic aging effects. For these 3 simulations we set true 
            # quadratic-term α values(Eq. (9)) of 0, 0.01 and 0.025 respectively, 
            # corresponding to total deviations away from linear brain aging of 
            # 0, 4y and 10y."
            Y_DEMEAN = utils.demean(Y)
            Y2=utils.demean(Yquad*Y_DEMEAN**2)
            print(Y2.shape, Y_DEMEAN.shape)
            Y2=Y2-Y_DEMEAN * (np.dot(np.linalg.pinv(Y_DEMEAN[..., np.newaxis]), Y2))
            Yb = Y + DELTA_TRUE + Y2

            X0 = np.random.normal(size=(NUM_SUBJECTS, NUM_FEATURES_BASE))
            X0[:, 0] = Yb
            X0 = utils.normalize(X0)
            if NUM_FEATURES_MIXED:
                Xmix = np.random.normal(size=(NUM_FEATURES_BASE, NUM_FEATURES_MIXED))**5
                X1 = np.dot(X0, Xmix)
                X2 = utils.normalize(X1)
            else:
                X2 = X0
            X3 = utils.demean(X2+NOISE_X*np.random.normal(size=X2.shape))
            for j_idx, J in enumerate(PCA_NUM_EVALUES): # permuting pcaU gives same results as J=1 (ie null model ~ bad model)
                if J == 0:
                    print("Running model without PCA reduction")
                else:
                    print(f"Running model using top {J} eigenvalues")
                delta1, delta2, delta3, delta2q, delta3q = DeltaCV(X3,Y,J)
                page1, page2, page3 = Y+delta1, Y+delta2, Y+delta3
                agecorr = corr(Y, delta1, page1, page2, page3)
                corrs = corr(DELTA_TRUE, delta1, delta2, delta3, delta2q, delta3q)
                means = np.mean(np.abs([delta1, delta2, delta3, delta2q, delta3q]), axis=1)
                AllResults[j_idx, :, sim] = np.concatenate([np.atleast_1d(J), agecorr, means, corrs])
                
        mean_results = np.concatenate([np.mean(AllResults, axis=2), np.std(AllResults, axis=2)], axis=1)
        df = pd.DataFrame(
            mean_results, 
            columns=[
                "J",
                "mean_corr_agedelta1",
                "mean_corr_age1", "mean_corr_age2", "mean_corr_age3",
                "mean_delta1", "mean_delta2", "mean_delta3", "mean_delta2q", "mean_delta3q",
                "mean_corr_delta1", "mean_corr_delta2", "mean_corr_delta3", "mean_corr_delta2q", "mean_corr_delta3q",
                "std_J",
                "std_corr_agedelta1",
                "std_corr_age1", "std_corr_age2", "std_corr_age3",
                "std_delta1", "std_delta2", "std_delta3", "std_delta2q", "std_delta3q",
                "std_corr_delta1", "std_corr_delta2", "std_corr_delta3", "std_corr_delta2q", "std_corr_delta3q",
            ]
        )
        df.to_csv("table1.csv")

        PLOT_COLS = ["J", "mean_delta1", "mean_delta2", "mean_delta3", "mean_delta2q", "mean_delta3q", "mean_corr_delta1", "mean_corr_delta2", "mean_corr_delta3", "mean_corr_delta2q", "mean_corr_delta3q"]
        print(df[PLOT_COLS])

        if "--plot" in sys.argv:
            from matplotlib import pyplot as plt
            df.update(df[PLOT_COLS].applymap('{:,.2f}'.format))
            utils.do_table(plt.subplot(1, 1, 1), df[PLOT_COLS])
            plt.show()

if __name__ == "__main__":
    main()
