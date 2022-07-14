"""
Brain age estimator

Methodology from Smith et al 2019
https://doi.org/10.1016/j.neuroimage.2019.06.017

1. Your vector of ages is Y (subjects 1)
2. Your matrix of brain imaging measures is X (subjects features/ voxels)
3. Subtract the means from Y and all columns in X
4. Use SVD to replace X with its top 10–25% vertical eigenvectors
5. Compute Y^2, demean it and orthogonalise it with respect to Y to give Y^2_o
6. Create matrix Y2 = [Y Y^2_o] 
7. The initial model is Y B1 = X β1 - δ1. Do:
    (a) Compute initial age prediction β1 = X^-1 Y giving Y B1 = X β1 (where X^-1 is the pseudo-inverse of X)
    (b) Compute initial brain age delta δ1 = Y B1 - Y
8. The corrected model is δ1 = Y2 β2 + δ2q. Do: 
    (a) Computecorrected model fit β2 = Y2^-1 δ1 (correcting for bias in the initial fit and quadratic brain aging)
    (b) Compute final brain age delta δ2q = δ1 - Y2 β2
"""
import numpy as np
from sklearn.decomposition import PCA

class BrainDelta:
    """
    Class to predict brain age from features (e.g. IDPs, voxels)
    """

    def __init__(self, ages, features, ev_proportion=0.25):
        """
        Train model using ground truth data
        
        :param ages: Array of subject ages
        :param features: 2D array of [subjects, features]
        :param ev_proportion: Proportion of features to be retained in model
        """
        # 1. Your vector of ages is Y (subjects 1)
        self.y = np.squeeze(ages)
        if self.y.ndim != 1:
            raise ValueError("Ages must be a 1D array")
        num_subjects = len(self.y)

        # 2. Your matrix of brain imaging measures is X (subjects features/ voxels)
        self.x = np.array(features)
        if self.x.ndim != 2:
            raise ValueError("Features must be a 2D array")
        if self.x.shape[0] != num_subjects:
            raise ValueError("Number of subjects does not match number of rows in feature matrix")
                
        # 3. Subtract the means from Y and all columns in X
        self.y_mean = np.mean(self.y)
        self.x_mean = np.mean(self.x, axis=0)
        self.y_demean = self.y - self.y_mean
        self.x_demean = self.x - self.x_mean

        # 4. Use SVD to replace X with its top 10–25% vertical eigenvectors
        # Note that np.linalg.svd returns eigenvalues/vectors sorted in descending
        # order as we require
        self.num_ev = max(1, int(ev_proportion * self.x_demean.shape[1]))
        self.pca = PCA(n_components=self.num_ev)
        self.x_reduced = self.pca.fit_transform(self.x_demean)

        # 5. Compute Y2, demean it and orthogonalise it with respect to Y to give Y2 o
        self.ysq = np.square(self.y_demean)
        self.ysq_mean = np.mean(self.ysq)
        self.ysq_demean = self.ysq - self.ysq_mean
        self.ysq_orth_offset = np.dot(self.y_demean/np.linalg.norm(self.y_demean), self.ysq_demean)
        self.ysq_orth = self.ysq_demean - self.ysq_orth_offset
                
        # 6. Create matrix [Y2 Y2o]
        self.y2 = np.array([self.y_demean, self.ysq_orth]).T

        # 7. The initial model is Y B1 = X β1 + δ1. Do:
        #    (a) Compute initial age prediction β1 = X^-1 Y giving Y_B1 = X β1 (where X^-1 is the pseudo-inverse of X). 
        self.b1 = np.dot(np.linalg.pinv(self.x_reduced), self.y_demean)
        self.y_b1 = np.dot(self.x_reduced, self.b1)

        #    (b) Compute initial brain age delta δ1 = Y_B1 Y. 
        self.d1 = self.y_b1 - self.y_demean

        # 8. The corrected model is δ1 = Y2 β2 + δ2q. Do: 
        #    (a) Compute corrected model fit β2 = Y2^-1 δ1 (correcting for bias in the initial fit and quadratic brain aging).
        self.b2 = np.dot(np.linalg.pinv(self.y2), self.d1)

        #    (b) Compute final brain age delta δ2q = δ1 - Y2 β2
        self.d2q = self.d1 - np.dot(self.y2, self.b2)

    def predict(self, ages, features, unbiased_model=True):
        ages = np.atleast_1d(ages)
        if ages.ndim != 1:
            raise ValueError("Ages must be a 1D array")

        features = np.atleast_2d(features)
        if features.ndim != 2:
            raise ValueError("Features must be 1D or 2D array")
        if features.shape[0] != ages.shape[0]:
            raise ValueError("Number of subjects must match in features and ages arrays")
        if features.shape[1] != self.x.shape[1]:
            raise ValueError("Number of features must match training features")

        ages_demean = ages - self.y_mean
        features_demean = features - self.x_mean

        features_reduced = self.pca.transform(features_demean)
        age_predict = np.dot(features_reduced, self.b1) + self.y_mean
        
        if unbiased_model:
            agesq_demean = np.square(ages_demean) - self.ysq_mean
            agesq_orth = agesq_demean - self.ysq_orth_offset
            y2 = np.array([ages_demean, agesq_orth]).T
            return age_predict - np.dot(y2, self.b2)
        else:
            return age_predict

if __name__ == "__main__":
    REPS_PER_AGE = 5
    MIN_AGE = 30
    MAX_AGE = 40
    NUM_AGES = (MAX_AGE - MIN_AGE + 1)
    NUM_SUBJECTS = NUM_AGES * REPS_PER_AGE
    FEATURE_WEIGHTS = [0.8, 0.1, 0.1]
    FEATURE_NOISE_STD = 0.5
    NUM_FEATURES = len(FEATURE_WEIGHTS)

    BRAIN_AGE_DELTA = 2.5

    # True ages
    Y = np.concatenate([np.repeat(age, REPS_PER_AGE) for age in range(MIN_AGE, MAX_AGE+1)])
    print("TRUE AGE")
    print(Y)

    # For all individuals of a given age we will assume brain age deltas varying linearly
    # between -10 and +10
    DELTA_TRUE = np.tile(np.linspace(-BRAIN_AGE_DELTA, BRAIN_AGE_DELTA, num=REPS_PER_AGE), NUM_AGES)
    BRAIN_AGE_TRUE = Y + DELTA_TRUE
    print("TRUE DELTA")
    print(DELTA_TRUE)
    print("TRUE BRAIN AGE")
    print(BRAIN_AGE_TRUE)

    # Construct some features that predict brain age with noise
    X = np.zeros((NUM_SUBJECTS, NUM_FEATURES), dtype=float)
    for idx, weights in enumerate(FEATURE_WEIGHTS):
        X[:, idx] = BRAIN_AGE_TRUE * weights + np.random.normal(size=(NUM_SUBJECTS), scale=FEATURE_NOISE_STD)

    print("FEATURES")
    print(X)

    b = BrainDelta(Y, X)

    print("BIASED DELTA")
    print(b.d1)

    print("UNBIASED DELTA")
    print(b.d2q)

    delta = b.predict(Y, X)
    print("PREDICTED AGES 1")
    print(X)
    print(delta)

    X2 = X[:5]
    delta = b.predict(Y[:5], X2)
    print("PREDICTED AGES 2")
    print(X2)
    print(delta)
