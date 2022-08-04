"""
Brain age estimator

Methodology from Smith et al 2019
https://doi.org/10.1016/j.neuroimage.2019.06.017

1. Your vector of ages is Y (subjects 1)
2. Your matrix of brain imaging measures is X (subjects features/ voxels)
3. Subtract the means from Y and all columns in X
4. Use SVD to replace X with its top 10 - 25% vertical eigenvectors
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

class Model:
    SIMPLE=1
    UNBIASED=2
    UNBIASED_QUADRATIC=3
    ALTERNATE=4
    ALTERNATE_QUADRATIC=5

class BrainDelta:
    """
    Class to predict brain age from features (e.g. IDPs, voxels)
    """
    def __init__(self):
        self._trained = False

    def train(self, ages, features, ev_proportion=None, ev_num=None):
        """
        Train model

        :param ages: Array of subject ages
        :param features: 2D array of [subjects, features]
        :param ev_proportion: Optional proportion of features to be retained in model
        :param ev_num: Optional number of features to be retained in model
        """
        if ev_proportion and ev_num:
            raise ValueError("Only one of ev_proportion and ev_num may be specified")

        # 1. Your vector of ages is Y (subjects 1)
        self.ytrain = np.squeeze(ages)
        if self.ytrain.ndim != 1:
            raise ValueError("Ages must be a 1D array")
        self.num_subjects_train = len(self.ytrain)

        # 2. Your matrix of brain imaging measures is X (subjects features/voxels)
        self.xtrain = np.array(features)
        if self.xtrain.ndim != 2:
            raise ValueError("Features must be a 2D array")
        if self.xtrain.shape[0] != self.num_subjects_train:
            raise ValueError("Number of subjects does not match number of rows in feature matrix")
        self.num_features_train = self.xtrain.shape[1]

        # 3. Subtract the means from Y and all columns in X
        self.y_mean = np.mean(self.ytrain)
        self.y_demean = self._standardize_y(self.ytrain)

        if ev_num or ev_proportion:
            # 4. Use SVD to replace X with its top 10–25% vertical eigenvectors
            # Note that np.linalg.svd returns eigenvalues/vectors sorted in descending
            # order as we require
            if ev_num:
                self.ev_num = ev_num
            else:
                self.ev_num = max(1, int(ev_proportion * self.x_demean.shape[1]))
            self.pca = PCA(n_components=self.ev_num)
            self.x_reduced = self.pca.fit_transform(self.xtrain)
        else:
            # No reduction of features
            self.ev_num = self.xtrain.shape[1]
            self.pca = None
            self.x_reduced = self.xtrain
        self.x_norm, self.x_mean, self.x_std = self._normalize(self.x_reduced)

        # 5. Compute Y2, demean it and orthogonalise it with respect to Y to give Y2 o
        self.ysq = np.square(self.y_demean)
        self.ysq_mean = np.mean(self.ysq)
        self.ysq_demean = self.ysq - self.ysq_mean
        self.ysq_orth_offset = np.dot(self.y_demean/np.linalg.norm(self.y_demean), self.ysq_demean)
        self.ysq_orth = self._orthogonalize(self.ysq_demean, self.y_demean)

        # 6. Create matrix [Y2 Y2o]
        self.y2sq = np.array([self.y_demean, self.ysq_orth]).T

        # Equivalent to y2sq when not using quadratic correction
        self.y2 = self.y_demean[..., np.newaxis]

        # 7. The initial model is Y B1 = X β1 + δ1. Do:
        #    (a) Compute initial age prediction β1 = X^-1 Y giving Y_B1 = X β1 (where X^-1 is the pseudo-inverse of X). 
        self.b1 = np.dot(np.linalg.pinv(self.x_norm), self.y_demean)
        y_b1 = np.dot(self.x_norm, self.b1)

        #    (b) Compute initial brain age delta δ1 = Y_B1 Y. 
        d1 = y_b1 - self.y_demean

        # 8. The corrected model is δ1 = Y2 β2 + δ2q. Do: 
        #    (a) Compute corrected model fit β2 = Y2^-1 δ1 (correcting for bias in the initial fit and quadratic brain aging).
        self.b2 = np.dot(np.linalg.pinv(self.y2), d1)
        self.b2sq = np.dot(np.linalg.pinv(self.y2sq), d1)

        # Alternative model
        self.gamma = np.dot(np.linalg.pinv(self.y2), self.x_norm)
        self.gammasq = np.dot(np.linalg.pinv(self.y2sq), self.x_norm)

        self._trained = True

    def _demean(self, x, axis=0):
        return x - np.nanmean(x, axis=axis)

    def _normalize(self, x, mean=None, std=None, axis=0):
        if mean is None:
            mean=np.mean(x, axis=axis)
        if std is None:
            std = np.nanstd(x, axis=axis)
        return (x-mean) / std, mean, std

    def _standardize_y(self, y):
        return y - self.y_mean

    def _standardize_x(self, x):
        return self._normalize(x, mean=self.x_mean, std=self.x_std)[0]

    def _orthogonalize(self, a, b):
        return a - np.dot(b/np.linalg.norm(b), a)

    def predict(self, age, features, model=Model.UNBIASED_QUADRATIC, return_delta=False):
        """
        Predict brain age

        :param age: Array of subject true ages
        :param features: Array of [subjects, features]. Must be same features used in training
        :param unbiased_model: Use the unbiased model (default True)
        :param return_delta: Return the brain age delta (brain age - true age) rather than brain age
        """
        if not self._trained:
            raise RuntimeError("Model is not trained")

        age = np.atleast_1d(age)
        if age.ndim != 1:
            raise ValueError("Age must be a 1D array")

        features = np.atleast_2d(features)
        if features.ndim != 2:
            raise ValueError("Features must be 1D or 2D array")
        if features.shape[0] != age.shape[0]:
            raise ValueError("Number of subjects must match in features and age arrays")
        if features.shape[1] != self.num_features_train:
            raise ValueError("Number of features must match training features")

        age_demean = self._standardize_y(age)
        if self.pca is not None:
            features = self.pca.transform(features)
        features_norm = self._standardize_x(features)

        # Generate the quadratic age dependence if we need it
        if model in (Model.UNBIASED_QUADRATIC, Model.ALTERNATE_QUADRATIC):
            agesq = np.square(age_demean)
            agesq_demean = agesq - np.mean(agesq)
            agesq_orth = self._orthogonalize(agesq_demean, age_demean)
            y2 = np.array([age_demean, agesq_orth]).T
        else:
            y2 = age_demean[:, np.newaxis]

        # Generate the prediction
        if model == Model.ALTERNATE_QUADRATIC:
            #print("dot2: ", y2.shape, self.gammasq.shape)
            d = features_norm - np.dot(y2, self.gammasq)
            delta_predict = np.squeeze(np.dot(d, np.linalg.pinv(self.gammasq[np.newaxis, 0, :])) )
            #print(delta_predict.shape)
        elif model == Model.ALTERNATE:
            #print("dot: ", age_demean.shape, self.gamma.shape)
            d = features_norm - np.dot(age_demean[..., np.newaxis], self.gamma)
            #print("d", d.shape, np.mean(d), np.std(d))
            delta_predict = np.squeeze(np.dot(d, np.linalg.pinv(self.gamma)))
            #print(delta_predict.shape)
        else:
            # Compute initial brain age delta δ1 = Y_B1 Y. 
            delta_predict = np.dot(features_norm, self.b1) - age_demean
            #  For unbiased models, compute final brain age delta δ2q = δ1 - Y2 β2
            if model == Model.UNBIASED_QUADRATIC:
                delta_predict -= np.dot(y2, self.b2sq)
            elif model == Model.UNBIASED:
                delta_predict -= np.dot(y2, self.b2)

        if return_delta:
            return delta_predict
        else:
            return delta_predict + age

if __name__ == "__main__":
    # How many subjects for each true age
    REPS_PER_AGE = 5

    # Subject age range
    MIN_AGE, MAX_AGE = 30, 40
    NUM_AGES = (MAX_AGE - MIN_AGE + 1)
    NUM_SUBJECTS = NUM_AGES * REPS_PER_AGE

    # Number of features and weights
    FEATURE_WEIGHTS = [0.8, 0.1, 0.1]
    FEATURE_NOISE_STD = 0.5
    NUM_FEATURES = len(FEATURE_WEIGHTS)

    # Difference in brain age between subjects of same true age
    BRAIN_AGE_DELTA = 5

    # True ages
    Y = np.concatenate([np.repeat(age, REPS_PER_AGE) for age in range(MIN_AGE, MAX_AGE+1)])
    print("TRUE AGE")
    print(Y)

    # For all individuals of a given age we will assume brain age deltas varying linearly
    DELTA_TRUE = np.tile(np.linspace(-BRAIN_AGE_DELTA/2, BRAIN_AGE_DELTA/2, num=REPS_PER_AGE), NUM_AGES)
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

    # Train the model
    b = BrainDelta()
    b.train(Y, X)

    print("BIASED DELTA")
    print(b.d1)

    print("UNBIASED DELTA")
    print(b.d2)

    # Check with some predictions
    delta = b.predict(Y, X)
    print("PREDICTED AGES (unbiased)")
    print(delta)

    delta = b.predict(Y, X, unbiased_model=False)
    print("PREDICTED AGES (biased)")
    print(delta)
