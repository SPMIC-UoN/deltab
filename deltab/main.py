#!/bin/env python
"""
DELTAB: Brain age calculation tool

Command line interface
"""
import os
import argparse
import sys

import numpy as np

from ._version import __version__
from . import BrainDelta, Model

def load_data(fname):
    for skiprows in (0, 1):
        for quotechar in (None, '"', "'"):
            for delimiter in (None, ",", "\t"):
                try:
                    return np.loadtxt(fname, delimiter=delimiter, quotechar=quotechar)
                except:
                    pass
    raise ValueError("Could not load data in {fname} - must be space, comma or tab delimited")

def remove_nan_subjects(ages, features):
    num_subjects = ages.shape[0]
    ages_out, features_out = [], []
    for idx in range(num_subjects):
        if np.isnan(ages[idx]) or np.count_nonzero(np.isnan(features[idx, :])):
            print(f"Removing subject {idx} because NaN found in age or features")
        else:
            ages_out.append(ages[idx])
            features_out.append(features[idx])
    return np.array(ages_out), np.array(features_out)

def main():
    parser = argparse.ArgumentParser(f'Brain age calculator v{__version__}', add_help=True)
    parser.add_argument('--load', help='Path to file to load trained model data from')
    parser.add_argument('--save', help='Path to file to save trained model data to')
    parser.add_argument('--train-ages', help='Path to delimited text file containing 1D real ages for training')
    parser.add_argument('--train-features',  help='Path to delimited text file containing 2D regressor features for training')
    parser.add_argument('--remove-nan-subjects', action="store_true", default=False, help='Remove subjects with NaN as age or in features')
    parser.add_argument('--feature-proportion', type=float, help='Proportion of features to retain in PCA reduction (0-1)')
    parser.add_argument('--feature-num', type=int, help='Number of features to retain in PCA reduction')
    parser.add_argument('--feature-var', type=float, help='Retain features that explain at lease this proportion of the variance')
    parser.add_argument('--kaiser-guttmann', action="store_true", default=False, help='Use Kaiser-Guttmann criterion to select number of features to retain in PCA reduction')
    parser.add_argument('--predict', help='Output mode', choices=['delta', 'age'], default="delta")
    parser.add_argument('--predict-ages', help='Path to delimited text file containing 1D true ages for prediction')
    parser.add_argument('--predict-features', help='Path to delimited text file containing 2D regressor features for prediction')
    parser.add_argument('--predict-model', help='Model for prediction', choices=['simple', 'unbiased', 'unbiased_quadratic', 'alternate', 'alternate_quadratic'], default='unbiased_quadratic')
    parser.add_argument('--predict-output', help='File to save prediction to', default="deltab.txt")
    parser.add_argument('--overwrite', action="store_true", default=False, help='If specified, overwrite any existing output')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    b = BrainDelta()
    if not args.load and not (args.train_ages and args.train_features):
        parser.error("Must either load trained model or provide training data")
    elif args.load:
        b.load(args.load)
    else:
        ages = load_data(args.train_ages)
        features = load_data(args.train_features)
        if args.remove_nan_subjects:
            ages, features = remove_nan_subjects(ages, features)
            np.savetxt("ages_included.txt", ages) # FIXME temporary for comparison
        b.train(ages, features, ev_proportion=args.feature_proportion, ev_num=args.feature_num, ev_kg=args.kaiser_guttmann, ev_var=args.feature_var)

    if args.save:
        b.save(args.save)

    if args.predict is not None:
        if args.predict_ages and args.predict_features:
            # We have data for prediction
            predict_ages = load_data(args.predict_ages)
            predict_features = load_data(args.predict_features)
        elif not args.load:
            if not args.predict_ages and not args.predict_features:
                # Use training data to output prediction
                predict_ages = ages
                predict_features = features
            elif not args.predict_ages or not args.predict_features:
                parser.error("--predict-ages and --predict-features must be specified together")
        else:
            parser.error("Prediction requested but no age/features provided")

        if os.path.exists(args.predict_output) and not args.overwrite:
            raise ValueError(f"Output file {args.predict_output} already exists - remove or specify a different name")
        prediction = b.predict(predict_ages, predict_features, model=Model.fromstr(args.predict_model), return_delta=True if args.predict == 'delta' else False)
        np.savetxt(args.predict_output, prediction)

if __name__ == "__main__":
    main()
