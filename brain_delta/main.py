#!/bin/env python
"""
BRAIN_DELTA: Brain age calculation tool

Command line interface
"""
import os
import argparse
import sys

import numpy as np

from ._version import __version__
from .brain_delta import BrainDelta, Model

def main():
    parser = argparse.ArgumentParser(f'Brain age calculator v{__version__}', add_help=True)
    parser.add_argument('--load', help='Path to file to load trained model data from')
    parser.add_argument('--save', help='Path to file to save trained model data to')
    parser.add_argument('--train-ages', help='Path to text file containing real ages for subjects')
    parser.add_argument('--train-features',  help='Path to text file containing regressor features for subjects')
    parser.add_argument('--feature-proportion', type=float, help='Proportion of features to retain in PCA reduction (0-1)')
    parser.add_argument('--feature-num', type=int, help='Number of features to retain in PCA reduction')
    parser.add_argument('--predict', help='Output mode', choices=['delta', 'age'])
    parser.add_argument('--predict-ages', help='Path to text file containing true ages for prediction')
    parser.add_argument('--predict-features', help='Path to text file containing regressor features for prediction')
    parser.add_argument('--predict-model', help='Model for prediction', choices=['simple', 'unbiased', 'unbiased_quadratic', 'alternate', 'alternate_quadratic'], default='unbiased_quadratic')
    parser.add_argument('--predict-output', help='File to save prediction to', default="braindelta.txt")
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
        ages = np.loadtxt(args.train_ages)
        features = np.loadtxt(args.train_features)
        b.train(ages, features, ev_proportion=args.feature_proportion, ev_num=args.feature_num)

    if args.save:
        b.save(args.save)

    if args.predict is not None:
        if args.predict_ages and args.predict_features:
            # We have data for prediction
            predict_ages = np.loadtxt(args.predict_ages)
            predict_features = np.loadtxt(args.predict_features)
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
