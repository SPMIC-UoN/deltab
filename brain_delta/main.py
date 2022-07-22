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
from .brain_delta import BrainDelta

def main():
    parser = argparse.ArgumentParser(f'Brain age calculator v{__version__}', add_help=True)
    parser.add_argument('--ages', required=True, help='Path to text file containing real ages for subjects')
    parser.add_argument('--features', required=True, help='Path to text file containing regressor features for subjects')
    parser.add_argument('--feature-proportion', type=float, help='Proportion of features to retain in PCA reduction (0-1)')
    parser.add_argument('--feature-num', type=int, help='Number of features to retain in PCA reduction')
    parser.add_argument('--predict-ages', help='Path to text file containing true ages for prediction')
    parser.add_argument('--predict-features', help='Path to text file containing regressor features for prediction')
    parser.add_argument('--simple-model', action="store_true", default=False, help='Use simple (biased) regression model')
    parser.add_argument('--include-quad', action="store_true", default=False, help='Include quadratic age dependence in model')
    parser.add_argument('--output-delta', action="store_true", default=False, help='Output brain age delta rather than actual age')
    parser.add_argument('-o', '--output', default="braindelta", help='Filename for output prediciton')
    parser.add_argument('--overwrite', action="store_true", default=False, help='If specified, overwrite any existing output')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    ages = np.loadtxt(args.ages)
    features = np.loadtxt(args.features)

    if args.predict_ages is not None and args.predict_features is not None:
        predict_ages = np.loadtxt(args.predict_ages)
        predict_features = np.loadtxt(args.predict_features)
    elif args.predict_ages is None and args.predict_features is None:
        predict_ages = ages
        predict_features = features
    else:
        raise ValueError("--predict-ages and --predict-features must be specified together")

    if os.path.exists(args.output) and not args.overwrite:
        raise ValueError(f"Output file {args.output} already exists - remove or specify a different name")
    os.makedirs(args.output, exist_ok=True)

    b = BrainDelta()
    b.train(ages, features, ev_proportion=args.feature_proportion, ev_num=args.feature_num, include_quad=args.include_quad)
    prediction = b.predict(predict_ages, predict_features, unbiased_model=not args.simple_model, return_delta=args.output_delta)
    np.savetxt(args.output, prediction)

if __name__ == "__main__":
    main()
