#!/bin/env python
"""
DELTAB: Brain age calculation tool

Command line interface
"""
import os
import argparse
import sys
import logging

import numpy as np

from ._version import __version__
from . import BrainDelta, Model

LOG = logging.getLogger(__name__)

def _load_data(fname):
    """
    Load Numpy array data from a text file
    
    The file may be delimited in various ways and may or may not have a 
    header. We basically try all possibilities until something works
    """
    for skiprows in (0, 1):
        for quotechar in (None, '"', "'"):
            for delimiter in (None, ",", "\t"):
                try:
                    return np.loadtxt(fname, delimiter=delimiter, quotechar=quotechar)
                except:
                    pass
    raise ValueError("Could not load data in {fname} - must be space, comma or tab delimited")

def _remove_nan_subjects(ages, features):
    """
    Remove subjects who have NaN in any feature (or NaN true age)
    """
    num_subjects = ages.shape[0]
    ages_out, features_out = [], []
    for idx in range(num_subjects):
        if np.isnan(ages[idx]) or np.count_nonzero(np.isnan(features[idx, :])):
            LOG.debug(f" - Removing subject {idx} because NaN found in age or features")
        else:
            ages_out.append(ages[idx])
            features_out.append(features[idx])
    if len(ages_out) != num_subjects:
        LOG.info(f" - Removed {num_subjects-len(ages_out)} subjects because NaN found in age or features")
    return np.array(ages_out), np.array(features_out)

def _nan_median_impute(features):
    """
    Replace NaN features with median value for that feature
    """
    median = np.nanmedian(features, axis=1)
    inds = np.where(np.isnan(features))
    if len(inds) > 0:
        LOG.info(f" - Imputing median for {len(inds)} NaN feature values")
        features[inds] = np.take(median, inds[1])
    return features

def _setup_logging(args):
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

def main():
    parser = argparse.ArgumentParser(f'Brain age calculator v{__version__}', add_help=True)
    parser.add_argument('--load', help='Path to file to load trained model data from')
    parser.add_argument('--save', help='Path to file to save trained model data to')
    parser.add_argument('--save-text', help='Path to folder to save trained model data in text format. Note that there is currently no load facility for this data')
    parser.add_argument('--train-ages', help='Path to delimited text file containing 1D real ages for training')
    parser.add_argument('--train-features',  help='Path to delimited text file containing 2D regressor features for training')
    parser.add_argument('--feature-nans', help='Strategy for dealing with subjects with NaN in features', choices=["median", "remove"], default="median")
    parser.add_argument('--feature-proportion', type=float, help='Proportion of features to retain in PCA reduction (0-1)')
    parser.add_argument('--feature-num', type=int, help='Number of features to retain in PCA reduction')
    parser.add_argument('--feature-var', type=float, help='Retain features that explain at lease this proportion of the variance')
    parser.add_argument('--kaiser-guttmann', action="store_true", default=False, help='Use Kaiser-Guttmann criterion to select number of features to retain in PCA reduction')
    parser.add_argument('--predict', help='Output mode', choices=['delta', 'age'], default="delta")
    parser.add_argument('--predict-ages', help='Path to delimited text file containing 1D true ages for prediction')
    parser.add_argument('--predict-features', help='Path to delimited text file containing 2D regressor features for prediction')
    parser.add_argument('--predict-model', help='Model for prediction', choices=['simple', 'unbiased', 'unbiased_quadratic', 'alternate', 'alternate_quadratic'], default='unbiased_quadratic')
    parser.add_argument('--predict-output', help='File to save prediction to', default="deltab.txt")
    parser.add_argument('--true-ages-output', help='File to save true ages, if required. Useful if removing subjects with NaNs', default=None)
    parser.add_argument('--overwrite', action="store_true", default=False, help='If specified, overwrite any existing output')
    parser.add_argument('--debug', action="store_true", default=False, help='Enable debug output')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    _setup_logging(args)
    b = BrainDelta()
    LOG.info(f"BRAIN AGE PREDICTOR v{__version__}")

    if not args.load and not (args.train_ages and args.train_features):
        parser.error("Must either load trained model or provide training data")
    elif args.load:
        LOG.info(f" - Loading model from {args.load}")
        b.load(args.load)
    else:
        LOG.info(f" - Training model - ages: {args.train_ages}, features: {args.train_features}")
        ages = _load_data(args.train_ages)
        features = _load_data(args.train_features)
        if args.feature_nans == "remove":
            ages, features = _remove_nan_subjects(ages, features)
        else:
            features = _nan_median_impute(features)
        b.train(ages, features, ev_proportion=args.feature_proportion, ev_num=args.feature_num, ev_kg=args.kaiser_guttmann, ev_var=args.feature_var)

    if args.save:
        b.save(args.save)
    if args.save_text:
        b.save(args.save_text)

    if args.predict is not None:
        if args.predict_ages and args.predict_features:
            LOG.info(f" - Generating prediction - ages: {args.predict_ages}, features: {args.predict_features}")
            predict_ages = _load_data(args.predict_ages)
            predict_features = _load_data(args.predict_features)
        elif not args.load:
            if not args.predict_ages and not args.predict_features:
                LOG.info(f" - Generating prediction using training data")
                predict_ages = ages
                predict_features = features
            elif not args.predict_ages or not args.predict_features:
                parser.error("--predict-ages and --predict-features must be specified together")
        else:
            parser.error("Prediction requested but no age/features provided")

        if os.path.exists(args.predict_output) and not args.overwrite:
            raise ValueError(f"Output file {args.predict_output} already exists - remove or specify a different name")
        prediction = b.predict(predict_ages, predict_features, model=Model.fromstr(args.predict_model), return_delta=True if args.predict == 'delta' else False)
        LOG.info(f" - Saving predicted {args.predict} to {args.predict_output}")
        np.savetxt(args.predict_output, prediction)

        if args.true_ages_output:
            LOG.info(f" - Saving true ages for included subjects to {args.true_ages_output}")
            np.savetxt(args.true_ages_output, predict_ages)

if __name__ == "__main__":
    main()
