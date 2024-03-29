#!/usr/bin/env python3

# =============================================================================
#
# This scripts trains a Delta_corr_ML model and uses it for predictions
# https://doi.org/10.1021/acs.jctc.3c00165
#
# Run:
# mlcorrect_corr.py [input files/directories] [options]
#
# Copyright (C) 2021-2024 Julius Kleine Büning
#
# =============================================================================


import sys
import os
import shutil
import argparse
import pickle
import numpy as np
import pandas as pd

# Suppress info and warning messages from tensorflow
# (Print: 0: INFO, WARNING, ERROR; 1: WARNING, ERROR; 2: ERROR; 3: none)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs.modeling

from functions_mlcorrect import *
import additional_corr as addcorr


########## OWN FUNCTIONS ##########

def build_model(train_dataset, n_nodes_1: int, n_nodes_2: int, drop: float, opt: str, activ: str) -> keras.Sequential:
    """Build the sequential regression model with layers.
    
    Args:
        train_dataset: Data set that shall be used for training (pd.DataFrame object).
        n_nodes_1: Number of nodes in 1st hidden layer.
        n_nodes_2: Number of nodes in 2nd hidden layer.
        drop: Dropout rate after the 1st hidden layer.
        opt: Optimizing algorithm.
        activ: Activation function.

    Returns:
        The trained model (keras.Sequential).
    """

    model = keras.Sequential([
        # 1st hidden layer
        layers.Dense(n_nodes_1, activation=activ, input_shape=[len(train_dataset.keys())]),
        layers.Dropout(drop),
        # 2nd hidden layer
        layers.Dense(n_nodes_2, activation=activ),
        # output layer (needs to have exactly 1 neuron because this is a regression problem)
        layers.Dense(1)
    ])

    if opt == 'rmsprop': optimizer = keras.optimizers.RMSprop(0.001)
    elif opt == 'adam': optimizer = keras.optimizers.Adam()
    elif opt == 'sgd': optimizer = keras.optimizers.SGD()
    else:
        print("ERROR: Unknown optimizer selected in build_model()!")
        exit()

    model.compile(loss='mae', optimizer=optimizer, metrics=[max_error_neg, max_error_pos, mean_error, 'mae', 'mse', rmse, standard_deviation])

    return model


def evaluate_labels(labels: pd.Series) -> dict:
    """Evaluate the MAX (neg/pos), ME, MAE, MSE, RMSE, and SD from given data labels.

    The metrics are defined as follows:
    MAX_neg (Maximum negative): Largest negative error (= lowest error value)
    MAX_pos (Maximum positive): Largest positive error (= largest error value)
    ME (Mean Error): Average of all (signed) errors.
    MAE (Mean Absolute Error): Average of the absoulte errors.
    MSE (Mean Squared Error): Average of the squared errors.
    RMSE (Root Mean Squared Error): Square root of average of the squared errors.
    SD (Standard Deviation): Standard deviation of all errors.
    
    Args:
        labels: Array with target data of the data set (pd.Series object).

    Returns:
        Dictionary with entries MAX_neg, MAX_pos, ME, MAE, MSE, RMSE, and SD.
    """

    # labels contain the target of the data set, which is reference - trial (here CC - DFT)
    # the error of the trial is labels*(-1) = trial - reference
    errors = [labels.iat[i]*(-1) for i in range(labels.size)]

    max_neg = min(errors)
    max_pos = max(errors)
    me = sum(errors) / len(errors)
    mae = sum([abs(e) for e in errors]) / len(errors)
    mse = sum([e**2 for e in errors]) / len(errors)
    rmse = np.sqrt(mse)
    sd = np.sqrt(sum([(e-me)**2 for e in errors]) / (len(errors)-1))

    """
    # NOTE: The following SD definition was used for the original ML4NMR paper (10.1021/acs.jctc.3c00165),
    # but is not in line with the definition in the SI (the denominator is probably len(errors) instead of len(errors)-1)
    sd = np.std(errors)
    """

    return {'MAX_neg': max_neg, 'MAX_pos': max_pos, 'ME': me, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'SD': sd}


def evaluate_data(data: pd.Series, refdata: pd.Series) -> dict:
    """Get the most important metrics from linear regression corrected data.
    
    Args:
        data: Array with the scaled data obtained from scale_linear() (pd.Series object).
        refdata: Array with the high-level reference data (pd.Series object).
    
    Returns:
        Same as evaluate_labels(): MAX_neg, MAX_pos, ME, MAE, MSE, RMSE, SD.
    """

    # Note that 'labels' always mean ref - trial (which is the target of the ML model)
    # and trial - ref refers to the actual error of a prediction method
    labels = refdata - data
    return evaluate_labels(labels)


def print_details(seg_dataset: pd.DataFrame, normed_data: pd.DataFrame , labels: pd.Series, name='?'):
    """Print a list of the true and predicted chemical shift/deviation and some other quantities of a training run.

    This is only meant for testing and for plots (can be invoked by the --printraw flag).

    Args:
        seg_dataset: A segment of the data set (e.g., the test data set) (pd.DataFrame object).
        normed_data: The normed data that belongs to seg_dataset (pd.DataFrame object).
        labels: The labels (targets) that belong to seg_dataset (pdf.Series object).
        name: Name specifier (str).
    
    Returns:
        Nothing (justs prints the data).
    """

    predictions = model.predict(normed_data).flatten()
    print("Detailed " + name + " set results:")
    print("Number & True deviation & Predicted deviation & shift_low & shift_high & Predicted shift & Prediction error DFT & Prediction error DFT+ML")

    for i in range(labels.size):
        print("{:4d} & {:13.8f} & {:13.8f} & {:13.8f} & {:13.8f} & {:13.8f} & {:13.8f} & {:13.8f}".format(
            labels.index[i], labels.iat[i], predictions[i], seg_dataset['shift_low'].iat[i],
            seg_dataset['shift_low'].iat[i] + labels.iat[i], seg_dataset['shift_low'].iat[i] + predictions[i],
            labels.iat[i]*(-1.0), predictions[i] - labels.iat[i]
        ))
    
    print("")

    return

########## END OWN FUNCTIONS ##########


if __name__ == '__main__':

    # Print header and version
    print("\n\n          +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+")
    print("          |                                               |")
    print("          |                  M L 4 N M R                  |")
    print("          |                       -                       |")
    print("          |               M L C O R R E C T               |")
    print("          |                                               |")
    print("          |            Julius B. Kleine Büning            |")
    print("          |   Mulliken Center for Theoretical Chemistry   |")
    print("          |          University of Bonn, Germany          |")
    print("          |                                               |")
    print("          +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+\n")

    print("              Version: {}\n\n".format(version_mlcorrect))

    # Print versions
    print("Using the following Python version:")
    print(sys.version)
    print("Using TensorFlow version: {}\n\n".format(tf.__version__))

    workdir = os.getcwd()

    # Initialize a parser for the command line input
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', type=str, help='Train the model based on data form a given data file, provide the path of the data file.', metavar='datapath')
    parser.add_argument('-p', '--predict', nargs=2, help='Use a pre-trained model to predict the correction for a sample, provide data path of sample data and pre-trained model.', metavar=('samplepath', 'modelpath'))
    parser.add_argument('-n', '--nucleus', choices=['h', 'c', 'H', 'C'], required=True, help='Choose the NMR nucleus for which to apply the trainig/correction (H or C; mandatory).', metavar='nuc')
    parser.add_argument('-s', '--randomseed', type=int, default=0, help='Random seed for the initial weights of the ML model, default: 0.', metavar='seed')
    parser.add_argument('-e', '--epochs', type=int, help='The maximum number of epochs for the training procedure, default: 1000/2000 (H/C).', metavar='Nepochs')
    parser.add_argument('--noearlystop', action='store_false', help='Disable early stopping of the training process, all epochs defined with -e/--epochs will be used.')
    parser.add_argument('--nneurons1', type=int, help='Number of neurons in 1st layer, default: 120 (1H), 80 (13C).', metavar='Nneu')
    parser.add_argument('--nneurons2', type=int, default=8, help='Number of neurons in 2nd layer, default: 8.', metavar='Nneu')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout rate of 1st layer, default: 0.15.', metavar='D')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer used in the ML training process, default: adam.', metavar='opt')
    parser.add_argument('--activation', type=str, default='softmax', help='Activation function used in the ML training process, default: softmax.', metavar='func')
    parser.add_argument('--exclude', type=str, help='Exclude descriptor(s) from ML training process, provide in string separated by blank spaces (ONLY USED FOR TESTING!).', metavar='descriptors')
    parser.add_argument('--printraw', type=str, choices=['train', 'test', 'both'], help='Print raw data for every data point in training/test set (only needed for plots).', metavar='data')
    args = parser.parse_args()

    # Check for training or prediction mode
    do_train = True if args.train is not None else False
    do_predict = True if args.predict is not None else False

    if do_train and do_predict:
        print("\nSorry, I can't train and predict at the same time!")
        exit()

    if not (do_train or do_predict):
        print("\nPlease tell me what to do. Use --train option to train a model or --predict to use a trained model.")
        exit()

    # Get all global parameters form the parsed input
    nuc = args.nucleus.lower()
    random_seed = args.randomseed
    early_stop = args.noearlystop
    if args.epochs is None:
        if nuc == 'h': epochs = 1000
        if nuc == 'c': epochs = 2000
    else:
        epochs = args.epochs
    if args.nneurons1 is None:
        if nuc == 'h': n_neurons_1 = 120
        if nuc == 'c': n_neurons_1 = 80
    else:
        n_neurons_1 = args.nneurons1
    n_neurons_2 = args.nneurons2
    dropout_rate = args.dropout
    optimizer = args.optimizer
    activation = args.activation
    exclude_descriptors = [] if args.exclude is None else args.exclude.split()
    print_testdetails = []
    if args.printraw in ['train', 'both']: print_testdetails.append('train')
    if args.printraw in ['test', 'both']: print_testdetails.append('test')

    # Set the global tf random seed in order to get reproducible results
    # more info on that: https://www.tensorflow.org/api_docs/python/tf/random/set_seed
    tf.random.set_seed(random_seed)   # this works with tf version 2.7


    ### TRAIN THE NMR SHIFT ERROR CORRECTION FROM THE DATA SET ###
    if do_train:

        # Print all settings and paths
        print("Started mlcorrect in training mode. Settings:")
        print("NMR nucleus                      : {}".format(nuc.upper()))
        print("Random seed for build model      : {}".format(random_seed))
        print("Maximum number of training epochs: {}".format(epochs))
        print("Early stopping activated         : {}".format(early_stop))
        print("Number of neurons in 1st layer   : {}".format(n_neurons_1))
        print("Number of neurons in 2nd layer   : {}".format(n_neurons_2))
        print("Dropout rate                     : {}".format(dropout_rate))
        print("Optimizer                        : {}".format(optimizer))
        print("Activation function (both layers): {}".format(activation))
        print("Exclude descriptors              : {}\n".format('No' if not exclude_descriptors else ', '.join(exclude_descriptors)))

        if os.path.isfile(args.train):
            dataset_path = os.path.abspath(args.train)
            print("Sample data used for training:\n{}\n\n".format(dataset_path))
        else:
            print("ERROR: The input data file {} does not exist or is not a file!".format(args.train))
            exit()

        # Read column names from 2nd line of the dataset file
        with open(dataset_path, 'r') as inp:
            header = [next(inp) for _ in range(2)]
        column_names = header[1].replace('#', '').split()

        # Read the file, dataset is a pandas.DataFrame object
        dataset = pd.read_csv(dataset_path, names=column_names, na_values='?', comment='#', sep=' ', skipinitialspace=True)

        # Print the complete data set that has been read in
        print("Complete data set:\n{}\n\n".format(dataset))

        # Optionally exclude some data columns from the input (FOR TESTING ONLY!)
        # symmetric = ACSF + SOAP
        descriptors = {
            'geometric': {
                'h': ['CN(X)', 'dist_HC', 'no_HCH', 'no_HYH', 'no_HYC', 'no_HYN', 'no_HYO'],
                'c': ['CN(X)', 'no_CH', 'no_CC', 'no_CN', 'no_CO', 'no_CYH', 'no_CYC', 'no_CYN', 'no_CYO']
            },
            'electronic': {
                'h': ['at_charge_mull', 'at_charge_loew', 'orb_charge_mull_s', 'orb_charge_mull_p', 'orb_charge_loew_s', 'orb_charge_loew_p', 'BO_loew', 'BO_mayer', 'mayer_VA'],
                'c': ['at_charge_mull', 'at_charge_loew', 'orb_charge_mull_s', 'orb_charge_mull_p', 'orb_charge_mull_d', 'orb_stdev_mull_p', 'orb_charge_loew_s', 'orb_charge_loew_p', 'orb_charge_loew_d', 'orb_stdev_loew_p', 'BO_loew_sum', 'BO_loew_av', 'BO_mayer_sum', 'BO_mayer_av', 'mayer_VA']
            },
            'magnetic': {
                'h': ['shift_low', 'shift_low_neighbor_C', 'shielding_dia', 'shielding_para', 'span', 'skew', 'asymmetry', 'anisotropy'],
                'c': ['shift_low', 'shielding_dia', 'shielding_para', 'span', 'skew', 'asymmetry', 'anisotropy']
            },
            'symmetric': {
                'h': list(addcorr.acsf_labels) + list(addcorr.soap_labels_reduced),
                'c': list(addcorr.acsf_labels) + list(addcorr.soap_labels_reduced)
            },
            'acsf': {
                'h': list(addcorr.acsf_labels),
                'c': list(addcorr.acsf_labels)
            }
        }

        for descriptor in ['geometric', 'electronic', 'magnetic', 'symmetric', 'acsf']:
            if descriptor in exclude_descriptors:
                for d in descriptors[descriptor][nuc]: dataset.pop(d)

        # Now, the last 1/8 of the data set is used as the test data set.
        # The first 7/8 are the training data set.
        # If you want to investigate different arrangements of the data (shuffle mode,
        # ordering, e.g., for cross-validation), change the data set file accordingly.

        # The first 87.5% (7/8) are training set, test set starts after these points
        start_test = int(0.875*len(dataset))
        end_test = len(dataset)
        test_dataset = dataset[start_test:end_test]
        train_dataset = dataset.drop(test_dataset.index)

        # Print training and test data sets
        print("Training data set:\n{}\n\n".format(train_dataset))
        print("Test data set:\n{}\n\n".format(test_dataset))

        ### Beginning of linear regression (LR) part ###
        # (This part is skipped if some descriptors are excluded for testing reasons)
        if not exclude_descriptors:
            print("LINEAR REGRESSION CORRECTION:\n")
            slope, intercept, r_squared = linear_fit(train_dataset['shift_low'], train_dataset['shift_high-low'])
            print("Training data fitted to linear function with:\nSlope:     {:10.6f}\nIntercept: {:10.6f}\nR^2:       {:10.6f}\n".format(slope, intercept, r_squared))

            lr_train_target = train_dataset['shift_high-low'] + train_dataset['shift_low']
            lr_train_unscaled = train_dataset['shift_low']
            lr_train_scaled = scale_linear(train_dataset['shift_low'], slope, intercept)

            lr_test_target = test_dataset['shift_high-low'] + test_dataset['shift_low']
            lr_test_unscaled = test_dataset['shift_low']
            lr_test_scaled = scale_linear(test_dataset['shift_low'], slope, intercept)

            stats_lr_train = evaluate_data(lr_train_unscaled, lr_train_target)
            stats_lr_train_scaled = evaluate_data(lr_train_scaled, lr_train_target)
            stats_lr_test = evaluate_data(lr_test_unscaled, lr_test_target)
            stats_lr_test_scaled = evaluate_data(lr_test_scaled, lr_test_target)

            print("Statistics evaluated on the traning data set:")
            for stat in stats_lr_train.keys():
                print("Original {:>7}: {:10.6f} ppm; with LR correction: {:10.6f} ppm".format(stat, stats_lr_train[stat], stats_lr_train_scaled[stat]))
            print("\nStatistics evaluated on the test data set:")
            for stat in stats_lr_test.keys():
                print("Original {:>7}: {:10.6f} ppm; with LR correction: {:10.6f} ppm".format(stat, stats_lr_test[stat], stats_lr_test_scaled[stat]))

        ### End of linear regression part ###

        # Get the overall statistics and exclude column 'shift_high-low' (columns are transposed to rows)
        train_stats = train_dataset.describe()
        train_stats.pop('shift_high-low')
        train_stats = train_stats.transpose()

        # Split the column that will be trained from the dataset to have it separately available
        # *_labels are now (one-dimensional) pandas.Series objects
        train_labels = train_dataset.pop('shift_high-low')
        test_labels = test_dataset.pop('shift_high-low')

        # Create version of train_stats without zeros to avoid NaN in the norm() method (this is a diry hack but just for safety)
        train_stats_nozero = train_stats.copy()
        for col in train_dataset.columns:
            # Change mean and std entries if all entries are the same (meaning std = 0 thus x/0 in norm() method)
            if all_equal(train_dataset[col]):
                print("ATTENTION: All values in column {} are equal. Brutally setting all normed values to 0.0 or 1.0.".format(col))
                # This looks strange, but ensures that norm() will give all 0.0 if unnormed values are all 0.0 ...
                if train_stats_nozero['mean'][col] == 0.0:
                    train_stats_nozero['std'][col] = 1.0
                # ... and will give all 1.0 if unnormed values are all the same but != 0.0
                else:
                    train_stats_nozero['std'][col] = train_stats_nozero['mean'][col]
                    train_stats_nozero['mean'][col] = 0.0

        # Now norm the training and test data
        normed_train_data = norm(train_dataset, train_stats_nozero)
        normed_test_data = norm(test_dataset, train_stats_nozero)

        # This is where the ML starts
        print("\n\nSTARTING THE ML TRAINING PROCESS\n")

        # Build the model and print a summary
        model = build_model(train_dataset, n_neurons_1, n_neurons_2, dropout_rate, optimizer, activation)
        print("Model summary:", model.summary())

        ### Train the model ###
        # If early_stop == True: stop training when values are not improving anymore (dynamic number of epochs)
        if early_stop:
            # The patience parameter is the amount of epochs to check for improvement
            stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            history = model.fit(
                normed_train_data, train_labels, epochs=epochs, validation_split=0.2,
                verbose=0, callbacks=[stop, tensorflow_docs.modeling.EpochDots()]
            )
        else:
            history = model.fit(
                normed_train_data, train_labels, epochs=epochs, validation_split=0.2,
                verbose=0, callbacks=[tensorflow_docs.modeling.EpochDots()]
            )
        print("")

        # Delete old version of saved model if one still exists
        model_path = os.path.join(workdir, "tf_model_" + nuc)
        if os.path.isdir(model_path):
            print("\nWARNING: Old model in {} will be deleted.".format(model_path))
            shutil.rmtree(os.path.join(workdir, "tf_model_" + nuc))

        # Save the new model in the working directory for predictions
        model.save(model_path)
        print("\nINFORMATION: New trained model is saved in: {}\n".format(model_path))

        # Save some other metadata needed for the prediction part via pickle in the same directory as the model
        with open(os.path.join(model_path, "metadata.pkl"), 'wb') as out:
            pickle.dump(column_names, out, pickle.HIGHEST_PROTOCOL)
            pickle.dump(train_stats_nozero, out, pickle.HIGHEST_PROTOCOL)
        
        # Print the history of the trainig over the epochs
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        print("Training history:\n", hist)

        # Test the model on the training and test data set (overfitted if much better on the training than on the test set)
        # Get the uncorrected metrics
        stats_train_old = evaluate_labels(train_labels)
        stats_test_old = evaluate_labels(test_labels)
        # Predict the train and test set and get the corrected metrics
        predictions_train = model.predict(normed_train_data).flatten()
        predictions_test = model.predict(normed_test_data).flatten()
        stats_train_new = evaluate_labels(pd.Series([train_labels.iat[i] - predictions_train[i] for i in range(train_labels.size)]))
        stats_test_new = evaluate_labels(pd.Series([test_labels.iat[i] - predictions_test[i] for i in range(test_labels.size)]))

        print("\n\n************************")
        print("*** FINAL STATISTICS ***")
        print("************************\n")

        print("*** TRAINING SET ***")
        print("                                         {:>9}{:>15}".format("original", "ML-corrected"))
        print("Minimum error value / ppm     (MAX_neg): {:9.4f}      {:9.4f}".format(stats_train_old['MAX_neg'], stats_train_new['MAX_neg']))
        print("Maximum error value / ppm     (MAX_pos): {:9.4f}      {:9.4f}".format(stats_train_old['MAX_pos'], stats_train_new['MAX_pos']))
        print("Mean (signed) Error / ppm     (ME)     : {:9.4f}      {:9.4f}".format(stats_train_old['ME'], stats_train_new['ME']))
        print("Mean Absolute Error / ppm     (MAE)    : {:9.4f}      {:9.4f} ({:.1f}% reduction)".format(stats_train_old['MAE'], stats_train_new['MAE'], (1 - stats_train_new['MAE']/stats_train_old['MAE'])*100))
        print("Mean Squared Error / ppm^2    (MSE)    : {:9.4f}      {:9.4f}".format(stats_train_old['MSE'], stats_train_new['MSE']))
        print("Root Mean Squared Error / ppm (RMSE)   : {:9.4f}      {:9.4f} ({:.1f}% reduction)".format(stats_train_old['RMSE'], stats_train_new['RMSE'], (1 - stats_train_new['RMSE']/stats_train_old['RMSE'])*100))
        print("Standard Deviation / ppm      (SD)     : {:9.4f}      {:9.4f}\n".format(stats_train_old['SD'], stats_train_new['SD']))

        print("*** TEST SET ***")
        print("                                         {:>9}{:>15}".format("original", "ML-corrected"))
        print("Minimum error value / ppm     (MAX_neg): {:9.4f}      {:9.4f}".format(stats_test_old['MAX_neg'], stats_test_new['MAX_neg']))
        print("Maximum error value / ppm     (MAX_pos): {:9.4f}      {:9.4f}".format(stats_test_old['MAX_pos'], stats_test_new['MAX_pos']))
        print("Mean (signed) Error / ppm     (ME)     : {:9.4f}      {:9.4f}".format(stats_test_old['ME'], stats_test_new['ME']))
        print("Mean Absolute Error / ppm     (MAE)    : {:9.4f}      {:9.4f} ({:.1f}% reduction)".format(stats_test_old['MAE'], stats_test_new['MAE'], (1 - stats_test_new['MAE']/stats_test_old['MAE'])*100))
        print("Mean Squared Error / ppm^2    (MSE)    : {:9.4f}      {:9.4f}".format(stats_test_old['MSE'], stats_test_new['MSE']))
        print("Root Mean Squared Error / ppm (RMSE)   : {:9.4f}      {:9.4f} ({:.1f}% reduction)".format(stats_test_old['RMSE'], stats_test_new['RMSE'], (1 - stats_test_new['RMSE']/stats_test_old['RMSE'])*100))
        print("Standard Deviation / ppm      (SD)     : {:9.4f}      {:9.4f}\n".format(stats_test_old['SD'], stats_test_new['SD']))

        """
        # The following final evaluation was used for the original ML4NMR paper (10.1021/acs.jctc.3c00165),
        # but is not in line with the data printed below that was also used for some of the plots.
        # Fortunately, this only affects the definintion of the RMSE and the SD which are slightly too low
        # (also MAX_neg and MAX_pos are wrong, but have not been calculated for the ML4NMR project).
        # The RMSE was not included in the manuscript and the SD is slightly wrong in the the text:
        # 1H test set SD for PBE0/pcSseg-2 with ML correction is 0.036 ppm instead of 0.034 ppm (2 appearances).
        # 13C test set SD for PBE0/pcSseg-2 with ML correction is 0.70 ppm instead of 0.66 ppm (2 appearances).
        # 13C test set SD for PBE0/pcSseg-2 without correction is 4.70 ppm instead of 4.69 ppm (2 appearances).
        # The last is due to the wrong definition of the SD in the old version (see evaluate_labels).

        stats_train = {model.metrics_names[i]: stat for i, stat in enumerate(model.evaluate(normed_train_data, train_labels, verbose=2))}
        stats_test = {model.metrics_names[i]: stat for i, stat in enumerate(model.evaluate(normed_test_data, test_labels, verbose=2))}

        print("*** TRAINIG SET ***")
        print("                                         {:>9}{:>15}".format("original", "ML-corrected"))
        print("Minimum error value / ppm     (MAX_neg): {:9.4f}      {:9.4f}".format(stats_train_old['MAX_neg'], stats_train['max_error_neg']))
        print("Maximum error value / ppm     (MAX_pos): {:9.4f}      {:9.4f}".format(stats_train_old['MAX_pos'], stats_train['max_error_pos']))
        print("Mean (signed) Error / ppm     (ME)     : {:9.4f}      {:9.4f}".format(stats_train_old['ME'], stats_train['mean_error']))
        print("Mean Absolute Error / ppm     (MAE)    : {:9.4f}      {:9.4f} ({:.1f}% reduction)".format(stats_train_old['MAE'], stats_train['mae'], (1 - stats_train['mae']/stats_train_old['MAE'])*100))
        print("Mean Squared Error / ppm^2    (MSE)    : {:9.4f}      {:9.4f}".format(stats_train_old['MSE'], stats_train['mse']))
        print("Root Mean Squared Error / ppm (RMSE)   : {:9.4f}      {:9.4f} ({:.1f}% reduction)".format(stats_train_old['RMSE'], stats_train['rmse'], (1 - stats_train['rmse']/stats_train_old['RMSE'])*100))
        print("Standard Deviation / ppm      (SD)     : {:9.4f}      {:9.4f}\n".format(stats_train_old['SD'], stats_train['standard_deviation']))

        print("*** TEST SET ***")
        print("                                         {:>9}{:>15}".format("original", "ML-corrected"))
        print("Minimum error value / ppm     (MAX_neg): {:9.4f}      {:9.4f}".format(stats_test_old['MAX_neg'], stats_test['max_error_neg']))
        print("Maximum error value / ppm     (MAX_pos): {:9.4f}      {:9.4f}".format(stats_test_old['MAX_pos'], stats_test['max_error_pos']))
        print("Mean (signed) Error / ppm     (ME)     : {:9.4f}      {:9.4f}".format(stats_test_old['ME'], stats_test['mean_error']))
        print("Mean Absolute Error / ppm     (MAE)    : {:9.4f}      {:9.4f} ({:.1f}% reduction)".format(stats_test_old['MAE'], stats_test['mae'], (1 - stats_test['mae']/stats_test_old['MAE'])*100))
        print("Mean Squared Error / ppm^2    (MSE)    : {:9.4f}      {:9.4f}".format(stats_test_old['MSE'], stats_test['mse']))
        print("Root Mean Squared Error / ppm (RMSE)   : {:9.4f}      {:9.4f} ({:.1f}% reduction)".format(stats_test_old['RMSE'], stats_test['rmse'], (1 - stats_test['rmse']/stats_test_old['RMSE'])*100))
        print("Standard Deviation / ppm      (SD)     : {:9.4f}      {:9.4f}\n".format(stats_test_old['SD'], stats_test['standard_deviation']))
        """

        # Optionally print all train/test set data points (e.g. for use in plots)
        # Metrics calculated from this list should be identical to the ones above under *** FINAL STATISTICS ***
        if 'train' in print_testdetails: print_details(train_dataset, normed_train_data, train_labels, name='training')
        if 'test' in print_testdetails: print_details(test_dataset, normed_test_data, test_labels, name='test')

        print("\n mlcorrect all done.")


    ### PREDICT THE NMR SHIFT ERROR CORRECTION FOR AN UNKNOWN MOLECULE ###
    if do_predict:

        # Print settings and paths
        print("Started mlcorrect in predictoin mode for NMR nucleus: {}\n".format(nuc.upper()))

        if os.path.isfile(args.predict[0]):
            sample_path = os.path.abspath(args.predict[0])
            print("Sample data used for the prediction:\n{}".format(sample_path))
        else:
            print("ERROR: The ML input compound data file {} does not exist or is not a file!".format(args.predict[0]))
            exit()
        if os.path.isdir(args.predict[1]):
            model_path = os.path.abspath(args.predict[1])
            print("Pre-trained model used for the prediction:\n{}\n\n".format(model_path))
        else:
            print("ERROR: The saved ML model directory {} does not exist or is not a directory!".format(args.predict[1]))
            exit()

        # Load the metadata from the training run
        with open(os.path.join(model_path, "metadata.pkl"), 'rb') as inp:
            column_names = pickle.load(inp)
            train_stats_nozero = pickle.load(inp)
        
        # Load and print the sample data
        sample_data = pd.read_csv(sample_path, names=column_names, na_values='?', comment='#', sep=' ', skipinitialspace=True)
        print("Sample data:\n{}\n\n".format(sample_data))
        
        # Get the labels and norm the data
        sample_labels = sample_data.pop('shift_high-low')
        normed_sample_data = norm(sample_data, train_stats_nozero)
        
        # Check if high-level reference data are available
        has_highlevel = True
        for i in sample_labels:
            if not isinstance(i, float): has_highlevel = False

        # Read additional information from the sample data file (stored at the end of the file)
        with open(sample_path, 'r') as inp:
            data = inp.readlines()

        for line in data:
            tmp = line.split()
            if tmp[0] == '#':
                if tmp[1] == 'ref_h:' and nuc == 'h': ref = float(tmp[2])
                if tmp[1] == 'ref_c:' and nuc == 'c': ref = float(tmp[2])
                if tmp[1] == 'atom_numbers:': atnums = tuple([int(at) for at in tmp[2:]])

        # Load the model from the training run
        model = keras.models.load_model(model_path, custom_objects={'max_error_neg': max_error_neg, 'max_error_pos': max_error_pos, 'mean_error': mean_error, 'rmse': rmse, 'standard_deviation': standard_deviation})

        # Use the pre-trained model to predict unknown shift corrections
        sample_predictions = model.predict(normed_sample_data).flatten()

        # Collect corrected data
        # ATTENTION: a corrected shielding constant cannot be calculated here (or would be wrong), because sigma(ref,ML) or sigma(ref,CC) are unknown
        # They would be e.g.: sigma(sample,ML) = sigma(ref,ML) - delta(sample,low) - deviation(sample,ML)
        final_data = []
        for i, at in enumerate(atnums):
            # NMR convention: delta(sample) = sigma(ref) - sigma(sample)
            datapoint = {
                'index': i,
                'atom': at,
                'shielding_low': ref - sample_data['shift_low'].iat[i],
                'shift_low': sample_data['shift_low'].iat[i],
                'deviation_predicted': sample_predictions[i],
                'shift_low_corrected': sample_data['shift_low'].iat[i] + sample_predictions[i]
            }
            if has_highlevel:
                datapoint['deviation_true'] = sample_labels.iat[i]
                datapoint['shift_true'] = datapoint['shift_low'] + sample_labels.iat[i]
            final_data.append(datapoint)
        
        # Print the final prediction results
        print("*** FINAL SAMPLE PREDICTION RESULTS ***\n")
        if has_highlevel:
            print("Number   Atom   Calculated shift   Predicted deviation   Predicted shift   True CC deviation   True CC shift")
            for at in final_data:
                print("{:4d}     {:3d}        {:7.2f}              {:7.2f}             {:7.2f}            {:7.2f}           {:7.2f}".format(
                    at['index'], at['atom'], at['shift_low'], at['deviation_predicted'], at['shift_low_corrected'], at['deviation_true'], at['shift_true']
                ))
        else:
            print("Number   Atom   Calculated shift   Predicted deviation   Predicted shift")
            for at in final_data:
                print("{:4d}     {:3d}        {:7.4f}              {:7.4f}             {:7.4f}".format(
                    at['index'], at['atom'], at['shift_low'], at['deviation_predicted'], at['shift_low_corrected']
                ))

        # Write an output file containing the ML-corrected shifts
        with open(os.path.join(workdir, "corrected_" + os.path.basename(sample_path)), 'w') as out:
            if nuc == 'h': out.write("# ML-corrected 1H NMR chemical shifts for data in {}\n".format(sample_path))
            if nuc == 'c': out.write("# ML-corrected 13C NMR chemical shifts for data in {}\n".format(sample_path))
            if has_highlevel:
                out.write("# atom shift_uncorrected shift_corrected shift_true_CC\n")
                for at in final_data:
                    out.write("{} {} {} {}\n".format(at['atom'], at['shift_low'], at['shift_low_corrected'], at['shift_true']))
            else:
                out.write("# atom shift_uncorrected shift_corrected\n")
                for at in final_data:
                    out.write("{} {} {}\n".format(at['atom'], at['shift_low'], at['shift_low_corrected']))
        
        print("\n mlcorrect all done.")

