import pandas as pd
import numpy as np


def preprocess_cup(path, save_path_input, save_path_target):
    """
    This function opens the file of the cup dataset, separates input columns from target
    and saves them in local files

    :param path: path of the file where the dataset is located
    :param save_path_input: path of the file where the input data will be saved
    :param save_path_target: path of the file where the target data will be saved
    """
    df = pd.read_csv(path, header=None,
                     names=['label', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'target_1',
                            'target_2'])

    # Extract target columns
    target = df[['target_1', 'target_2']]
    # Drop target column and Id (not a useful feature for Training)
    df.drop(labels=['target_1', 'target_2', 'label'], axis=1, inplace=True)

    np.set_printoptions(threshold=np.inf)
    np.save(save_path_input, df)
    np.save(save_path_target, target)
    return


def preprocess_cup_blind(path, save_path_input):
    """
    This function opens the file of the cup's blind test dataset, deletes the id column and saves it to a file

    :param path: path of the file where the dataset is located
    :param save_path_input: path of the file where the input data will be saved
    """
    df = pd.read_csv(path, header=None,
                     names=['label', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10'])

    df.drop(labels=['label'], axis=1, inplace=True)
    np.save(save_path_input, df)
    return


# Pre-processing for CUP dataset
preprocess_cup('../dataset_cup/ML-CUP21-TR.csv', '../dataset_cup/encoded_dataset/encoded_df_cup_input_training',
               '../dataset_cup/encoded_dataset/encoded_df_cup_target_training')

# Pre-processing for CUP's blind test
preprocess_cup_blind('../dataset_cup/ML-CUP21-TS.csv', '../dataset_cup/encoded_dataset/encoded_df_cup_blind_ts_input')

