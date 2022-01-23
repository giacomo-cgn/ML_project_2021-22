import pandas as pd
from oneHotEncoding import encode_1_hot
import numpy as np


def encode_monk(path, save_path_input, save_path_target):
    """
    This function opens the file of the monk dataset, separates the input columns from target,
    encodes them and saves them in local files

    :param path: path of the file where the dataset is located
    :param save_path_input: path of the file where the input data will be saved
    :param save_path_target: path of the file where the target data will be saved
    """
    df = pd.read_table(path, '\s+', header=None, names=['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id'])

    # Extract target column
    target = df['class']
    # Drop target column and Id (not useful a useful feature for Training)
    df.drop(labels=['class', 'id'], axis=1, inplace=True)

    np.set_printoptions(threshold=np.inf)

    # Performs One-Hot Encoding
    encoded_df = encode_1_hot(df)
    np.save(save_path_input, encoded_df)
    target_numpy = target.to_numpy()
    np.save(save_path_target, target_numpy)
    print("File successfully encoded!")
    return


def monk_output_processing(output, threshold=0.50):
    """
    This function takes the output with float values and transforms them in binary values {0, 1}
    depending on a certain threshold

    :param output: output of the Neural Network with float values
    :param threshold: above this threshold the single output value is 1, otherwise it's 0. Default = 0.5

    :return: output: output of the Neural Network with binary values
    """
    for i, o in enumerate(output):
        output[i] = 0 if o < threshold else 1
    return output


# Encode datasets
'''
encode_monk('../dataset_monk/monks-1.train', '../dataset_monk/encoded_dataset/encoded_df_monk1_input_training',
       '../dataset_monk/encoded_dataset/encoded_df_monk1_target_training')

encode_monk('../dataset_monk/monks-1.test', '../dataset_monk/encoded_dataset/encoded_df_monk1_input_test',
       '../dataset_monk/encoded_dataset/encoded_df_monk1_target_test')

encode_monk('../dataset_monk/monks-2.train', '../dataset_monk/encoded_dataset/encoded_df_monk2_input_training',
       '../dataset_monk/encoded_dataset/encoded_df_monk2_target_training')

encode_monk('../dataset_monk/monks-2.test', '../dataset_monk/encoded_dataset/encoded_df_monk2_input_test',
       '../dataset_monk/encoded_dataset/encoded_df_monk2_target_test')

encode_monk('../dataset_monk/monks-3.train', '../dataset_monk/encoded_dataset/encoded_df_monk3_input_training',
       '../dataset_monk/encoded_dataset/encoded_df_monk3_target_training')

encode_monk('../dataset_monk/monks-3.test', '../dataset_monk/encoded_dataset/encoded_df_monk3_input_test',
       '../dataset_monk/encoded_dataset/encoded_df_monk3_target_test')

'''