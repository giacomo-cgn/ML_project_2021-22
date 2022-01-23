from datetime import datetime
import numpy as np
import os
import json


def save_results(loss_for_epochs_tr=[], loss_for_epochs_ts=[], acc_for_epochs_tr=[], acc_for_epochs_ts=[],
                 weight_struct_arr=[], weight_bias_arr=[], data_path=""):
    """
    This function saves a model's loss and accuracy values during the epochs for both Training and Test set,
    and the weight structure to a file

    :param loss_for_epochs_tr: loss for Training set
    :param loss_for_epochs_ts: loss for Test set
    :param acc_for_epochs_tr: accuracy for Training set
    :param acc_for_epochs_ts: accuracy for Test set
    :param weight_struct_arr: weights of the network
    :param weight_bias_arr: bias of the layers of the network
    :param data_path: path of the file in which to save the informations
    """

    if data_path:
        # Existing folder path
        folder_path = data_path + "/"
    else:
        # Create folder path with current datetime for test
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
        folder_path = "../results/test/" + date_str + "/"
        os.makedirs(folder_path, exist_ok=True)
    print(folder_path)

    np.save(folder_path + "loss_for_epoch_tr.npy", loss_for_epochs_tr)
    np.save(folder_path + "loss_for_epoch_ts.npy", loss_for_epochs_ts)
    if acc_for_epochs_tr:
        np.save(folder_path + "acc_for_epochs_tr.npy", acc_for_epochs_tr)
    if acc_for_epochs_ts:
        np.save(folder_path + "acc_for_epochs_ts.npy", acc_for_epochs_ts)

    np.save(folder_path + "layer_weight_struct_arr.npy", weight_struct_arr)
    np.save(folder_path + "layer_weight_bias_arr.npy", weight_bias_arr)

    return


def save_params(cross_val_loss=[], params="", dataset=""):
    """
    This function saves hyper-parameters and validation loss of a model to a file

    :param cross_val_loss: value of the resulting average validation loss
    :param params: hyper-parameters of the model
    :param dataset: name of the dataset
    """
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = "../results/" + dataset

    print(path)
    folder_path = str(path) + "/" + date_str + "/"
    print(folder_path)
    os.makedirs(folder_path, exist_ok=True)

    np.save(folder_path + "cross_val_loss.npy", cross_val_loss)
    with open(folder_path + 'params.json', 'w') as outfile:
        json.dump(params, outfile)
    return
