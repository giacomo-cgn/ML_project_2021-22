import numpy as np
import json


def load_results(path, accuracy_flag = False):
    """
    This function loads losses and accuracies for both Training and Test set, the hyper-parameters who generated
    them and the weight structure
    
    :param path: path in which the files to load from are saved
    :param accuracy_flag: if there are any accuracies to laod

    :return: accuracy_for_epochs_tr: accuracies on Training set
    :return: accuracy_for_epochs_ts: accuracies on Test set
    :return: loss_for_epochs_tr: losses on Training set
    :return: loss_for_epochs_ts: losses on Test set
    :return: params: hyper-parameters to load
    :return: weight_struct_arr: weight structure of the model to load
    """
    accuracy_for_epochs_tr = []
    accuracy_for_epochs_ts = []
    if accuracy_flag:
        accuracy_for_epochs_tr = np.load(path + "acc_for_epochs_tr.npy", allow_pickle=True)
        accuracy_for_epochs_ts = np.load(path + "acc_for_epochs_ts.npy", allow_pickle=True)
    loss_for_epochs_tr = np.load(path + "loss_for_epoch_tr.npy", allow_pickle=True)
    loss_for_epochs_ts = np.load(path + "loss_for_epoch_ts.npy", allow_pickle=True)
    val_loss = np.load(path + "cross_val_loss.npy", allow_pickle=True)
    with open(path + "params.json", "r") as read_file:
        params = json.load(read_file)
    weight_struct_arr = np.load(path + "layer_weight_struct_arr.npy", allow_pickle=True)
    weight_bias_arr = np.load(path + "layer_weight_bias_arr.npy", allow_pickle=True)

    return accuracy_for_epochs_tr, accuracy_for_epochs_ts, loss_for_epochs_tr, loss_for_epochs_ts, params, \
           weight_struct_arr, weight_bias_arr, val_loss

