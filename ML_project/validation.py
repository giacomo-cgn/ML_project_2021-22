from NeuralNetwork import *
import numpy as np
from MONK_data_processing import monk_output_processing


def kfold_cross_validation(n_folds, x_input, y_target, params, batches, epoch_limit, accuracy_flag=False):
    """
    This function performs k-fold cross validation on a Neural Network created with the set of hyper-parameters passed
    and returns the estimates useful for model selection

    :param n_folds: number of folds to split the input and target in
    :param x_input: input data
    :param y_target: target data
    :param params: dictionary of hyper-parameters for the Neural Network
    :param batches: number of batches in which to split the dataset in training
    :param epoch_limit: maximum number of epochs to avoid diverging
    :param accuracy_flag: specifies if accuracy must be computed or not

    :return: params: dictionary of hyper-parameters for the Neural Network
    :return: tot_val_loss: mean loss on every validation fold
    :return: fold_loss_arr: loss on training folds
    """

    # Split folds
    input_folds = np.array_split(x_input, n_folds, axis=1)
    target_folds = np.array_split(y_target, n_folds, axis=1)

    tot_val_loss = 0
    tot_val_accuracy = 0
    fold_loss_arr = []
    fold_valloss_arr = []
    val_accuracy = []
    overflow_check = False

    for k in range(n_folds):
        # Training folds
        tr_set_input_fold = [x for i, x in enumerate(input_folds) if i != k]
        tr_set_target_fold = [x for i, x in enumerate(target_folds) if i != k]
        # Merge training folds
        tr_set_input = np.concatenate(tr_set_input_fold, axis=1)
        tr_set_target = np.concatenate(tr_set_target_fold, axis=1)
        tr_set_target = np.reshape(tr_set_target, (np.shape(y_target)[0], np.shape(tr_set_input)[1]))

        # Validation fold
        val_set_input = input_folds[k]
        val_set_target = target_folds[k]

        # Create and train NN
        nn = NeuralNetwork(params["layer_arr"], params["learning_rate"], params["activ_func"],
                           params["activ_func_out"], params["loss"], params["momentum"], params["lambda_reg"],
                           params["reg_type"], params["rand"], params["lr_type"], params["tau"])

        _, nn_result_loss_arr, nn_result_acc_arr, nn_result_valloss_arr, _, _ = nn.train(tr_set_input, tr_set_target, epoch_limit, batches
         , val_input=val_set_input, val_target=val_set_target, accuracy_flag=accuracy_flag) # Validation Early Stopping

        if len(nn_result_loss_arr) < 1:  #overflow check
            print('Overflow during cross validation detected')
            overflow_check = True
            break

        fold_loss_arr.append(nn_result_loss_arr)
        fold_valloss_arr.append(nn_result_valloss_arr)

        h_output, val_loss = nn.predict(val_set_input, val_set_target)

        if accuracy_flag:
            # Compute Validation Accuracy
            h_output = monk_output_processing(h_output[0])
            val_accuracy.append(sklearn.metrics.accuracy_score(val_set_target[0], h_output))

        tot_val_loss = tot_val_loss + val_loss
        tot_val_accuracy = np.sum(val_accuracy)

    tot_val_loss = tot_val_loss / n_folds
    if overflow_check:
        # If overflow occurred during one of the training
        print('tot validation accuracy, loss, params: overflowed training, ', str(params))
        tot_val_loss = math.inf
    else:
        print('tot validation accuracy, loss, params:', tot_val_accuracy / n_folds, tot_val_loss, str(params))

    return params, tot_val_loss, fold_loss_arr
