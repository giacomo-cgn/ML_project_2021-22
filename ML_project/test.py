import numpy as np
from NeuralNetwork import *
from save_results import save_results
from load_results import load_results
from data_visualization import visualize
import json


def test(tr_input, tr_target, test_input, test_target, params, max_epoch, path, accuracy_flag=False):
    """
    Function that lets you build a Neural Network with the selected hyper-parameters and train it while
    computing the loss on the Training set and error on Test set. The resulting NN and estimates can be saved to a file.
    
    :param tr_input: Training set input values
    :param tr_target: Training set target values
    :param test_input: Test set input values
    :param test_target: Test set target values
    :param params: hyper-parameters for the Neural Netwrok
    :param max_epoch: maximum number of epochs to avoid diverging
    :param path: path in which to save the weights and computed estimates of the Neural Network
    :param accuracy_flag: specifies if accuracy must be computed or not
    """
    nn = NeuralNetwork(params["layer_arr"], params["learning_rate"], params["activ_func"],
                       params["activ_func_out"], params["loss"], params["momentum"], params["lambda_reg"],
                       params["reg_type"], params["rand"], params["lr_type"], params["tau"])

    h_output, tr_loss_arr, tr_acc_arr, _, test_loss_arr, test_acc_arr = nn.train(tr_input, tr_target, max_epoch,
                                                                          params["batches"],
                                                                          test_input=test_input,
                                                                          test_target=test_target,
                                                                          accuracy_flag=accuracy_flag)

    # Save models
    z = input("Save model test results?")
    if z == "y":
        # Saves weights of nn neurons
        weight_struct_arr = []
        weight_bias_arr = []

        for layer in nn.layer_arr:
            weight_struct_arr.append(layer.layer_weight_structure)
            weight_bias_arr.append(layer.bias)

        if accuracy_flag:
            save_results(tr_loss_arr, test_loss_arr, tr_acc_arr, test_acc_arr, weight_struct_arr, weight_bias_arr,
                         data_path=path)
        else:
            save_results(tr_loss_arr, test_loss_arr, [], [], weight_struct_arr, weight_bias_arr, data_path=path)


x = input("Choose a dataset: 1. Monk1, 2. Monk2, 3. Monk3, 4. Cup: ")
x = int(x)

# Loads Training Set and Test Set for the appropriate datasets
if x == 1:
    input_tr = np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk1_input_training.npy"))
    # Square parenthesis were added to avoid losing array's dimensions for Monk datasets with only one output
    target_tr = np.array(
        [np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk1_target_training.npy"))])

    input_test = np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk1_input_test.npy"))
    target_test = np.array([np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk1_target_test.npy"))])

    p = np.random.permutation(input_tr.shape[1])
    input_data = input_tr[:, p]
    target = target_tr[:, p]
elif x == 2:
    input_tr = np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk2_input_training.npy"))
    target_tr = np.array(
        [np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk2_target_training.npy"))])

    input_test = np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk2_input_test.npy"))
    target_test = np.array([np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk2_target_test.npy"))])

elif x == 3:
    input_tr = np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk3_input_training.npy"))
    target_tr = np.array(
        [np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk3_target_training.npy"))])

    input_test = np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk3_input_test.npy"))
    target_test = np.array([np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk3_target_test.npy"))])

elif x == 4:
    input_tr = np.transpose(np.load("../dataset_cup/encoded_dataset/encoded_df_cup_input_training.npy"))
    target_tr = np.transpose(np.load("../dataset_cup/encoded_dataset/encoded_df_cup1_target_training.npy"))

    # Split into Training set (70%) and Internal Test set (30%)
    input_split = np.split(input_tr, [int(np.shape(input_tr)[1] * 0.7), int(np.shape(input_tr)[1])], axis=1)
    target_split = np.split(target_tr, [int(np.shape(input_tr)[1] * 0.7), int(np.shape(input_tr)[1])], axis=1)

    # Training set
    input_data = input_split[0]
    target = target_split[0]

    # Internal Test set
    input_test = input_split[1]
    target_test = target_split[1]

# Variable which indicates if the task is a regression or classification task
is_classification_task = True
if x == 4:
    is_classification_task = False
    path = "../results/cup/"

else:
    path = "../results/monk" + str(x) + "/"

folder_name = input("Insert folder name to recover hyparams from (type 0 for default):")

if folder_name != "0":
    path = path + folder_name + "/"
else:
    path = path + "2022-01-19_22-19\/"

# Load hyper-parameters from file
with open(path + "params.json", "r") as read_file:
    params = json.load(read_file)
max_epoch = 800
accuracy_flag = is_classification_task

print("Hyper-parameters loaded: ", params)

test(input_tr, target_tr, input_test, target_test, params, max_epoch, path, accuracy_flag)

try:
    # Load results from file
    accuracy_tr, accuracy_ts, loss_tr, loss_ts, params, _, _, _ = load_results(path, accuracy_flag)

    # Visualize TR and TS loss curves
    visualize([loss_tr, loss_ts], ['Training set', 'Test set'], ['Epochs', 'Loss'])

    # If enabled, visualize TR and TS accuracies
    if accuracy_flag:
        visualize([accuracy_tr, accuracy_ts], ['Training set', 'Test set'], ['Epochs', 'Accuracy'])

except:
    print("No file found!")
