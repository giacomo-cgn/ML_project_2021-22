from NeuralNetwork import *
from validation import kfold_cross_validation
from data_visualization import multiplot_visualization
from save_results import save_params
from joblib import Parallel, delayed
from os import cpu_count


def grid_search_parallel(x_input, y_target, layer_arr_list, epoch_limit, learning_rate_arr=[0.1],
                         activ_func_arr=["sigmoid"],
                         activ_func_out_arr=["sigmoid"], loss_arr=['MSE'],
                         alpha_arr=[0.01], lambda_arr=[0.01], batches=1, reg_type_arr=["l1"], rand=False,
                         lr_type=["fixed"],
                         tau_arr=[0], accuracy_flag=False, num_folds=5):
    """
    This function performs a grid-search with the different combinations of hyper-parameters for model selection.
    Specifically, it performs several k-fold cross validations in parallel (on different CPUs) and
    returns for each cross validation, i.e. for each set of hyper-parameters, the mean loss of the folds of
    validation and the losses on the training folds such that the loss curve can be visualized with graphs.

    :param x_input: input data
    :param y_target: target data
    :param layer_arr_list: list of topologies of the neural network
    :param epoch_limit: maximum number of epochs to avoid diverging
    :param learning_rate_arr: list of different etas
    :param activ_func_arr: list of activation functions for hidden layers
    :param activ_func_out_arr: list of activation functions for the output layer
    :param loss_arr: list of loss functions
    :param alpha_arr: list of momentum coefficients
    :param lambda_arr: list of regularization lambdas
    :param batches: number of batches in which to split the dataset in training
    :param reg_type_arr: list of regularization types
    :param rand: boolean parameter, if true the NN is trained as a Randomized Neural Network
    :param lr_type: specifies if learning rate is fixed or variable
    :param tau_arr: list of tau values for variable learning rates
    :param accuracy_flag: specifies if accuracy must be computed or not
    :param num_folds: number of folds for k-fold cross validation


    :return: results_params: list of dictionaries of hyper-parameters for the Neural Network
    :return: results_val_loss: list of mean loss on every validation fold corresponding to a specific set of
                                hyper-parameters
    :return: results_loss_arr: list of losses on training folds corresponding to a specific set of
                                hyper-parameters
    """

    parameters_list = []

    for layer_arr in layer_arr_list:
        for learning_rate in learning_rate_arr:
            for activ_func in activ_func_arr:
                for activ_func_out in activ_func_out_arr:
                    for loss in loss_arr:
                        for momentum in alpha_arr:
                            for lambda_reg in lambda_arr:
                                for tau in tau_arr:
                                    for reg_type in reg_type_arr:
                                        params = dict()
                                        params["layer_arr"] = layer_arr
                                        params["learning_rate"] = learning_rate
                                        params["activ_func"] = activ_func
                                        params["activ_func_out"] = activ_func_out
                                        params["loss"] = loss
                                        params["momentum"] = momentum
                                        params["lambda_reg"] = lambda_reg
                                        params["reg_type"] = reg_type
                                        params["rand"] = rand
                                        params["lr_type"] = lr_type
                                        params["tau"] = tau

                                        parameters_list.append(params)

    results = []

    results.append(Parallel(n_jobs=cpu_count(), verbose=50)(delayed(kfold_cross_validation)(
        num_folds, x_input, y_target, parameters_list[i], batches, epoch_limit, accuracy_flag
    ) for i in range(len(parameters_list))))

    results_params, results_val_loss, results_loss_arr = zip(*results[0])

    return results_params, results_val_loss, results_loss_arr


# Choose dataset
x = input("Choose a dataset: 1. Monk1, 2. Monk2, 3. Monk3, 4. Cup: ")
x = int(x)


if x == 1:
    input_data = np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk1_input_training.npy"))
    # Square parenthesis were added to avoid losing array's dimensions for Monk datasets with only one output
    target = np.array([np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk1_target_training.npy"))])

    # Shuffle
    p = np.random.permutation(input_data.shape[1])
    input_data = input_data[:, p]
    target = target[:, p]
elif x == 2:
    input_data = np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk2_input_training.npy"))
    target = np.array([np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk2_target_training.npy"))])
elif x == 3:
    input_data = np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk3_input_training.npy"))
    target = np.array([np.transpose(np.load("../dataset_monk/encoded_dataset/encoded_df_monk3_target_training.npy"))])
elif x == 4:
    input_data = np.transpose(np.load("../dataset_cup/encoded_dataset/encoded_df_cup_input_training.npy"))
    target = np.transpose(np.load("../dataset_cup/encoded_dataset/encoded_df_cup1_target_training.npy"))

    # Split into Development set and Internal Test set
    input_split = np.split(input_data, [int(np.shape(input_data)[1] * 0.7), int(np.shape(input_data)[1])], axis=1)
    target_split = np.split(target, [int(np.shape(input_data)[1] * 0.7), int(np.shape(input_data)[1])], axis=1)

    # Development set
    input_data = input_split[0]
    target = target_split[0]

    # Internal Test set
    input_test = input_split[1]
    target_test = target_split[1]


# Flag specifying if accuracy must be computed or not
accuracy_flag = True
is_monk = True
if x == 4:
    accuracy_flag = False
    is_monk = False

# Hyper-parameters for Monk datasets
if x == 1 or x == 2 or x == 3:
    layer_arr_list = [[17, 5, 1]]
    max_epochs = 500
    lr_arr = [0.2, 0.4, 0.6]
    act_func_arr = ["sigmoid"]
    act_func_out_arr = ["sigmoid"]
    loss_arr = ["MSE"]
    momentum_arr = [0.5,0.7,0.8]
    lambda_arr = [0.0, 0.001]
    batches = 1
    reg_type_arr = ["l1", "l2"]
    rand = False
    lr_type = "fixed"
    tau_arr = [0]

else:
    # Hyper-parameters for CUP
    layer_arr_list = [[10, 50, 10, 2], [10, 8, 5, 3, 2], [10, 10, 5, 2]]
    max_epochs = 800
    lr_arr = [0.001, 0.1, 0.005]
    act_func_arr = ["leakyrelu"]
    act_func_out_arr = ["identity"]
    loss_arr = ["MEE"]
    momentum_arr = [0.0, 0.5, 0.7]
    lambda_arr = [1e-3, 1e-2]
    batches = 16
    tau_arr = [2000, 5000, 10000]
    lr_type = "linear_decay"
    reg_type_arr = ["l1", "l2"]
    rand = False

num_folds = 5

params_results, val_losses, losses = grid_search_parallel(input_data, target, layer_arr_list, max_epochs,
                                                          lr_arr, act_func_arr, act_func_out_arr, loss_arr,
                                                          momentum_arr, lambda_arr, batches,
                                                          reg_type_arr, rand, lr_type, tau_arr, accuracy_flag,
                                                          num_folds)

print("Validation losses: ", val_losses)
print("Index of minimum validation loss is ", np.argmin(val_losses), "with value", np.min(val_losses))

# Results Graphs
for j, l in enumerate(losses):
    multiplot_visualization(l, 'loss', ['epochs', 'loss'],
                            [params_results[j] for x in l], (3, 3))
    if len(losses) > 5:
        break


# Visualize graphs of the models
y = 0
while True:
    y = int(input('Choose index of graph to visualize (type -1 to break): '))
    if y == -1:
        break
    print(params_results[y])

    try:
        multiplot_visualization(losses[y], 'loss', ['epochs', 'loss'],
                                [params_results[y] for x in losses[y]],
                                (3, 3))
    except Exception:
        print("error")

    z = input("Save the model params?")

    if z == "y" or z == "Y":
        params = params_results[y]
        val_loss = val_losses[y]
        params["batches"] = batches

        # Saves params and cross validation loss on file
        save_params(val_loss, params, dataset="monk" + str(x) if is_monk else "cup")
