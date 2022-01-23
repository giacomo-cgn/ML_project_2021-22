from load_results import load_results
from NeuralNetwork import *

paths = ["2022-01-14_16-54", "2022-01-16_19-18",
         "2022-01-18_16-55", "2022-01-20_14-53"]


def predictEnsemble(input_test, target_test=[]):

    list_nn = []

    tot_loss_tr = 0
    tot_loss_vl = 0

    for i in paths:

        # Load models
        _, _, loss_tr, loss_ts, params, weight_struct_arr, weight_bias_arr, val_loss = load_results("../results/cup/" + i + "/", False)

        nn = NeuralNetwork(params["layer_arr"], params["learning_rate"], params["activ_func"],
                               params["activ_func_out"], params["loss"], params["momentum"], params["lambda_reg"],
                               params["reg_type"], params["rand"], params["lr_type"], params["tau"])

        nn.upload_layer_weights(weight_struct_arr, weight_bias_arr)
        list_nn.append(nn)

        # Compute TR loss
        tot_loss_tr += loss_tr[-1]

        # Compute VL loss
        tot_loss_vl += val_loss

    # If we have a target
    if len(target_test) > 0:
        tot_output = 0
        j = 0
        for i in paths:
            # Compute TS loss and predictions
            h_output, loss_ts_2 = list_nn[j].predict(input_test, target_test)
            tot_output = tot_output + h_output
            j = j + 1

        # Compute averages
        tot_output = tot_output / len(paths)
        tot_loss_tr = tot_loss_tr / len(paths)
        tot_loss_vl = tot_loss_vl / len(paths)

        # Loss estimate for ensemble
        tot_loss_ts = list_nn[0].loss_function(tot_output, target_test) / np.shape(target_test)[1]

        print("Loss TR: ", tot_loss_tr)
        print("Loss VL: ", tot_loss_vl)
        print("Loss TS: ", tot_loss_ts)

    else:

        tot_output = 0
        j = 0

        for i in paths:
            h_output, _ = list_nn[j].predict(input_test)
            tot_output = tot_output + h_output
            j = j + 1

        tot_output = tot_output / len(paths)

    return tot_output



