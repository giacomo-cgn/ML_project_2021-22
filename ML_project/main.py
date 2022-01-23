from load_results import load_results
from data_visualization import visualize


x = input("Choose a dataset: 1. Monk1, 2. Monk2, 3. Monk3, 4. Monk3+regularization, 5. Cup: ")
x = int(x)

# Loads the best models for the corresponding dataset and visualizes their learning and accuracy curves
if x == 1:

    acc_tr, acc_ts, loss_tr, loss_ts, params, _, _ = load_results("../results/monk1/2022-01-19_22-19/",
                                                                               True)

    visualize([loss_tr, loss_ts], ['Training set', 'Test set'], ['Epochs', 'Loss'])
    visualize([acc_tr, acc_ts], ['Training set', 'Test set'], ['Epochs', 'Accuracy'])
    print('Final Losses: TR ', loss_tr[-1], 'TS ', loss_ts[-1])
    print('Final Accuracies: TR ', acc_tr[-1], 'TS ', acc_ts[-1])

elif x == 2:

    acc_tr, acc_ts, loss_tr, loss_ts, params, _, _ = load_results("../results/monk2/2022-01-19_22-27/",
                                                                               True)

    visualize([loss_tr, loss_ts], ['Training set', 'Test set'], ['Epochs', 'Loss'])
    visualize([acc_tr, acc_ts], ['Training set', 'Test set'], ['Epochs', 'Accuracy'])
    print('Final Losses: TR ', loss_tr[-1], 'TS ', loss_ts[-1])
    print('Final Accuracies: TR ', acc_tr[-1], 'TS ', acc_ts[-1])

elif x == 3:

    acc_tr, acc_ts, loss_tr, loss_ts, params, _, _ = load_results("../results/monk3/2022-01-19_22-34/",
                                                                               True)

    visualize([loss_tr, loss_ts], ['Training set', 'Test set'], ['Epochs', 'Loss'])
    visualize([acc_tr, acc_ts], ['Training set', 'Test set'], ['Epochs', 'Accuracy'])
    print('Final Losses: TR ', loss_tr[-1], 'TS ', loss_ts[-1])
    print('Final Accuracies: TR ', acc_tr[-1], 'TS ', acc_ts[-1])

elif x == 4:

    acc_tr, acc_ts, loss_tr, loss_ts, params, _, _ = load_results("../results/monk3/2022-01-19_22-53(reg)/",
                                                                               True)

    visualize([loss_tr, loss_ts], ['Training set', 'Test set'], ['Epochs', 'Loss'])
    visualize([acc_tr, acc_ts], ['Training set', 'Test set'], ['Epochs', 'Accuracy'])
    print('Final Losses: TR ', loss_tr[-1], 'TS ', loss_ts[-1])
    print('Final Accuracies: TR ', acc_tr[-1], 'TS ', acc_ts[-1])

elif x == 5:

    _, _, loss_tr, loss_ts, params, weight_struct_arr, weight_bias_arr = load_results("../results/cup/2022-01-18_16-55/",
                                                                               False)

    visualize([loss_tr, loss_ts], ['Training set', 'Test set'], ['Epochs', 'Loss'])
    print('Final Losses: TR ', loss_tr[-1], 'TS ', loss_ts[-1])



