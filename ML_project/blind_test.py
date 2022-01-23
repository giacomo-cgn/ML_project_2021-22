from ensemble import predictEnsemble
from NeuralNetwork import *
import pandas as pd

blind_ts_input = np.transpose(np.load("../dataset_cup/encoded_dataset/encoded_df_cup_blind_ts_input.npy"))

# Predicts on the blind test set
output = predictEnsemble(blind_ts_input)

# Saves results in csv format
df = pd.DataFrame(output.T)
df.index = np.arange(1, len(df)+1)
df.to_csv("../results/cup/blind/blind.csv")
