import numpy as np


def encode_1_hot(data):
	"""
	This function applies one-hot-encoding to the data passed thr

	:param data: data to be encoded, patterns are on the rows

	:return: data_encoded: the encoded data
	"""

	bits_for_column = [3, 3, 2, 3, 4, 2]
	num_cols = sum(bits_for_column)
	num_patterns = data.shape[0]
	data_encoded = np.zeros((num_patterns, num_cols), dtype=np.intc)

	for row_index, row in data.iterrows():
		for col_Index in range(0, len(row)):

			if col_Index == 0:
				data_encoded[row_index][row[col_Index]-1] = 1
			else:
				matrix_col_index = row[col_Index] + sum(bits_for_column[0:col_Index]) - 1
				data_encoded[row_index][matrix_col_index] = 1
	return data_encoded

