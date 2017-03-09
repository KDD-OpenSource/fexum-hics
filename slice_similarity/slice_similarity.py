import numpy as np
import pandas as pd


def continuous_similarity_matrix(dfs):
	length = len(dfs.major_axis)
	volumn = np.zeros((length, length))
	overlap = np.zeros((length, length))
	
	for index, df in dfs.iteritems():
		end = np.array([df.end] * length)
		start = np.array([df.start] * length)
		min_range = np.minimum((end - start), (end - start).T)
		min_overlap_range = np.minimum((end.T - start), (end.T - start).T)
		
		if not overlap.any(): 
			overlap = np.minimum(min_range, min_overlap_range)
		else:
			overlap = overlap * np.minimum(min_range, min_overlap_range)
		overlap[overlap < 0] = 0
		
		if not volumn.any():
			volumn = end - start
		else:
			volumn = volumn * (end - start)
		
	min_volumn = np.maximum(volumn, volumn.T)
	
	min_volumn[min_volumn <= 0] = 1
	overlap[overlap < 0] = 0

	return overlap/min_volumn


def categorical_similarity_matrix(dfs):
	length = len(dfs.major_axis)
	volumn = np.zeros((length, length))
	overlap = np.zeros((length, length))

	for index, df in dfs.iteritems():

		data_array = np.array([np.array(df).tolist()] * len(df))
		size_array = np.apply_along_axis(lambda x: (x*1).sum(), 2, data_array)

		current_overlap = np.apply_along_axis(lambda x: (x*1).sum(), 2, np.logical_and(data_array, data_array.transpose(1, 0, 2)))

		if not overlap.any(): 
			overlap = current_overlap
		else:
			overlap = overlap * current_overlap
	
		if not volumn.any(): 
			volumn = size_array
		else:
			volumn = volumn * size_array

	min_volumn = np.minimum(volumn, volumn.T)
	min_volumn[min_volumn == 0] = 1

	return overlap/min_volumn





