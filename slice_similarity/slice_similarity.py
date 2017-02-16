import numpy as np
import pandas as pd


def calculate_similarity_matrix(dfs):
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
		
	max_volumn = np.maximum(volumn, volumn.T)
	
	return overlap/max_volumn





