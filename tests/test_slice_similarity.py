from unittest import TestCase
from hics.slice_similarity import continuous_similarity_matrix, categorical_similarity_matrix
import numpy as np
import pandas as pd


class Test_slice_similarity(TestCase):
	def test_categorical(self):
		result = np.array([[1, 0.5, 0.5], [0.5, 1, 0.25], [0.5, 0.25, 1]])

		categorical = pd.Panel({
			'X3' : pd.DataFrame({'a' : [1, 0, 1], 'b' : [1, 0, 0], 'c' : [1, 1, 1], 'd' : [0, 1, 0]}),
			'X4' : pd.DataFrame({'a' : [1, 1, 0], 'b' : [1, 0, 0], 'c' : [1, 1, 1], 'd' : [0, 0, 1]})
			})

		similarity = categorical_similarity_matrix(categorical)
		self.assertTrue(np.all(similarity == result))

	def test_continuous(self):
		result = np.array([[1, 0, 0], [0, 1, 2/3], [0, 2/3, 1]])

		continuous = pd.Panel({
			'X3' : pd.DataFrame({'from_value' : [0.5, 0, 0], 'to_value' : [1, 0.5, 0.75]}),
			'X4' : pd.DataFrame({'from_value' : [0.5, 0, 0], 'to_value' : [1, 0.75, 0.5]})
			})

		similarity = continuous_similarity_matrix(continuous)
		self.assertTrue(np.all(similarity == result))


if __name__ == '__main__':
	unittest.main()