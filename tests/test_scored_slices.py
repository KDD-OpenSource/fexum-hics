from unittest import TestCase
from hics.bivariate_correlation import ScoredSlices
import numpy as np
import pandas as pd


class Test_scored_slices(TestCase):
	def test_categorical(self):
		result = np.array([[1, 0.5, 0.5], [0.5, 1, 0.25], [0.5, 0.25, 1]])

		continuous = ['X1', 'X2']
		categorical = [{'name' : 'X3', 'values' : ['a', 'b', 'c', 'd']}, {'name' : 'X4', 'values' : ['a', 'b', 'c', 'd']}]

		slices = {'features' : {
			'X3' : [[1, 1, 1, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
			'X4' : [[1, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1]],
			'X1' : {'from_value' : [0.5, 0, 0], 'to_value' : [1, 0.5, 0.75]},
			'X2' : {'from_value' : [0.5, 0, 0], 'to_value' : [1, 0.75, 0.5]}},
			'scores' : [1, 2, 3]}

		scored_slices = ScoredSlices(categorical, continuous, 2, 0.1)
		scored_slices.add_slices(slices)
		scored_slices.reduce_slices()

		self.assertTrue(len(scored_slices.continuous.iloc[0].index) == 2)
		self.assertTrue(np.all(np.array(scored_slices.continuous['X1', :, 'from_value']) == np.array([0, 0.5])))
		self.assertTrue(np.all(np.array(scored_slices.continuous['X1', :, 'to_value']) == np.array([0.75, 1])))
		self.assertTrue(np.all(np.array(scored_slices.continuous['X2', :, 'from_value']) == np.array([0, 0.5])))
		self.assertTrue(np.all(np.array(scored_slices.continuous['X2', :, 'to_value']) == np.array([0.5, 1])))
		self.assertTrue(np.all(np.array(scored_slices.categorical['X3']) == np.array([[1, 0, 1, 0], [1, 1, 1, 0]])))
		self.assertTrue(np.all(np.array(scored_slices.categorical['X4']) == np.array([[0, 0, 1, 1], [1, 1, 1, 0]])))


if __name__ == '__main__':
	unittest.main()