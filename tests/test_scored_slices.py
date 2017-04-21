from unittest import TestCase
from hics.bivariate_correlation import ScoredSlices
import numpy as np
import pandas as pd
import json


class Test_scored_slices(TestCase):
	def test_reduce_slices(self):
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

	def test_to_output(self):
		result = [{'deviation': 2.5, 'features': {'ft_con_2': {'from': 8.0, 'to': 9.0}, 'ft_con_1': {'from': 2.0, 'to': 3.0}, 'ft_cat_2': ['1'], 'ft_cat_1': ['a', 'b']}}]
		result_json = json.dumps(result)

		ft_cat_1 = pd.DataFrame({'a' : [1], 'b' : [1]})
		ft_cat_2 = pd.DataFrame({'1' : [1], '2' : [0]})
		ft_con_1 = pd.DataFrame({'from' : [2], 'to' : [3]})
		ft_con_2 = pd.DataFrame({'from' : [8], 'to' : [9]})
		continuous = {'ft_con_1' : ft_con_1, 'ft_con_2' : ft_con_2}
		categorical = {'ft_cat_1' : ft_cat_1, 'ft_cat_2' : ft_cat_2}
		scores = pd.Series([2.5])
		
		scored_slices = ScoredSlices([], [], 2, 0.1)
		scored_slices.continuous = continuous
		scored_slices.categorical = categorical
		scored_slices.scores = scores
		
		output = scored_slices.to_output()
		output_json = json.dumps(output)

		self.assertTrue(result_json == output_json)


if __name__ == '__main__':
	unittest.main()