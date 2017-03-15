from unittest import TestCase
from hics.contrast_meassure import HiCS
import numpy as np
import pandas as pd


class Test_HiCS(TestCase):
	alpha = 0.1
	iterations = 100

	def test_cashed_marginal_distribution(self):
		correct_result = pd.DataFrame({'value' : [1, 2, 3], 'count' : [1, 2, 5], 'probability' : [0.125, 0.25, 0.625]})
		dataset = pd.DataFrame({'test_marginal' : [2, 2, 1, 3, 3, 3, 3, 3]})
		test_HiCS = HiCS(dataset, self.alpha, self.iterations)
		dist = test_HiCS.cashed_marginal_distribution('test_marginal')
		self.assertTrue(dist.equals(correct_result))

	def test_cashed_sorted_indices(self):
		correct_result = np.array([2, 1, 0])
		dataset = pd.DataFrame({'to_sort' : [10, 5, 0]})
		test_HiCS = HiCS(dataset, self.alpha, self.iterations)
		sorted_index = test_HiCS.cashed_sorted_indices('to_sort')
		self.assertTrue(np.all(correct_result == sorted_index))

	def test_calculate_conditional_distribution(self):
		correct_result = pd.DataFrame({'value' : [0, 1, 2], 'count' : [3, 1, 1], 'probability' : [0.6, 0.2, 0.2]})
		dataset = pd.DataFrame({'target' : [1, 1, 1, 0, 0, 0, 2, 2, 2], 'feature' : [0, 1, 2, 3, 4, 5, 6, 7, 8]})
		condition = {'feature' : 'feature', 'indices' : [2, 3, 4, 5, 6], 'from_value' : 2, 'to_value' : 6}
		target = 'target'
		test_HiCS = HiCS(dataset, self.alpha, self.iterations)
		cond_dist = test_HiCS.calculate_conditional_distribution([condition], target)
		self.assertTrue(cond_dist.equals(correct_result))

	def test_create_discrete_condition(self):
		dataset = pd.DataFrame({'feature' : [1]*20 + [2]*3 + [0]*1 })
		test_HiCS = HiCS(dataset, self.alpha, self.iterations)
		number_instances = round(len(dataset)*self.alpha)
		condition = test_HiCS.create_discrete_condition('feature', number_instances)
		
		self.assertTrue('feature' in condition)
		self.assertTrue('values' in condition)
		self.assertTrue('indices' in condition)

		chosen_indices = dataset.loc[dataset['feature'].isin(condition['values']), : ].index.values 

		self.assertTrue(number_instances <= len(condition['indices']))			#test only works that way when using 1 dimension
		self.assertTrue(pd.Series(condition['indices']).isin(chosen_indices).all())	
		self.assertTrue(pd.Series(chosen_indices).isin(condition['indices']).all())


	def test_create_continuous_condition(self):
		dataset = pd.DataFrame({'feature' : np.around(a = np.random.rand(100), decimals = 2)})
		test_HiCS = HiCS(dataset, self.alpha, self.iterations)
		number_instances = round(len(dataset)*self.alpha)
		condition = test_HiCS.create_continuous_condition('feature', number_instances)

		self.assertTrue('feature' in condition)
		self.assertTrue('indices' in condition)
		self.assertTrue('from_value' in condition)
		self.assertTrue('to_value' in condition)

		chosen_indices = dataset.loc[np.logical_and(dataset['feature'] >= condition['from_value'], dataset['feature'] <= condition['to_value']), : ].index.values

		self.assertTrue(number_instances <= len(condition['indices']))			#test only works that way when using 1 dimension
		self.assertTrue(pd.Series(condition['indices']).isin(chosen_indices).all())	
		self.assertTrue(pd.Series(chosen_indices).isin(condition['indices']).all())


if __name__ == '__main__':
	unittest.main()