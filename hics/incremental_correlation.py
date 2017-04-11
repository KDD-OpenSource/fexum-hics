import numpy as np
import pandas as pd
import sys
from random import randint
from hics.contrast_meassure import HiCS
from hics.scored_slices import ScoredSlices
from hics.result_storage import DefaultResultStorage


class IncrementalCorrelation:
	def __init__(self, data, target, result_storage, iterations = 10, alpha = 0.1, drop_discrete = False):
		self.subspace_contrast = HiCS(data, alpha, iterations)

		self.target = target
		self.features = [str(ft) for ft in data.columns.values if str(ft) != target]

		if drop_discrete:
			self.features = [ft for ft in self.features if self.subspace_contrast.types[ft] != 'discrete']

		self.result_storage = result_storage

	def _update_relevancy_table(self, new_relevancies):
		current_relevancies = self.result_storage.get_relevancies()

		new_index = [index for index in new_relevancies.index if not index in current_relevancies.index]

		current_relevancies = current_relevancies.append(pd.DataFrame(data = 0, index = new_index, columns = current_relevancies.columns))

		current_relevancies.loc[new_relevancies.index, 'relevancy'] = (current_relevancies.iteration / (current_relevancies.iteration + new_relevancies.iteration)) * current_relevancies.relevancy + (new_relevancies.iteration / (current_relevancies.iteration + new_relevancies.iteration)) * new_relevancies.relevancy
		current_relevancies.loc[new_relevancies.index, 'iteration'] += new_relevancies.iteration

		self.result_storage.update_relevancies(current_relevancies)

	def _update_redundancy_table(self, new_weights, new_redundancies):
		current_redundancies, current_weights = self.result_storage.get_redundancies()

		current_redundancies = (current_weights / (new_weights + current_weights)) * current_redundancies + (new_weights / (new_weights + current_weights)) * new_redundancies
		current_redundancies.fillna(0, inplace = True)
		current_weights += new_weights

		self.result_storage.update_redundancies(current_redundancies)

	def _update_slices(self, new_slices):
		current_slices = self.result_storage.get_slices()

		for feature_set, new_slics in new_slices.items():
			if not feature_set in current_slices:
				current_slices[feature_set] = new_slices[feature_set]

			else:
				current_slices[feature_set].add_slices(new_slices[feature_set])

			current_slices[feature_set].reduce_slices()

		self.result_storage.update_slices(current_slices)

	def _relevancy_dict_to_df(self, new_scores):
		indices = [tuple(index) for index in new_scores]
		scores = [score for index, score in new_scores.items()]
		new_relevancies = pd.DataFrame(data = scores ,index = indices)
		return new_relevancies

	def _add_slices_to_dict(self, subspace, slices, slices_store):
		subspace_tuple = tuple(sorted(subspace))
		if not subspace_tuple in slices_store:
			categorical = [{'name' : ft, 'values' : self.subspace_contrast.values(ft)} for ft in subspace if self.subspace_contrast.type(ft) == 'categorical']
			continuous = [ft for ft in subspace if self.subspace_contrast.type(ft) == 'continuous']
			slices_store[subspace_tuple] = ScoredSlices(categorical, continuous)

		slices_store[subspace_tuple].add_slices(slices)
		return slices_store

	def update_bivariate_relevancies(self, runs = 5):
		new_slices = {}
		new_scores = {tuple(feature) : {'relevancy' : 0, 'iteration' : 0} for feature in self.features}

		for i in range(runs):
			for feature in self.features:
				subspace_tuple = tuple(feature)
				subspace_score, subspace_slices = self.subspace_contrast.calculate_contrast([feature], self.target, True)

				new_slices = self._add_slices_to_dict([feature], subspace_slices, new_slices)

				new_scores[subspace_tuple]['relevancy'] += subspace_score
				new_scores[subspace_tuple]['iteration'] += 1

		new_relevancies = self._relevancy_dict_to_df(new_scores)
		new_relevancies.relevancy /= new_relevancies.iteration

		self._update_relevancy_table(new_relevancies)
		self._update_slices(new_slices)
				
	def update_multivariate_relevancies(self, fixed_features = [], k = 5, runs = 5):
		new_slices = {}
		new_scores = {}

		feature_list = [feature for feature in self.features if not feature in fixed_features]
		max_k = k - len(fixed_features)

		for i in range(runs):
			subspace = fixed_features[:]
			
			if 0 < max_k:
				end_index = randint(1, max_k + 1)
				subspace += np.random.permutation(feature_list)[0:end_index].tolist()

			subspace_tuple = tuple(sorted(subspace))
			subspace_score, subspace_slices = self.subspace_contrast.calculate_contrast(subspace, self.target, True)
			
			if not subspace_tuple in new_scores:	
				new_scores[subspace_tuple] = {'relevancy' : 0, 'iteration' : 0}

			new_scores[subspace_tuple]['relevancy'] += subspace_score
			new_scores[subspace_tuple]['iteration'] += 1
			
			new_slices = self._add_slices_to_dict(subspace, subspace_slices, new_slices)

		new_relevancies = self._relevancy_dict_to_df(new_scores)
		new_relevancies.relevancy /= new_relevancies.iteration

		self._update_relevancy_table(new_relevancies)
		self._update_slices(new_slices)

	def update_redundancies(self, k = 5, runs = 10):
		new_redundancies = pd.DataFrame(data = 0, columns = self.features, index = self.features)
		new_weights = pd.DataFrame(data = 0, columns = self.features, index = self.features)

		for i in range(runs):
			number_features = randint(1, k)
			selected_features = np.random.permutation(self.features)[0:number_features + 1].tolist()
			target = selected_features[number_features]
			subspace = selected_features[0:number_features]
			
			score = self.subspace_contrast.calculate_contrast(subspace, target, False)

			for ft in subspace:
				new_redundancies.loc[ft, target] = new_redundancies.loc[ft, target] + score 
				new_weights.loc[ft, target] = new_weights.loc[ft, target] + 1 
				new_redundancies.loc[target, ft] = new_redundancies.loc[target, ft] + score
				new_weights.loc[target, ft] = new_weights.loc[target, ft] + 1

		new_redundancies = new_redundancies / new_weights
		new_redundancies.fillna(0, inplace = True)

		self.result_storage.update_redundancies(new_redundancies, new_weights)



