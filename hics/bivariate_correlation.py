import numpy as np
import pandas as pd
import sys
import json
from random import randint
from hics.contrast_meassure import HiCS
from hics.slice_similarity import continuous_similarity_matrix, categorical_similarity_matrix
from hics.slice_selection import select_by_similarity
from math import pow


class ScoredSlices:
	def __init__(self, categorical, continuous, to_keep = 5, threshold = None):
		self.continuous = pd.Panel({feature : pd.DataFrame(columns = ['end', 'start'])
			for feature in continuous})

		self.categorical = pd.Panel({feature['name'] : pd.DataFrame(columns = feature['values'])
			for feature in categorical})

		self.scores = pd.Series()
		self.to_keep = to_keep
		
		if threshold == None:
			self.threshold = pow(0.6, len(categorical) + len(continuous))
		else:
			self.threshold = threshold

	def add_slices(self, slices):
		temp_continuous = {}
		temp_categorical = {}

		self.scores = self.scores.append(pd.Series(slices['scores']), ignore_index = True).sort_values(ascending = False, inplace = False)

		for feature in self.continuous.items.values:
			content = slices['features'][feature]
			temp_continuous[feature] = pd.concat([self.continuous[feature], pd.DataFrame(content)], ignore_index = True)
			temp_continuous[feature] = temp_continuous[feature].loc[self.scores.index, :].reset_index(drop = True)

		for feature in self.categorical.items.values:
			content = slices['features'][feature]
			temp_categorical[feature] = pd.concat([self.categorical[feature], pd.DataFrame(content, columns = self.categorical[feature].columns.values)], ignore_index = True)
			temp_categorical[feature] = temp_categorical[feature].loc[self.scores.index, :].reset_index(drop = True)

		self.scores.reset_index(drop = True, inplace = True)
		self.continuous = pd.Panel(temp_continuous)
		self.categorical = pd.Panel(temp_categorical)

	def select_slices(self, similarity):
		indices = list(range(len(similarity)))
		selected = []

		for i in range(self.to_keep):
			if not indices:
				break

			selected.append(indices[0])
			selection = indices[0]

			indices = [index for index in indices if similarity[selection, index] < self.threshold]

		return selected

	def reduce_slices(self):
		if not self.continuous.empty:
			continuous_similarity = continuous_similarity_matrix(self.continuous)
		else:
			continuous_similarity = np.ones((len(self.scores), len(self.scores)))

		if not self.categorical.empty:
			categorical_similarity = categorical_similarity_matrix(self.categorical)
		else:
			categorical_similarity = np.ones((len(self.scores), len(self.scores)))

		similarity = continuous_similarity * categorical_similarity

		selected = self.select_slices(similarity)

		if not self.categorical.empty:
			self.categorical = self.categorical[:, selected, :]
			self.categorical = pd.Panel({name : content.reset_index(drop = True) 
				for name, content in self.categorical.iteritems()})
		
		if not self.continuous.empty:
			self.continuous = self.continuous[:, selected, :]
			self.continuous = pd.Panel({name : content.reset_index(drop = True) 
				for name, content in self.continuous.iteritems()})

		self.scores = self.scores.loc[selected].reset_index(drop = True)


class IncrementalBivariateCorrelation:
	def __init__(self, data, target, iterations = 10, alpha = 0.1, drop_discrete = True):
		self.subspace_contrast = HiCS(data, alpha, iterations)

		self.target = target
		self.features = [str(ft) for ft in data.columns.values if str(ft) != target]

		if drop_discrete:
			self.features = [ft for ft in self.features if self.subspace_contrast.types[ft] != 'discrete']

		self.feature_relevancies = pd.DataFrame({'feature' : self.features, 'score' : np.zeros(len(self.features))})
		self.redundancy_table = pd.Panel({'redundancy' : pd.DataFrame(data = 0, columns = self.features, index = self.features), 
			'weight' : pd.DataFrame(data = 0, columns = self.features, index = self.features)})
		self.relevancy_cycles = 0

		self.subspace_slices = {}
		self.subspace_relevancies = {}

	def subspace_relevancy(self, subspace, cach_slices = False):
		score, slices = self.subspace_contrast.calculate_contrast(features = subspace, target = self.target, return_slices = True)
		
		if cach_slices:
			subspace_string = json.dumps(subspace)

			if not subspace_string in self.subspace_slices:
				categorical = [{'name' : ft, 'values' : self.subspace_contrast.values(ft)} for ft in subspace if self.subspace_contrast.type(ft) == 'categorical']
				continuous = [ft for ft in subspace if self.subspace_contrast.type(ft) == 'continuous']
				self.subspace_slices[subspace_string] = ScoredSlices(categorical, continuous)

			self.subspace_slices[subspace_string].add_slices(slices)
			self.subspace_slices[subspace_string].reduce_slices()	
			return score, self.subspace_slices[subspace_string]

		else:
			categorical = [{'name' : ft, 'values' : self.subspace_contrast.values(ft)} for ft in subspace if self.subspace_contrast.type(ft) == 'categorical']
			continuous = [ft for ft in subspace if self.subspace_contrast.type(ft) == 'continuous']
			scored_slices = ScoredSlices(categorical, continuous)
			scored_slices.add_slices(slices)
			scored_slices.reduce_slices()
			return score, scored_slices

	def update_relevancies(self):
		for feature in self.features:
			score, dummy = self.subspace_relevancy(subspace = [feature], cach_slices = True)
			old_score = self.feature_relevancies.loc[self.feature_relevancies.feature == feature , 'score']
			self.feature_relevancies.loc[self.feature_relevancies.feature == feature , 'score'] = (old_score * self.relevancy_cycles + score)/(self.relevancy_cycles + 1)

		self.relevancy_cycles = self.relevancy_cycles + 1

	def update_redundancies(self, k = 5, redundancy_checks = 20):
		temp_redundancy_table = pd.Panel({'redundancy' : pd.DataFrame(data = 0, columns = self.features, index = self.features), 
			'weight' : pd.DataFrame(data = 0, columns = self.features, index = self.features)})
		
		for i in range(redundancy_checks):
			number_features = randint(1, k)
			selected_features = np.random.permutation(self.features)[0:number_features + 1].tolist()
			target = selected_features[number_features]
			subspace = selected_features[0:number_features]
			
			score = self.subspace_contrast.calculate_contrast(features = subspace, target = target)

			for ft in subspace:
				temp_redundancy_table.loc['redundancy', ft, target] = temp_redundancy_table['redundancy', ft, target] + score 
				temp_redundancy_table.loc['weight', ft, target] = temp_redundancy_table['weight', ft, target] + 1 
				temp_redundancy_table.loc['redundancy', target, ft] = temp_redundancy_table['redundancy', target, ft] + score
				temp_redundancy_table.loc['weight', target, ft] = temp_redundancy_table['weight', target, ft] + 1

		self.redundancy_table['redundancy'] = (self.redundancy_table['redundancy'] * self.redundancy_table['weight'] + temp_redundancy_table['redundancy'])/(self.redundancy_table['weight'] + temp_redundancy_table['weight'])
		self.redundancy_table['weight'] = self.redundancy_table['weight'] + temp_redundancy_table['weight']
		self.redundancy_table['redundancy'].fillna(0, inplace = True)

	def calculate_correlation(self, k = 5, redundancy_checks = 20, callback = print, limit = sys.maxsize):
		
		while self.relevancy_cycles < limit:
			self.update_relevancies()
			self.update_redundancies(k, redundancy_checks)
			
			callback(self.feature_relevancies, self.redundancy_table, self.subspace_slices)

		return self.feature_relevancies, self.redundancy_table, self.subspace_slices

