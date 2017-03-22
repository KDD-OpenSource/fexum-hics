import numpy as np
import pandas as pd
import sys
import json
from random import randint
from hics.contrast_meassure import HiCS
from hics.scored_slices import ScoredSlices
from hics.result_storage import DefaultResultStorage
from math import pow


class IncrementalCorrelation:
	def __init__(self, data, target, result_storage, iterations = 10, alpha = 0.1, drop_discrete = False):
		self.subspace_contrast = HiCS(data, alpha, iterations)

		self.target = target
		self.features = [str(ft) for ft in data.columns.values if str(ft) != target]

		if drop_discrete:
			self.features = [ft for ft in self.features if self.subspace_contrast.types[ft] != 'discrete']

		self.result_storage = result_storage

	def update_bivariate_relevancies(self, runs = 5):
		new_slices = {}
		new_scores = {json.dumps([feature]) : {'relevancy' : 0, 'iteration' : 0} for feature in self.features}

		for i in range(runs):
			for feature in self.features:
				subspace_str = json.dumps([feature])
				new_scores[subspace_str]['relevancy'] += self.calculate_contrast([feature], self.target, new_slices)
				new_scores[subspace_str]['iteration'] += 1

		new_scores_df = pd.DataFrame.from_dict(new_scores, orient = 'index')
		new_scores_df.relevancy /= new_scores_df.iteration

		self.result_storage.update_relevancies(new_scores_df)
		self.result_storage.update_slices(new_slices)

	def update_multivariate_relevancies(self, k = 5, runs = 5):
		new_slices = {}
		new_scores = {}

		for i in range(runs):
			number_features = randint(1, k)
			selected_features = np.random.permutation(self.features)[0:number_features + 1].tolist()
			target = selected_features[number_features]
			subspace = selected_features[0:number_features]
			subspace_str = json.dumps(sorted(subspace))
			score = self.calculate_contrast(subspace, target, new_slices)

			if not subspace_str in new_scores:
				new_scores[subspace_str] = {'relevancy' : score, 'iteration' : 1}
			else:
				new_scores[subspace_str]['relevancy'] += score
				new_scores[subspace_str]['iteration'] += 1

		new_scores_df = pd.DataFrame.from_dict(new_scores, orient = 'index')
		new_scores_df.relevancy /= new_scores_df.iteration

		self.result_storage.update_relevancies(new_scores_df)
		self.result_storage.update_slices(new_slices)

	def update_redundancies(self, k = 5, runs = 10):
		new_redundancies = pd.DataFrame(data = 0, columns = self.features, index = self.features)
		new_weights = pd.DataFrame(data = 0, columns = self.features, index = self.features)

		for i in range(runs):
			number_features = randint(1, k)
			selected_features = np.random.permutation(self.features)[0:number_features + 1].tolist()
			target = selected_features[number_features]
			subspace = selected_features[0:number_features]
			
			score, _dummy = self.calculate_contrast(features = subspace, target = target)

			for ft in subspace:
				new_redundancies.loc[ft, target] = new_redundancies.loc[ft, target] + score 
				new_weights.loc[ft, target] = new_weights.loc[ft, target] + 1 
				new_redundancies.loc[target, ft] = new_redundancies.loc[target, ft] + score
				new_weights.loc[target, ft] = new_weights.loc[target, ft] + 1

		new_redundancies = new_redundancies / new_weights
		new_redundancies.fillna(0, inplace = True)
		self.result_storage.update_redundancies(new_redundancies, new_weights)

	def feature_relevancies(self, features, runs = 1):
		subspace_str = jspn.dumps(sorted(features))

		new_slices = {}
		new_score = {subspace_str : {'relevancy' : 0, 'iteration' : 0}}

		for i in range(runs):
			new_score[subspace_str]['relevancy'] += self.calculate_contrast(features, self.target, new_slices)
			new_score[subspace_str]['iteration'] += 1

		new_scores_df = pd.DataFrame.from_dict(new_scores, orient = 'index')
		new_scores_df.relevancy /= new_scores_df.iteration

		self.result_storage.update_relevancies(new_scores_df)
		self.result_storage.update_slices(new_slices)

	def calculate_contrast(self, features, target, slices_storage : dict() = None):
		if slices_storage == None:
			return self.subspace_contrast.calculate_contrast(features, target, False)

		else:
			score, slices = self.subspace_contrast.calculate_contrast(features, target, True)
			subspace_str = json.dumps(sorted(features))

			if not subspace_str in slices_storage:
				categorical = [{'name' : ft, 'values' : self.subspace_contrast.values(ft)} for ft in features if self.subspace_contrast.type(ft) == 'categorical']
				continuous = [ft for ft in features if self.subspace_contrast.type(ft) == 'continuous']
				slices_storage[subspace_str] = ScoredSlices(categorical, continuous)

			slices_storage[subspace_str].add_slices(slices)
			#slices_storage[subspace_str].reduce_slices()	
			return score






	


