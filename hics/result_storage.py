import numpy as np
import pandas as pd
from hics.scored_slices import ScoredSlices


class AbstractResultStorage:

	def update_relevancies(self, new_relevancies : pd.DataFrame):
		raise NotImplementedError()

	def update_redundancies(self, new_redundancies : pd.DataFrame, new_weights : pd.DataFrame):
		raise NotImplementedError()

	def update_slices(self, new_slices : dict()):
		raise NotImplementedError()


class DefaultResultStorage(AbstractResultStorage):

	def __init__(self, features : list()):
		self.relevancies = pd.DataFrame(columns = ['relevancy', 'iteration'])

		rendundancy_dict = {'redundancy' : pd.DataFrame(data = 0, columns = features, index = features), 'weight' : pd.DataFrame(data = 0, columns = features, index = features)}
		self.redundancies = pd.Panel(rendundancy_dict)

		self.slices = {}

	def update_relevancies(self, new_relevancies : pd.DataFrame):
		new_index = [index for index in new_relevancies.index if not index in self.relevancies.index]

		self.relevancies = self.relevancies.append(pd.DataFrame(data = 0, index = new_index, columns = self.relevancies.columns))
		
		self.relevancies.loc[new_relevancies.index, 'relevancies'] = (self.relevancies.iteration / (self.relevancies.iteration + new_relevancies.iteration)) * self.relevancies.relevancy + (new_relevancies.iteration / (self.relevancies.iteration + new_relevancies.iteration)) * new_relevancies.relevancy

		self.relevancies.loc[new_relevancies.index, 'iteration'] += new_relevancies.iteration

	def update_redundancies(self, new_redundancies : pd.DataFrame, new_weights : pd.DataFrame):
		self.redundancies.redundancy = (self.redundancies.weight / (new_weights + self.redundancies.weight)) * self.redundancies.redundancy + (new_weights / (new_weights + self.redundancies.weight)) * new_redundancies
		self.redundancies.redundancy.fillna(0, inplace = True)

		self.redundancies.weight += new_weights

	def update_slices(self, new_slices):
		for feature_set, new_slics in new_slices.items():
			if not feature_set in self.slices:
				self.slices[feature_set] = new_slices[feature_set]

			else:
				self.slices[feature_set].add_slices(new_slices[feature_set])


			self.slices[feature_set].reduce_slices()

