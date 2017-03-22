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

	def get_redundancies(self):
		raise NotImplementedError()

	def get_relevancies(self):
		raise NotImplementedError()

	def get_slices(self):
		raise NotImplementedError()


class DefaultResultStorage(AbstractResultStorage):

	def __init__(self, features : list()):
		self.relevancies = pd.DataFrame(columns = ['relevancy', 'iteration'])

		rendundancy_dict = {'redundancy' : pd.DataFrame(data = 0, columns = features, index = features), 'weight' : pd.DataFrame(data = 0, columns = features, index = features)}
		self.redundancies = pd.Panel(rendundancy_dict)

		self.slices = {}

	def update_relevancies(self, new_relevancies : pd.DataFrame):
		self.relevancies = new_relevancies

	def update_redundancies(self, new_redundancies : pd.DataFrame, new_weights : pd.DataFrame):
		self.redundancies.redundancy = new_redundancies
		self.redundancies.weight = new_weights

	def update_slices(self, new_slices):
		self.slices = new_slices

	def get_redundancies(self):
		return self.redundancies.redundancy, self.redundancies.weight

	def get_relevancies(self):
		return self.relevancies

	def get_slices(self):
		return self.slices
