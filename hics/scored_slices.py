import numpy as np
import pandas as pd
from hics.slice_similarity import continuous_similarity_matrix, categorical_similarity_matrix
from hics.slice_selection import select_by_similarity


class ScoredSlices:
    def __init__(self, categorical, continuous, to_keep=5, threshold=None):
        self.continuous = {feature: pd.DataFrame(columns=['to_value', 'from_value'])
            for feature in continuous}

        self.categorical = {feature['name']: pd.DataFrame(columns=feature['values'])
            for feature in categorical}

        self.scores = pd.Series()
        self.to_keep = to_keep

        if threshold is None:
            self.threshold = ScoredSlices.default_threshold(len(categorical) + len(continuous))
        else:
            self.threshold = threshold

    def add_slices(self, slices):
        if isinstance(slices, dict):
            self.add_from_dict(slices)

        else:
            self.add_from_object(slices)

    def add_from_object(self, slices):
        self.scores = self.scores.append(pd.Series(slices.scores)).sort_values(ascending=False, inplace=False)

        for feature, df in slices.continuous.items():
            self.continuous[feature] = pd.concat([self.continuous[feature], df], ignore_index=True)
            self.continuous[feature] = self.continuous[feature].loc[self.scores.index, :].reset_index(drop=True)

        for feature, df in slices.categorical.items():
            self.categorical[feature] = pd.concat([self.categorical[feature], df], ignore_index=True)
            self.categorical[feature] = self.categorical[feature].loc[self.scores.index, :].reset_index(drop=True)

        self.scores.reset_index(drop=True, inplace=True)

    def add_from_dict(self, slices):
        new_scores = pd.Series(slices['scores'])
        self.scores = self.scores.append(new_scores, ignore_index=True).sort_values(ascending=False, inplace=False)

        for feature in self.continuous:
            content = pd.DataFrame(slices['features'][feature])
            self.continuous[feature] = pd.concat([self.continuous[feature], content], ignore_index=True)
            self.continuous[feature] = self.continuous[feature].loc[self.scores.index, :].reset_index(drop=True)

        for feature in self.categorical:
            content = pd.DataFrame(slices['features'][feature], columns=self.categorical[feature].columns)
            self.categorical[feature] = pd.concat([self.categorical[feature], content], ignore_index=True)
            self.categorical[feature] = self.categorical[feature].loc[self.scores.index, :].reset_index(drop=True)

        self.scores.reset_index(drop=True, inplace=True)

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
        if self.continuous:
            continuous_similarity = continuous_similarity_matrix(self.continuous)
        else:
            continuous_similarity = np.ones((len(self.scores), len(self.scores)))

        if self.categorical:
            categorical_similarity = categorical_similarity_matrix(self.categorical)
        else:
            categorical_similarity = np.ones((len(self.scores), len(self.scores)))

        similarity = continuous_similarity * categorical_similarity

        selected = self.select_slices(similarity)

        if self.categorical:
            self.categorical = {key: df.loc[selected, :].reset_index(drop=True)
                for key, df in self.categorical.items()}

        if self.continuous:
            self.continuous = {key: df.loc[selected, :].reset_index(drop=True)
                for key, df in self.continuous.items()}

        self.scores = self.scores.loc[selected].reset_index(drop=True)

    def to_dict(self):
        continuous_dict = {name: df.to_dict(orient='list') for name, df in self.continuous.items()}
        categorical_dict = {name: df.to_dict(orient='list') for name, df in self.categorical.items()}
        scores_list = self.scores.tolist()
        return {'continuous': continuous_dict, 'categorical': categorical_dict, 'scores': scores_list, 'to_keep': self.to_keep, 'threshold': self.threshold}

    def to_output(self, name_mapping=None):
        if name_mapping is None:
            name_mapping = ScoredSlices.default_name_mapping

        result = []
        for index, value in self.scores.iteritems():
            current_result = {'deviation': value, 'features': {}}

            if self.continuous:
                for feature, df in self.continuous.items():
                    current_result['features'][name_mapping(feature)] = df.loc[index, :].to_dict()

            if self.categorical:
                for feature, df in self.categorical.items():
                    selected_values = df.columns[df.loc[index, :] == 1].astype(float).tolist()  # TODO: remove that bullshit
                    current_result['features'][name_mapping(feature)] = selected_values
            result.append(current_result)
        return result

    @staticmethod
    def default_threshold(dimensions):
        return pow(0.6, dimensions)

    @staticmethod
    def from_dict(dictionary):
        continuous = {name: pd.DataFrame(description)
            for name, description in dictionary['continuous'].items()}
        categorical = {name: pd.DataFrame(description)
            for name, description in dictionary['categorical'].items()}
        scores_series = pd.Series(dictionary['scores'])

        slices = ScoredSlices([], [], to_keep=dictionary['to_keep'], threshold=dictionary['threshold'])
        slices.categorical = categorical
        slices.continuous = continuous
        slices.scores = scores_series

        return slices

    @staticmethod
    def default_name_mapping(name):
        return name
