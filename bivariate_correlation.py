import numpy as np
import pandas as pd
from random import randint
from hics.contrast_meassure import HiCS
from slice_similarity.slice_similarity import calculate_similarity_matrix
from slice_similarity.slice_selection import select_by_similarity


def scored_slices_to_blocks(slices, scores, features):
	blocks = []

	for i in slices.major_axis:
		featureRanges = []
		for ft in features:
			featureRanges.append({'start' : slices[ft].loc[i, 'start'], 'end' : slices[ft].loc[i, 'end'], 'name' : ft}) 
		blocks.append({'significance' : scores.loc[i, 'score'], 'featureRanges' : featureRanges})

	return blocks


def slices_to_scored_slices(slices, features):
	scores = pd.DataFrame({'score' : []})
	slices_panel = pd.Panel({ft : pd.DataFrame({'start' : [], 'end' : []}) for ft in features})

	for slc in slices:
		for cond in slc['conditions']:
			slices_panel.loc[cond['feature'], len(slices_panel[cond['feature']]), ['start', 'end']] = (cond['from_value'],  cond['to_value'])

		scores = scores.append(pd.DataFrame({'score' : [slc['score']]}), ignore_index = True)

	scores = scores.sort_values(by = 'score')[::-1]
	slices_panel = slices_panel[: ,scores.index.values, :]
	return slices_panel, scores


def calculate_relevancies(subspace_contrast: HiCS, target, features):
	relevancies = []
	
	for feature in features:
		score, slices = subspace_contrast.calculate_contrast(features = [feature], target = target, return_slices = True)

		slices_panel, slice_scores = slices_to_scored_slices(slices, [feature])

		similarity = calculate_similarity_matrix(slices_panel)
		selection = select_by_similarity(slices_panel, similarity, 0.5)
		
		slices_panel = slices_panel.loc[:, selection, :]
		slice_scores = slice_scores.loc[selection, :]

		blocks = scored_slices_to_blocks(slices_panel, slice_scores, [feature])
		
		relevancies.append({'name' : feature, 'score' : score, 'scoredBlocks' : blocks})

	return relevancies


def calculate_redundancies(subspace_contrast: HiCS, features, k = 5, checked_subspaces = 50):
	redundancy_table = pd.Panel({'redundancy' : pd.DataFrame(data = 0, columns = features, index = features), 'weight' : pd.DataFrame(data = 0, columns = features, index = features)})

	for i in range(checked_subspaces):
		number_features = randint(1, k)
		selected_features = np.random.permutation(features)[0:number_features + 1].tolist()
		target = selected_features[number_features]
		subspace = selected_features[0:number_features]
		
		score = subspace_contrast.calculate_contrast(subspace, target)

		for ft in subspace:
			redundancy_table.loc['redundancy', ft, target] = redundancy_table['redundancy', ft, target] + score 
			redundancy_table.loc['weight', ft, target] = redundancy_table['weight', ft, target] + 1 
			redundancy_table.loc['redundancy', target, ft] = redundancy_table['redundancy', ft, target] + score
			redundancy_table.loc['weight', target, ft] = redundancy_table['weight', ft, target] + 1

	redundancy_table['redundancy'] = redundancy_table['redundancy'] / redundancy_table['weight']
	redundancy_table['redundancy'].fillna(0, inplace = True)

	x_features = np.array([features]*len(features))
	y_features = x_features.T
	feature_pairs = np.c_[y_features.ravel(), x_features.ravel()]
	feature_pairs = feature_pairs[feature_pairs[:,0] < feature_pairs[:,1]]

	relevancies = [{'feature1' : fts[0], 'feature2' : fts[1], 'weight' : redundancy_table['weight', fts[0], fts[1]], 'redundancy' : redundancy_table['redundancy', fts[0], fts[1]]} for fts in feature_pairs]

	return relevancies


def bivariate_HiCS(filepath, target, drop_discrete = True, alpha = 0.1, iterations = 50, k = 5, redundancy_checks = 500):
	data = pd.read_csv(filepath)
	features = [str(column) for column in data.columns.values if str(column) != target]
	
	subspace_contrast = HiCS(data, alpha, iterations)
	
	if drop_discrete:
		features = [ft for ft in features if subspace_contrast.types[ft] != 'discrete']

	relevancies = calculate_relevancies(subspace_contrast, target, features)
	redundancies = calculate_redundancies(subspace_contrast, features, k, redundancy_checks)

	return {'relevancies' : relevancies, 'redundancies' : redundancies}