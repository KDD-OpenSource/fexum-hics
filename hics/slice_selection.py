import numpy as np
import pandas as pd


def select_by_similarity(dfs, similarity, threshold):
    selected = []
    for index in dfs.major_axis:
        select = True
        for s in selected:
            if similarity[s, index] > threshold:
                select = False
                break
        if select:
            selected.append(index)
    return selected
