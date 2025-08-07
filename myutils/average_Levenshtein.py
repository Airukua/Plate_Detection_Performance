import Levenshtein
import numpy as np

def average_edit_distance(df, ocr_col='ocr_result', corrected_col='corrected_text'):
    distances = []

    for pred, truth in zip(df[ocr_col].astype(str), df[corrected_col].astype(str)):
        pred = pred.replace(" ", "")
        truth = truth.replace(" ", "")
        distances.append(Levenshtein.distance(pred, truth))

    return np.mean(distances)
