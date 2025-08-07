import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def get_character_confusion(df, ocr_col='ocr_result', corrected_col='corrected_text', 
                                          normalize=False, output_path='confusion_matrix.csv'):
    ocr_chars = []
    gt_chars = []

    for pred, truth in zip(df[ocr_col].astype(str), df[corrected_col].astype(str)):
        pred = pred.replace(" ", "")
        truth = truth.replace(" ", "")
        min_len = min(len(pred), len(truth))

        for i in range(min_len):
            ocr_chars.append(pred[i])
            gt_chars.append(truth[i])

    labels = sorted(set(gt_chars + ocr_chars))
    cm = confusion_matrix(gt_chars, ocr_chars, labels=labels, normalize='true' if normalize else None)
    
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    cm_df.to_csv(output_path)
    print(f"Confusion matrix saved to {output_path}")
    
    return cm_df

def get_most_confused_characters(cm_df):
    if cm_df.columns[0] != cm_df.index.name:
        cm_df = cm_df.set_index(cm_df.columns[0])

    cm_df = cm_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    confusion_info = []
    for true_char in cm_df.index:
        for pred_char in cm_df.columns:
            if true_char != pred_char:
                count = cm_df.at[true_char, pred_char]
                if count > 0:
                    confusion_info.append((true_char, pred_char, count))

    confusion_info.sort(key=lambda x: x[2], reverse=True)
    top_confusions = pd.DataFrame(confusion_info[:10], columns=["True Character", "Predicted As", "Count"])
    return top_confusions