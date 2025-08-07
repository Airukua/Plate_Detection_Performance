# This code inspired by : https://stackoverflow.com/questions/63531985/calculate-ocr-accuracy
# Refine by the help of ChatGPT
import difflib
import pandas as pd 

def _character_accuracy(ocr_result: str, corrected_text: str) -> float:
    """
    Calculate the character-level accuracy between OCR result and corrected text.

    Args:
        ocr_result (str): The text output from OCR.
        corrected_text (str): The manually corrected text.

    Returns:
        float: A ratio representing the character-level accuracy (0.0 to 1.0).
    """
    if pd.isna(ocr_result) or pd.isna(corrected_text):
        return 0.0

    matcher = difflib.SequenceMatcher(None, str(ocr_result), str(corrected_text))
    return matcher.ratio()