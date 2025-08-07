def _plate_level_accuracy(df, ocr_col='ocr_result', corrected_col='corrected_text'):
    """
    Calculate plate-level accuracy by comparing cleaned OCR results and corrected text.

    Parameters:
        df (pd.DataFrame): DataFrame containing OCR results and ground truth.
        ocr_col (str): Column name for OCR results. Default is 'ocr_result'.
        corrected_col (str): Column name for corrected/ground-truth text. Default is 'corrected_text'.

    Returns:
        float: Plate-level accuracy in percentage.
    """
    df[ocr_col] = df[ocr_col].astype(str).str.replace(' ', '', regex=False)
    df[corrected_col] = df[corrected_col].astype(str).str.replace(' ', '', regex=False)
    
    df['plate_correct'] = df[ocr_col] == df[corrected_col]
    accuracy = df['plate_correct'].mean() * 100
    
    return accuracy
