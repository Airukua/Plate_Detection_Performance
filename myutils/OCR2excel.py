import pandas as pd

def save_ocr_results(filenames, ocr_results, confidences, output_path=None):
    """
    Save OCR results to an Excel file.

    Parameters:
        filenames (list): List of filenames.
        ocr_results (list): List of OCR results.
        confidences (list): List of confidence scores.
        output_path (str): Path to save the Excel file..
    """
    df_ocr = pd.DataFrame({
        'filename': filenames,
        'ocr_result': ocr_results,
        'confidence': confidences
    })
    df_ocr.to_excel(output_path, index=False)
    print(f"Saved OCR results to {output_path}")
