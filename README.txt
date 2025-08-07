## Evaluating Text Detection Using PaddleOCR with RetinaNet, YOLOv5, and Faster R-CNN

# Project Description :
This project aims to evaluate the performance of PaddleOCR in recognizing text from object detection outputs produced by three popular models: RetinaNet, YOLOv5, and Faster R-CNN. Additionally, it assesses the impact of various image preprocessing techniques on OCR accuracy.

Models Evaluated
   1. RetinaNet

   2. YOLOv5

   3. Faster R-CNN

Preprocessing Techniques Evaluated:
   1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
   2. Grayscale
   3. Padding
   4. No Invert Colors
   5. All Combined (padding, No Invert, thresholding, Grayscale, CLAHE(Contrast Limited Adaptive Histogram Equalization))

install requirement first:
--> Python 3.11.11
--> pip install -r requirements.txt

How to Run It:
--> open main.ipynb
--> run all