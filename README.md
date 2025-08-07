## 🔠 PaddleOCR for Licence Plate Character Recognition

PaddleOCR is used in the second stage of our system pipeline to perform **optical character recognition (OCR)** on detected licence plates. It was selected due to its lightweight architecture, open-source nature, multi-language support, and high accuracy.

### 📋 PaddleOCR Architecture Overview

The PaddleOCR pipeline consists of three main stages:

1. **Text Detection**
   Utilizes **DBNet (Differentiable Binarization)** to locate text regions in an image.

2. **Direction Classification**
   Identifies the orientation of detected text (e.g., rotated), and aligns it horizontally for optimal recognition.

3. **Text Recognition**
   Applies a **CRNN (Convolutional Recurrent Neural Network)** with **CTC (Connectionist Temporal Classification) Loss** to convert the aligned text into digital characters.

---

### 🧪 OCR Evaluation Based on Detection Models

PaddleOCR was tested using outputs from three detection models:

* YOLOv5
* RetinaNet
* Faster-RCNN

The evaluation metrics include:

* **Plate-level Accuracy** – Full plate recognized correctly
* **Character-level Accuracy** – Percentage of characters recognized correctly
* **Edit Distance** – Average number of corrections needed to match OCR output to ground truth

---

### 🔍 Initial OCR Results (Without Preprocessing)

| Detection Model | Avg OCR Confidence | Undetected Plates |
| --------------- | ------------------ | ----------------- |
| RetinaNet       | 82.39%             | 17                |
| YOLOv5          | 81.25%             | 19                |
| Faster-RCNN     | 78.79%             | 23                |

---

### 🎛️ OCR Performance with Preprocessing

Preprocessing methods such as **padding** and **grayscale conversion** were applied to improve recognition.

| Preprocessing Method | Plate Accuracy (%) | Character Accuracy (%) | Edit Distance |
| -------------------- | ------------------ | ---------------------- | ------------- |
| No Preprocessing     | 41.76              | 77.54                  | 2.37          |
| Padding Only         | 41.76              | **83.93**              | **1.82**      |
| Grayscale            | **43.53**          | 83.60                  | 1.85          |
| Contrast Adjustment  | 36.47              | 79.52                  | 2.31          |
| No Invert            | 15.29              | 62.81                  | 4.03          |
| All Preprocessing    | 21.18              | 63.78                  | 3.91          |

---

### 🏆 Conclusion

* **YOLOv5 combined with padding** produced the best OCR results, achieving a **character accuracy of 85.85%** with only **9 plates undetected**.
* **Padding** and **grayscale conversion** were the most effective preprocessing techniques.
* Other techniques such as image inversion negatively impacted OCR performance and are not recommended for this task.

