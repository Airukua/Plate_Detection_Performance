# This code inspired by : https://github.com/oshmita26/Automatic-number-plate-recognition-with-PaddleOCR
import os
from PIL import Image
import cv2
import numpy as np
from paddleocr import PaddleOCR
from skimage.segmentation import clear_border
from myutils.fasterrcnn import fasterrcnn_crop
from myutils.yolov5 import yolov5_crop
from myutils.retinanet import retina_crop
from torchvision import transforms
import torch

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def process_images(
    image_paths: list[str],
    model,
    model_name: str,
    padding: int = 5,
    grayscale: bool = False,
    threshold: bool = False,
    invert: bool = False,
    contrast: bool = False, 
    device: str = 'cpu' if torch.cuda.is_available() else 'cuda'
):
    ocr_detected_name = []
    ocr_detected_accuracy = []
    filenames = []

    for image in image_paths:
        image_load = cv2.imread(image)
        height, width, _ = image_load.shape

        # Object detection
        if model_name == 'yolov5':
            result = yolov5_crop(model, image, confidence_threshold=0.001)

        elif model_name == 'retinanet':
            transform = transforms.Compose([transforms.ToTensor()])
            image_tensor = transform(image_load).unsqueeze(0).to(device)
            result = retina_crop(model, image_tensor)

        elif model_name == 'fastrcnn':
            transform = transforms.Compose([transforms.ToTensor()])
            image_tensor = transform(image_load).unsqueeze(0).to(device)
            result = fasterrcnn_crop(model, image_tensor)

        else:
            print(f"Unsupported model: {model_name}")
            continue

        # Handle case when no box is detected
        if result is None:
            print(f"{image} - No box detected")
            ocr_detected_name.append('None')
            ocr_detected_accuracy.append(0)
            filenames.append(os.path.basename(image))
            continue

        x1, y1, x2, y2 = result

        # Crop with padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        cropped = image_load[y1:y2, x1:x2]
        processed = cropped

        # Optional image processing
        if grayscale:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        if threshold:
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            processed = clear_border(processed)

        if invert:
            processed = cv2.bitwise_not(processed)

        if contrast:
            processed = cv2.convertScaleAbs(processed, alpha=1.5, beta=0)

        # OCR
        ocr_result = ocr.ocr(processed, cls=True)
        base_name = os.path.splitext(os.path.basename(image))[0]
        filename = f"{base_name}.jpg"
        filenames.append(filename)

        if not ocr_result or not ocr_result[0]:
            print(f"Image: {filename} - No OCR detected")
            ocr_detected_name.append('None')
            ocr_detected_accuracy.append(0)
        else:
            ocr_detected_name.append(ocr_result[0][0][1][0])
            ocr_detected_accuracy.append(ocr_result[0][0][1][1])

    return ocr_detected_name, ocr_detected_accuracy, filenames
