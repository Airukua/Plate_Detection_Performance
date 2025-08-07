import torch

def yolov5_crop(model, image_path, device='cpu' if torch.cuda.is_available() else 'cuda', confidence_threshold=0.5):
    results = model(image_path, size=320)
    df = results.pandas().xyxy[0]

    df_filtered = df[df['confidence'] > confidence_threshold]

    if not df_filtered.empty:
        row = df_filtered.iloc[0]
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        return x1, y1, x2, y2
    else:
        print(f"No detections with confidence above {confidence_threshold} for image: {image_path}")
        return None