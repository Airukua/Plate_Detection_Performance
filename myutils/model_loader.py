import torch
from myutils.fasterrcnn import create_model
from myutils.retinanet import get_model_instance_segmentation

def load_model(model_name: str, weight_path: str, num_classes: int = None) -> torch.nn.Module:
    """
    Load a custom-trained object detection model.

    Args:
        model_name (str): One of ['yolov5', 'yolov8', 'fastrcnn', 'retinanet']
        weight_path (str): Path to the trained model weights (.pth or .pt)
        num_classes (int): Number of classes (including background) — required for RetinaNet and Faster R-CNN

    Returns:
        torch.nn.Module: Loaded model in eval mode
    """
    model_name = model_name.lower()

    if model_name == 'yolov5':
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight_path)
        model.eval()
        return model

    elif model_name == 'yolov8':
        from ultralytics import YOLO
        model = YOLO(weight_path)
        return model

    elif model_name == 'fastrcnn':
        if num_classes is None:
            raise ValueError("You must specify num_classes for Faster R-CNN.")

        model = create_model(num_classes=num_classes)
        state_dict = torch.load(weight_path, map_location='cpu')

        try:
            model.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict)
        except RuntimeError as e:
            print(f"[WARNING] Strict load failed: {e}")
            print("Attempting partial state_dict load...")

            model_state = model.state_dict()
            ckpt_state = state_dict['model'] if 'model' in state_dict else state_dict
            filtered = {k: v for k, v in ckpt_state.items()
                        if k in model_state and v.shape == model_state[k].shape}
            model_state.update(filtered)
            model.load_state_dict(model_state, strict=False)
            print("Partial weights loaded (non-strict mode).")

        model.eval()
        return model

    elif model_name == 'retinanet':
        if num_classes is None:
            raise ValueError("You must specify num_classes for RetinaNet.")
        
        model = get_model_instance_segmentation(num_classes)
        state_dict = torch.load(weight_path, map_location='cpu')

        try:
            model.load_state_dict(state_dict['model'] if 'model' in state_dict else state_dict)
        except RuntimeError as e:
            print(f"[WARNING] Strict load failed: {e}")
            print("Attempting partial state_dict load...")

            model_state = model.state_dict()
            ckpt_state = state_dict['model'] if 'model' in state_dict else state_dict
            filtered = {k: v for k, v in ckpt_state.items()
                        if k in model_state and v.shape == model_state[k].shape}
            model_state.update(filtered)
            model.load_state_dict(model_state, strict=False)
            print("Partial weights loaded (non-strict mode).")

        model.eval()
        return model

    else:
        raise ValueError("Invalid model name. Choose from ['yolov5', 'yolov8', 'fastrcnn', 'retinanet']")
