from torchvision import transforms
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from PIL import Image
import torch


def create_retinanet_model(num_classes, pretrained_backbone=True, trainable_backbone_layers=3):
    """
    Create a RetinaNet model with ResNet50-FPN backbone
    
    Args:
        num_classes (int): Number of output classes
        pretrained_backbone (bool): Whether to use pretrained backbone
        trainable_backbone_layers (int): Number of trainable layers in the backbone
    
    Returns:
        RetinaNet model
    """
    # Validate trainable_backbone_layers
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained_backbone, trainable_backbone_layers, 5, 3
    )
    
    # Create backbone
    backbone = resnet_fpn_backbone(
        backbone_name='resnet50',
        pretrained=pretrained_backbone,
        trainable_layers=trainable_backbone_layers
    )
    
    # Create anchor generator with different sizes and aspect ratios
    # License plates typically have a wider aspect ratio
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )
    
    # Create the RetinaNet model
    model = RetinaNet(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator
    )
    
    return model


def get_model_instance_segmentation(num_classes):
    """
    Gets the RetinaNet model. This is a wrapper function for create_retinanet_model
    for consistency with other detection models.
    
    Args:
        num_classes (int): Number of output classes 
    
    Returns:
        RetinaNet model
    """
    return create_retinanet_model(num_classes)

def retina_crop(model, image_tensor, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)[0]

    boxes = outputs['boxes']

    if len(boxes) == 0:
        return None

    x1, y1, x2, y2 = map(int, boxes[0].tolist())
    return x1, y1, x2, y2