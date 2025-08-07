from torchvision.models.detection import FasterRCNN
from torchvision.models import mobilenet_v2
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision
import torch 
from torchvision.transforms import functional as F


def create_model(num_classes):
    backbone = mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model

def fasterrcnn_crop(model, image_tensor, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)

    boxes = outputs[0]['boxes']

    if len(boxes) == 0:
        return None

    box = boxes[0]
    x1, y1, x2, y2 = map(int, box.tolist())
    return x1, y1, x2, y2