"""Load the selected checkpoint and apply the training preprocessing."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms as T


class LandUseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None).features
        self.projection = nn.Sequential(nn.Conv2d(1280, 128, 1), nn.ReLU())
        self.classifier = nn.Sequential(nn.Dropout(.30), nn.Linear(128, 5))

    def forward(self, images):
        features = self.backbone(images)
        tokens = self.projection(features).flatten(2).transpose(1, 2)
        return self.classifier(tokens.mean(dim=1))


def load_model(checkpoint_path):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    package = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if package["architecture"] != "efficientnet_b0" or package["additional_spatial_attention"]:
        raise ValueError("The checkpoint must be the selected EfficientNet-B0 baseline.")
    model = LandUseClassifier()
    model.load_state_dict(package["model_state_dict"], strict=True)
    model.eval()
    prep = package["preprocessing"]
    transform = T.Compose([
        T.Resize(tuple(prep["resize"])),
        T.ToTensor(),
        T.Normalize(prep["mean"], prep["std"]),
    ])
    return model, transform, package


def read_image(raw_bytes):
    with Image.open(BytesIO(raw_bytes)) as original:
        original.load()
        return original.convert("RGB")


def pixel_hash(image):
    rgb = image.convert("RGB")
    return sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


@torch.inference_mode()
def predict(model, transform, image):
    x = transform(image.convert("RGB")).unsqueeze(0)
    scores = model(x).softmax(1)[0].cpu().numpy()
    if scores.shape != (5,) or not np.isfinite(scores).all():
        raise ValueError("Invalid model output.")
    return scores
