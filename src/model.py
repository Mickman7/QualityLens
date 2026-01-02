import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2, pretrained=True):
    """
    Loads MobileNetV3-Small and modifies the final layer for 
    Fresh vs. Rotten classification.
    """
    # 1. Load the pre-trained MobileNetV3
    # We use 'Small' because it's faster for CPU-based apps
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # 2. Find the input size of the original last layer
    # MobileNetV3 has a 'classifier' section; we look at the very last linear layer
    in_features = model.classifier[3].in_features

    # 3. Replace the last layer
    # The original was trained on 1,000 classes (ImageNet). 
    # We change it to 2 (Fresh vs. Rotten).
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    # Quick test to make sure the model loads
    my_model = get_model()
    print("Model successfully loaded!")
    print(my_model.classifier)