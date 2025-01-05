'''

YOLOv10s
'''

import torch.nn as nn
from ultralytics import YOLO
import torch

class YOLOSimCLR(nn.Module):

    def __init__(self, out_dim, pretrained=True):
        super(YOLOSimCLR, self).__init__()

        # Initialize YOLOv10 model and load pretrained weights if specified
        if pretrained:
            yolo_model = YOLO("yolov10s.pt")
        else:
            yolo_model = YOLO("yolov10s.yaml")

        ## Extract all backbone layers from the YOLO model
        trained_layers = 11  # Adjust based on YOLOv10 architecture
        self.model = nn.Sequential(*list(yolo_model.model.model)[:trained_layers])

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 64,64)  # Dummy input with the same size as the trained images
            dim_mlp = self.model(dummy_input).shape[1]

        # Add MLP projection head with two linear layers and a ReLU activation
        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, x):
        features = self.model(x)  ## backbone of yolo
        pooled_features = nn.AdaptiveAvgPool2d((1, 1))(features)
        flattened_features = pooled_features.flatten(1)
        compact_features = self.projection_head(flattened_features)
        return compact_features



if __name__ == '__main__':
    out_dim = 128
    model = YOLOSimCLR(out_dim, True)
    print(model)

    # Optionally, test the model with a dummy input
    dummy_input = torch.randn(1, 3, 64, 64)
    output = model(dummy_input)
    print("Output shape:", output.shape)