'''
Check the training, validation, and testing result in ./output/fine_tune folder

'''
from ultralytics import YOLO
import torch
import os
from pathlib import Path
import yaml

checkpoint_path = './output/SimCLR/simclr_checkpoint_best.pth.tar'  ## simclr_checkpoint_best.pth.tar
epochs = 50

## Step 0: Setup Roboflow dataset: fine-tune the model on the dataset using the generated YAML file
dataset_root = Path('./datasets/Roboflow').resolve()  # Convert to absolute path
train_data = {
    'image_dir': str(dataset_root / 'train/images'),
    'label_dir': str(dataset_root / 'train/labels')
}
val_data = {
    'image_dir': str(dataset_root / 'valid/images'),
    'label_dir': str(dataset_root / 'valid/labels')
}
test_data = {
    'image_dir': str(dataset_root / 'test/images'),
    'label_dir': str(dataset_root / 'test/labels')
}

names = [
    '24V-power-cord', 'acousto-optic-alarm', 'area-display', 'bus-isolation-module',
    'coded-smoke-detector', 'coded-temperature-detector', 'dedicated-metal-module-box-for-fire-pump',
    'dedicated-metal-module-box-for-fire-smoke-exhaust-fan', 'dedicated-metal-module-box-for-fire-supplementary-fan',
    'deflation-indicator-light', 'electrical-fire-monitoring-line', 'emergency-manual-start-stop-button',
    'explosion-proof-smoke-detector', 'fire-broadcasting-line', 'fire-equipment-power-monitoring-line',
    'fire-fan-manual-control-line', 'fire-hydrant-button', 'fire-telephone-extension', 'fire-water-pump-manual-control-line',
    'gas-spray-audible-and-visual-alarm', 'gun-type-infrared-camera-in-the-basement', 'i-o-module', 'input-module',
    'light-display', 'manual-alarm-button-with-fire-telephone-jack', 'manual-automatic-switching-device', 'metal-modular-box',
    'normally-open-smoke-exhaust-valve-with-280-operation', 'normally-open-smoke-exhaust-valve-with-70-operation',
    'normally-open-smoke-exhaust-valve-with-70-operation-closed-in-case-of-fire', 'pressure-switch-flow-switch-start-pump-line',
    'pressure-switch-gas-extinguisher', 'safety-signal-valve', 'secondary-fire-shutter-door-control-box',
    'security-video-intercom-door-machine', 'smoke-vent', 'speaker', 'the-electromagnetic-valve', 'video-intercom-card-reader',
    'voltage-signal-sensor', 'water-flow-indicator'
]

# Create and save data configuration in yaml for YOLO
data_config = {
    'path': str(dataset_root),  # Absolute path to the dataset root
    'train': str(train_data['image_dir']),
    'val': str(val_data['image_dir']),
    'test': str(test_data['image_dir']),
    'nc': len(names),
    'names': names
}

data_yaml_path = str(dataset_root / 'data.yaml')
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)
print(f"Data configuration saved to {data_yaml_path}")

## Step 1: Load the Model and Pretrained SimCLR-YOLO Weights
model = YOLO("yolov10s.pt")  ## load with yolo's pretrained weight
# model = YOLO("yolov10s.yaml")  ## load without yolo's pretrained weight
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    combined_weights = checkpoint['state_dict']

    ## Load the SimCLR's backbone weights into the model, allowing for missing or unexpected keys
    missing, unexpected = model.model.load_state_dict(combined_weights, strict=False)

    ## Verify the loading process
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    print("Combined model weights loaded successfully!")
else:
    print(f"Warning: Checkpoint not loaded at {checkpoint_path}. Using default YOLOv10s weights.")


## Step 2: Freeze the Backbone Layers to retain learned Features
trained_layers = 11
for idx, layer in enumerate(model.model.model[:trained_layers]):
    for param in layer.parameters():
        param.requires_grad = False
print("Backbone layers have been frozen.")

print('\n','='*10,"Layers will be trained:", '='*10)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

## Step 3: Fine-Tune the Model
os.makedirs('./output/fine_tuned/', exist_ok=True)
model.train(
    data=data_yaml_path,
    epochs=epochs,
    imgsz=640,
    batch=4,
    lr0=1e-5,
    pretrained=False,
    project='./output/fine_tuned/',
    name='train_results',
)

final_model_path = './output/fine_tuned/yolo_final.pt'
model.save(final_model_path)
print(f"Fine-tuned model saved to {final_model_path}")

## Step 4: Evaluate the Model on the Validation Dataset
model.eval()
metrics = model.val(
    project='./output/fine_tuned/',
    name='validation_results'
)

## Step 5: Make Predictions on Test Images
results = model.predict(
    source=test_data['image_dir'],
    save=True,
    project='./output/fine_tuned/',
    name='test_predictions',
    imgsz=640
)

print("Fine-tuning, evaluation, and predictions completed.")
