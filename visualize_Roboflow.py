import os
import cv2
import numpy as np

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

image_dir = "./datasets/Roboflow/train/images"
label_dir = "./datasets/Roboflow/train/labels"

colors = np.random.uniform(0, 255, size=(len(names), 3))

def draw_bounding_boxes(image_path, label_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    h, w = image.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])

        # denomalize
        x_center *= w
        y_center *= h
        width *= w
        height *= h

        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        color = colors[class_id]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        label = names[class_id]
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x_min, y_min - text_height - 10), (x_min + text_width, y_min), color, -1)
        cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return image

def visualize_images(image_dir, label_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            print(f"Label file not found for {image_file}")
            continue

        image_with_boxes = draw_bounding_boxes(image_path, label_path)
        if image_with_boxes is not None:
            cv2.imshow("Image with Bounding Boxes", image_with_boxes)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_images(image_dir, label_dir)