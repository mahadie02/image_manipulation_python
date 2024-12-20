import os
import torch
from torchvision import models, transforms
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Paths for input, output, and model weights
input_folder = "input_images"  # Folder containing input images
output_folder = "output_images"  # Folder to save binary masks
model_weights_folder = "model_weights"  # Folder to store model weights

os.makedirs(output_folder, exist_ok=True)
os.makedirs(model_weights_folder, exist_ok=True)

# --- Download DeepLabV3+ and Backbone Weights ---
def download_weights(weights_folder):
    deeplab_weight_file = os.path.join(weights_folder, "deeplabv3_resnet50_coco.pth")
    resnet_weight_file = os.path.join(weights_folder, "resnet50-0676ba61.pth")

    # Download DeepLabV3+ weights if not already present
    if not os.path.exists(deeplab_weight_file):
        print("Downloading DeepLabV3+ weights...")
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        torch.hub.download_url_to_file(weights.url, deeplab_weight_file)
        print(f"Downloaded DeepLabV3+ weights to {deeplab_weight_file}")
    else:
        print(f"Using existing DeepLabV3+ weights from {deeplab_weight_file}")

    # Download ResNet-50 backbone weights if not already present
    if not os.path.exists(resnet_weight_file):
        print("Downloading ResNet-50 backbone weights...")
        backbone_url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
        torch.hub.download_url_to_file(backbone_url, resnet_weight_file)
        print(f"Downloaded ResNet-50 weights to {resnet_weight_file}")
    else:
        print(f"Using existing ResNet-50 weights from {resnet_weight_file}")

    return deeplab_weight_file, resnet_weight_file

# --- Load DeepLabV3+ Model ---
def load_deeplabv3(deeplab_weight_file, resnet_weight_file):
    model = models.segmentation.deeplabv3_resnet50(weights=None)  # Initialize model without weights

    # Load the model weights
    state_dict = torch.load(deeplab_weight_file, map_location=torch.device('cpu'), weights_only=True)

    # Ignore unexpected keys and load the rest of the state dictionary
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Print warnings for missing/unexpected keys
    if missing_keys:
        print(f"Missing keys when loading state_dict: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys when loading state_dict: {unexpected_keys}")

    # Load only the backbone part of ResNet-50 weights and ignore the fully connected (fc) layer
    resnet_state_dict = torch.load(resnet_weight_file, map_location=torch.device('cpu'), weights_only=True)
    
    # Remove fully connected (fc) layer weights
    resnet_state_dict.pop('fc.weight', None)
    resnet_state_dict.pop('fc.bias', None)

    # Load the backbone weights into the model
    model.backbone.load_state_dict(resnet_state_dict, strict=False)
    
    model.eval()
    return model

# --- Apply DeepLabV3+ to generate binary mask ---
def apply_deeplabv3(image, model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(image).unsqueeze(0)

    # Get segmentation output
    with torch.no_grad():
        output = model(img_tensor)['out']

    # Get the class predictions
    mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # --- Debug: Print unique classes in the mask ---
    unique_classes = np.unique(mask)
    print(f"Unique classes found in mask: {unique_classes}")

    # Generate binary mask: 0 for green space (vegetation), 255 for non-green space
    # COCO class index for vegetation (forest) is 21
    vegetation_class_index = 21
    binary_mask = np.where(mask == vegetation_class_index, 0, 255).astype(np.uint8)

    # --- Debug: Visualize the mask ---
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Generated Binary Mask")
    plt.show()

    return binary_mask

# --- Process Images ---
def process_images(input_folder, output_folder, model):
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path).convert('RGB')

        # Apply DeepLabV3+ and generate mask
        mask = apply_deeplabv3(image, model)

        # Save mask as grayscale image
        output_path = os.path.join(output_folder, f"mask_{i + 1:04d}.png")
        Image.fromarray(mask).save(output_path)

        print(f"Processed {image_file} and saved mask as {output_path}")

# --- Run the Process ---
deeplab_weight_file, resnet_weight_file = download_weights(model_weights_folder)
deeplabv3_model = load_deeplabv3(deeplab_weight_file, resnet_weight_file)
process_images(input_folder, output_folder, deeplabv3_model)
