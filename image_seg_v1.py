import cv2
import numpy as np
import os

def segment_and_create_mask(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Loop through each image file and process it
    for idx, image_file in enumerate(image_files):
        # Load the image
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)

        # Convert the image to HSV color space for better color segmentation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define thresholds for green space (e.g., grass, trees, etc.)
        lower_green = np.array([35, 40, 40])   # Adjust these values as needed
        upper_green = np.array([90, 255, 255]) # Adjust these values as needed

        # Create a binary mask for green space
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # Invert the mask to get non-green spaces as white (255) and green spaces as black (0)
        binary_mask = cv2.bitwise_not(green_mask)

        # Save the binary mask as a PNG file, with sequential naming
        output_path = os.path.join(output_folder, f'mask_{idx + 1:03d}.png')
        cv2.imwrite(output_path, binary_mask)

        print(f"Saved mask for {image_file} as {output_path}")

# Define input and output folders
input_folder = 'old_images'
output_folder = 'new_images'

# Run the segmentation and create masks
segment_and_create_mask(input_folder, output_folder)
