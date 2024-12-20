import cv2
import numpy as np
import os

def segment_and_create_mask(input_folder, output_folder, min_area=500):
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
        lower_green = np.array([30, 40, 20])   # Adjust these values as needed
        upper_green = np.array([90, 255, 255]) # Adjust these values as needed

        # Create a binary mask for green space
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # Invert the mask to get non-green spaces as white (255) and green spaces as black (0)
        binary_mask = cv2.bitwise_not(green_mask)

        # Remove small regions based on area threshold
        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank mask
        filtered_mask = np.zeros_like(binary_mask)

        # Loop through contours and filter by area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                # Keep only the regions larger than the specified area
                cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Use morphological operations to smooth the mask
        kernel = np.ones((5, 5), np.uint8)
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel)

        # Save the filtered mask as a PNG file, with sequential naming
        output_path = os.path.join(output_folder, f'{idx + 1:03d}.png')
        cv2.imwrite(output_path, filtered_mask)

        print(f"Saved mask for {image_file} as {output_path}")

# Define input and output folders and minimum area for regions to keep
input_folder = "dataset_split\images" 
output_folder = "dataset_split\masks" 
min_area = 500  # Adjust the minimum area for filtering small regions

# Run the segmentation and create masks
segment_and_create_mask(input_folder, output_folder, min_area)
