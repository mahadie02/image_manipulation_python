import os
from PIL import Image
import rembg

def remove_background(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            # Full path to the input image
            input_path = os.path.join(input_folder, filename)

            # Open the image and remove the background
            with open(input_path, 'rb') as input_file:
                input_image = input_file.read()
                output_image = rembg.remove(input_image)

                # Change the file extension to .png
                output_filename = os.path.splitext(filename)[0] + '.png'
                output_path = os.path.join(output_folder, output_filename)

                # Save the image with the background removed
                with open(output_path, 'wb') as output_file:
                    output_file.write(output_image)

                print(f"Processed: {filename} -> {output_filename}")

# Define input and output folders
input_folder = "input_images"
output_folder = "output_images"

# Call the background removal function
remove_background(input_folder, output_folder)
