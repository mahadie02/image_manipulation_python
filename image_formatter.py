import os
from PIL import Image

def convert_images(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpg'):
            # Full path to the input image
            input_path = os.path.join(input_folder, filename)

            # Open the image using Pillow
            with Image.open(input_path) as img:
                # Convert the image to RGB (just in case)
                img = img.convert('RGB')

                # Change the file extension to .png
                output_filename = os.path.splitext(filename)[0] + '.png'
                output_path = os.path.join(output_folder, output_filename)

                # Save the image as PNG
                img.save(output_path, 'PNG')

                print(f"Converted: {filename} -> {output_filename}")

# Define input and output folders
input_folder = "input_images"
output_folder = "output_images"

# Call the conversion function
convert_images(input_folder, output_folder)
