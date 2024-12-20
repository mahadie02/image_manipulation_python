import os
from PIL import Image

def resize_images(source_folder, destination_folder, resolution=(512, 512), file_prefix="image"):
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get a list of image files in the source folder
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    # Resize and save images serially
    for index, file_name in enumerate(sorted(image_files)):
        file_path = os.path.join(source_folder, file_name)
        with Image.open(file_path) as img:
            # Resize image
            resized_img = img.resize(resolution, Image.Resampling.LANCZOS)
            # Save image in the destination folder with serial name
            output_name = f"{file_prefix}{index + 1:03d}.jpg"
            output_path = os.path.join(destination_folder, output_name)
            resized_img.save(output_path)

    print(f"Resized {len(image_files)} images and saved them to '{destination_folder}'.")

# Example usage:
source_folder = "input_images"      # Replace with your source folder path
destination_folder = "output_images"   # Replace with your destination folder path
resize_images(source_folder, destination_folder, resolution=(512, 512), file_prefix="")
