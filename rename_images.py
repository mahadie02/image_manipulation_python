import os
import shutil

def rename_images(source_folder, destination_folder, file_prefix, start_number):
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get a list of image files in the source folder
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    # Rename and save images serially starting from start_number
    for index, file_name in enumerate(sorted(image_files)):
        file_path = os.path.join(source_folder, file_name)
        # Define new file name with prefix and serial number
        new_file_name = f"{file_prefix}{start_number + index:03d}.jpg"
        new_file_path = os.path.join(destination_folder, new_file_name)
        
        # Copy the original file to the new location with the new name
        shutil.copy(file_path, new_file_path)

    print(f"Renamed {len(image_files)} images and saved them to '{destination_folder}'.")


# Declare variables
source_folder = "input_images"        # Replace with your source folder path
destination_folder = "output_images"   # Replace with your destination folder path
file_prefix = ""                    # Optional prefix for the file names
start_number = 61                   # The starting number for image naming

# Call the function
rename_images(source_folder, destination_folder, file_prefix, start_number)
