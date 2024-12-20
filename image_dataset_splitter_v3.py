from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os
import shutil
import random
from zipfile import ZipFile

# Function to split, shuffle, and save dataset
def split_and_save_data(source_folder, output_folder, num_train, num_val, num_test):
    # Load images and masks
    image_files = sorted(os.listdir(os.path.join(source_folder, 'images')))
    mask_files = sorted(os.listdir(os.path.join(source_folder, 'masks')))

    # Make sure both images and masks are sequentially aligned
    assert len(image_files) == len(mask_files), "Number of images and masks must be the same!"

    # Combine images and masks for shuffling
    combined = list(zip(image_files, mask_files))

    # Shuffle images and masks while keeping correspondence
    random.shuffle(combined)
    image_files, mask_files = zip(*combined)

    # Split data
    train_images = image_files[:num_train]
    train_masks = mask_files[:num_train]

    val_images = image_files[num_train:num_train+num_val]
    val_masks = mask_files[num_train:num_train+num_val]

    test_images = image_files[num_train+num_val:num_train+num_val+num_test]
    test_masks = mask_files[num_train+num_val:num_train+num_val+num_test]

    # Function to save images and masks
    def save_images_and_masks(images, masks, output_subfolder, prefix):
        os.makedirs(os.path.join(output_folder, output_subfolder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, output_subfolder, 'masks'), exist_ok=True)

        for i, (image, mask) in enumerate(zip(images, masks)):
            image_dest = os.path.join(output_folder, output_subfolder, 'images', f'{prefix}_image_{i+1:03}.png')
            mask_dest = os.path.join(output_folder, output_subfolder, 'masks', f'{prefix}_mask_{i+1:03}.png')

            shutil.copy(os.path.join(source_folder, 'images', image), image_dest)
            shutil.copy(os.path.join(source_folder, 'masks', mask), mask_dest)

    # Save train, val, and test sets
    save_images_and_masks(train_images, train_masks, 'train', 'train')
    save_images_and_masks(val_images, val_masks, 'valid', 'valid')
    save_images_and_masks(test_images, test_masks, 'test', 'test')

# Function to zip the dataset
def zip_dataset(output_folder, zip_filename='dataset.zip'):
    zip_path = os.path.join(output_folder, zip_filename)
    with ZipFile(zip_path, 'w') as zipf:
        for folder_name in ['train', 'valid', 'test']:
            folder_path = os.path.join(output_folder, folder_name)
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_folder)
                    zipf.write(file_path, arcname)
    print(f"Dataset successfully zipped as {zip_path}")

# Function to authenticate Google Drive
def authenticate_drive(client_secrets_path):
    gauth = GoogleAuth()

    # Load the client_secrets.json from the specified path
    gauth.LoadClientConfigFile(client_secrets_path)

    # Modify OAuth settings to enforce offline access and prompt consent

    
    if os.path.exists("credentials.json"):
        gauth.LoadCredentialsFile("credentials.json")
    else:
        # Authenticate using the web browser if no saved credentials
        gauth.settings['get_refresh_token'] = True  # Ensures refresh token is requested
        gauth.LocalWebserverAuth()

    # If credentials are expired or invalid, refresh or authenticate again
    if gauth.access_token_expired:
        gauth.Refresh()  # This will now work with the refresh token
    gauth.SaveCredentialsFile("credentials.json")
    # Check if credentials are valid or expired
    if gauth.access_token_expired:
        gauth.Refresh()  # Refresh token should now work
    gauth.SaveCredentialsFile("credentials.json")

    # Return the authenticated GoogleDrive instance
    return GoogleDrive(gauth)

# Function to upload the zip file to Google Drive
def upload_to_drive(drive, local_file_path, drive_folder_id):
    file_name = os.path.basename(local_file_path)

    # Check if file already exists in the specified folder
    file_list = drive.ListFile({
        'q': f"'{drive_folder_id}' in parents and title='{file_name}' and trashed=false"
    }).GetList()

    if file_list:
        # File exists, update it
        drive_file = file_list[0]
        drive_file.SetContentFile(local_file_path)
        drive_file.Upload()
        print(f"File '{file_name}' updated successfully in Google Drive.")
    else:
        # File doesn't exist, upload it
        drive_file = drive.CreateFile({
            'title': file_name,
            'parents': [{'id': drive_folder_id}]
        })
        drive_file.SetContentFile(local_file_path)
        drive_file.Upload()
        print(f"File '{file_name}' uploaded successfully to Google Drive.")



# Main function
def main():
    # Paths
    source_folder = 'dataset_split'
    output_folder = 'dataset_split\\dataset'
    local_zip_file = os.path.join(output_folder, 'dataset.zip')

    # Google Drive Configuration
    drive_folder_id = '118UZpfIO49tXJmda6CEit_rnmIOy-nih'  # Replace with your Google Drive folder ID
    client_secrets_path = os.path.join(output_folder, 'client_secrets.json')  # Path to client_secrets.json

    # Parameters for splitting
    num_train = 0  # Number of train images
    num_val = 0     # Number of validation images
    num_test = 0    # Number of test images

    # Split, shuffle, and save dataset
    split_and_save_data(source_folder, output_folder, num_train, num_val, num_test)

    # Zip the output dataset
    zip_dataset(output_folder)

    # Authenticate Google Drive and upload the zip file
    drive = authenticate_drive(client_secrets_path)
    upload_to_drive(drive, local_zip_file, drive_folder_id)

    print("Dataset splitted, zipped, and uploaded to Google Drive successfully.")

# Run the main function
if __name__ == "__main__":
    main()
