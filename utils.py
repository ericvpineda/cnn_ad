import os 
from PIL import Image 
import numpy as np 

def get_image_properties(directory, resize_dim):
    
    pixels, labels, pics = [], [], [] 
    all_folders = os.listdir(directory)
    for i in range(len(all_folders)):

        foldername = all_folders[i]
        folder_path = directory + "/" + foldername
        all_files = os.listdir(folder_path)
        all_files_size = len(all_files)

        if all_files_size > 0:
            
            # Save single example image from each folder
            image = Image.open(folder_path + '/' + all_files[0]).convert('1')
            pics.append([image, foldername])

            for j in range(all_files_size):

                file_path = folder_path + '/' + all_files[j]
                # Convert image to greyscale 
                image = Image.open(file_path).convert('1')
                image = image.resize(resize_dim, Image.ANTIALIAS)
                image_pixels = list(image.getdata())
                pixels.append(image_pixels)
                labels.append(i)

    pixels = np.array(pixels)
    labels = np.array(labels)
    # Normalize pixel intensities
    pixels = pixels / 255.0
    return [pixels, labels, pics]
