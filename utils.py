import os 
from PIL import Image 
import numpy as np 

def get_image_properties(directory, resize_dim):
    
    pixels, labels = [], [] 
    all_folders = os.listdir(directory)
    for i in range(len(all_folders)):

        foldername = all_folders[i]
        folder_path = directory + "/" + foldername
        
        if len(os.listdir(folder_path)) > 0:

            for filename in os.listdir(folder_path):

                file_path = folder_path + '/' + filename
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
    return [pixels, labels]
