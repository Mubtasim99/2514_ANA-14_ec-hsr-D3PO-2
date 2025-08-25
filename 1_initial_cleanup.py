"""Imports data as numpy array
"""

import os
import numpy as np
from loguru import logger
from bioio import BioImage
import bioio_ome_tiff

logger.info('Import ok')

input_path = 'P:/Mub/ecDNA/Renewed Cell Lines/2514_ec-hsr-D3PO-1/2514_biorep1/2514_files-for-quant'
output_folder = 'results/initial_cleanup/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


def image_converter(image_name, input_folder, output_folder, array=True):
    """converts images from instrument filetype (e.g., czi, lif, tif) to numpy array

    Args:
        image_name (str): image name
        input_folder (str): path to input folder
        output_folder (str): path to output folder
        array (numpy array, optional): saves image as numpy array. Defaults to True.
    """ 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    full_path = None
    for ext in valid_extensions:
        test_path = os.path.join(input_folder, image_name + ext)
        if os.path.exists(test_path):
            full_path = test_path
            break
    if full_path is None:
        logger.warning(f'File not found for {image_name}')
        return

    # Load image
    img = BioImage(full_path)
    image = img.get_image_data("CYX", B=0, Z=0, V=0, T=0)

    if array:
        np.save(f'{output_folder}{image_name}.npy', image)


# ---------------Initialize file_list---------------
# read in all folders with image files
valid_extensions = ['.czi', '.tif', '.tiff', '.lif']
file_list = [[f'{filename}' for filename in files if any(ext in filename.lower() for ext in valid_extensions)]
             for root, dirs, files in os.walk(f'{input_path}')]

# flatten file_list
file_list = [item for sublist in file_list for item in sublist]

# flag substrings of images you do not want processed, e.g., 'UT' for all untransfected controls
do_not_quantitate = []

# ---------------Collect image names & convert---------------
image_names = []
for filename in file_list:
    if all(word not in filename for word in do_not_quantitate):
        name = os.path.splitext(filename)[0]  # works for .czi, .tif, .lif
        image_names.append(name)

# remove duplicates
image_names = list(dict.fromkeys(image_names))

# ---------------Run conversion---------------
for name in image_names:
    image_converter(name, input_folder=input_path, output_folder=output_folder)

logger.info('initial cleanup complete :)')

