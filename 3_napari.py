"""Quality control: use napari to validate cellpose-generated masks
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import napari
from loguru import logger
from skimage.segmentation import clear_border
from napari.settings import get_settings
get_settings().application.ipy_interactive = False

image_folder = 'results/initial_cleanup/'
mask_folder = 'results/cellpose_masking/'
output_folder = 'results/napari_masking/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


def filter_masks(before_image, image_name, mask):
    """Quality control of cellpose-generated masks
    - Select the cell layer and using the fill tool set to 0, remove all unwanted cells.
    - Finally, using the brush tool add or adjust any masks within the appropriate layer.

    Args:
        before_image (ndarray): self explanatory
        image_name (str): self explanatory
        mask (ndarray): self explanatory

    Returns:
        ndarray: stacked masks
    """
    cells = mask[0, :, :].copy()
    nuclei = mask[1, :, :].copy()
    
    viewer = napari.Viewer()

    # create the viewer and add the image
    viewer = napari.view_image(before_image, name='before_image')
    # add the labels
    viewer.add_labels(cells, name='cells')
    viewer.add_labels(nuclei, name='nuclei')

    napari.run()

    np.save(f'{output_folder}{image_name}_mask.npy',
            np.stack([cells, nuclei]))
    logger.info(
        f'Processed {image_name}. Mask saved to {output_folder}{image_name}')

    return np.stack([cells, nuclei])


def stack_channels(name, masks_filtered, cells_filtered_stack):
    masks_filtered[name] = cells_filtered_stack

# ----------------Initialise file list----------------
# read in numpy masks
cell_masks = np.load(f'{mask_folder}cellpose_cellmasks.npy')
nuc_masks = np.load(f'{mask_folder}cellpose_nucmasks.npy')

# clean filenames
file_list = [filename for filename in os.listdir(
    image_folder) if 'npy' in filename]

images = {filename.replace('.npy', ''): np.load(
    f'{image_folder}{filename}') for filename in file_list}

raw_stacks = {
    image_name: np.stack([cell_masks[x, :, :], nuc_masks[x, :, :]])
    for x, image_name in (enumerate(images.keys()))}

# make new dictionary to check for saturation
image_names = images.keys()
image_values = zip(images.values(), raw_stacks.values())
saturation_check = dict(zip(image_names, image_values))

# ----------------filtering saturated cells and/or near border----------------
masks_filtered = {}
logger.info('removing saturated cells,  border cells, keeping cells expressing GFP')
for name, image in saturation_check.items():
    labels_filtered = []
    unique_val, counts = np.unique(image[1][0, :, :], return_counts=True)

    # loop to remove saturated cells and cells without GFP foci
    for label in unique_val[1:]:
        pixel_count = np.count_nonzero(image[1][0, :, :] == label)
        cell = np.where(image[1][0, :, :] == label, image[0][0,:,:], 0)
        nonzero_count = np.count_nonzero(cell > 0)
        containsGFP_count = np.count_nonzero(cell > 200)
        saturated_count = np.count_nonzero(cell > 60000)

        # if not too small, contains <1% saturated cells, and >5% bona fide GFP signal
        if len(cell[cell!=0]) > 1000:
            if (saturated_count/pixel_count) < 0.01:
                if (containsGFP_count/pixel_count) > 0.05:
                    labels_filtered.append(label)

    # keep filtered cells
    cells_filtered = np.where(
        np.isin(image[1][0, :, :], labels_filtered), image[1][0, :, :], 0)
    
    # remove cells near border
    cells_filtered = clear_border(cells_filtered, buffer_size=10)

    # keep intracellular nuclei
    intra_nuclei = np.where(
        cells_filtered >= 1, image[1][1, :, :], 0)
    
    # filter out small nuclei
    # mask_eroded = np.where(cell_eroded == 1, num, 0)
    nuc_unique_val, nuc_counts = np.unique(
        intra_nuclei, return_counts=True)
    for nuc_label in nuc_unique_val[1:]:
        nuc_test = np.where(intra_nuclei == nuc_label, nuc_label, 0)
        if np.unique(nuc_test, return_counts=True)[-1][-1] < 6000:
            intra_nuclei = np.where(intra_nuclei == nuc_label, 0, intra_nuclei)

    # stack the filtered masks
    cells_filtered_stack = np.stack((cells_filtered.copy(), intra_nuclei.copy()))
    stack_channels(name, masks_filtered, cells_filtered_stack)

# ---------------Manually filter masks---------------
# ONLY RUN THIS CHUNK ONCE. Manually validate cellpose segmentation.
already_filtered_masks = [filename.replace('_mask.npy', '') for filename in os.listdir(
    f'{output_folder}') if '_mask.npy' in filename]

unval_images = dict([(key, val) for key, val in images.items()
                    if key not in already_filtered_masks])

filtered_masks = {}
for image_name, image_stack in unval_images.items():
    image_stack
    mask_stack = masks_filtered[image_name].copy()
    filtered_masks[image_name] = filter_masks(
        image_stack, image_name, mask_stack)

# ---------------Process filtered masks---------------
# TODO make below lines a new script
# To reload previous masks for per-cell extraction
filtered_masks = {masks.replace('_mask.npy', ''): np.load(
    f'{output_folder}{masks}', allow_pickle=True) for masks in os.listdir(f'{output_folder}') if '_mask.npy' in masks}

logger.info('removing nuclei from cell masks')
cytoplasm_masks = {}
for name, img in filtered_masks.items():
    name
    cell_mask = img[0, :, :]
    nuc_mask = img[1, :, :]
    # make binary masks
    cell_mask_binary = np.where(cell_mask, 1, 0)
    nuc_mask_binary = np.where(nuc_mask, 1, 0)
    single_cytoplasm_masks = []
    # need this elif in case images have no masks
    if len(np.unique(cell_mask).tolist()) > 1:
        for num in np.unique(cell_mask).tolist()[1:]:
            num
            # subtract whole nuclear mask per cell
            cytoplasm = np.where(cell_mask == num, cell_mask_binary, 0)
            cytoplasm_minus_nuc = np.where(cytoplasm == nuc_mask_binary, 0, cytoplasm)
            if np.count_nonzero(cytoplasm) != np.count_nonzero(cytoplasm_minus_nuc):
                # re-assign label
                cytoplasm_num = np.where(cytoplasm_minus_nuc, num, 0)
                single_cytoplasm_masks.append(cytoplasm_num)
            else:
                single_cytoplasm_masks.append(
                    np.zeros(np.shape(cell_mask)).astype(int))
    else:
        single_cytoplasm_masks.append(
        np.zeros(np.shape(cell_mask)).astype(int))
    # add cells together and update dict
    summary_array = sum(single_cytoplasm_masks)
    cytoplasm_masks[name] = summary_array
logger.info('nuclei removed')

# ---------------save arrays---------------
np.save(f'{output_folder}cytoplasm_masks.npy', cytoplasm_masks)
logger.info('mask arrays saved')
