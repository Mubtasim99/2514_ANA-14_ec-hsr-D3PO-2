"""Detect puncta, measure features, visualize data
"""

import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import skimage.io
import functools
import cv2
from skimage import measure, segmentation, morphology
from scipy.stats import skewtest
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
from statannotations.Annotator import Annotator
from loguru import logger
from matplotlib_scalebar.scalebar import ScaleBar
plt.rcParams.update({'font.size': 14})

input_folder = 'results/initial_cleanup/'
mask_folder = 'results/napari_masking/'
output_folder = 'results/summary_calculations/'
plotting_folder = 'results/plotting/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

if not os.path.exists(plotting_folder):
    os.mkdir(plotting_folder)

def feature_extractor(mask, properties=False):

    if not properties:
        properties = ['area', 'eccentricity', 'label', 'major_axis_length', 'minor_axis_length', 'perimeter', 'coords']

    return pd.DataFrame(skimage.measure.regionprops_table(mask, properties=properties))

# ----------------Initialise file list----------------
file_list = [filename for filename in os.listdir(
    input_folder) if 'npy' in filename]

images = {filename.replace('.npy', ''): np.load(
    f'{input_folder}{filename}') for filename in file_list}

masks = {masks.replace('_mask.npy', ''):np.load(f'{mask_folder}{masks}', allow_pickle=True) 
         for masks in os.listdir(f'{mask_folder}') if '_mask.npy' in masks} 
#masks = np.load(f'{mask_folder}',
          #      allow_pickle=True).item()

# make dictionary from images and masks array
image_mask_dict = {
    key: np.stack([images[key][0, :, :], images[key][1, :, :], masks[key][0,:,:]])
    for key in masks
}

# ----------------collect feature information----------------
# remove saturated cells in case some were added during manual validation
not_saturated = {}
for name, image in image_mask_dict.items():
    labels_filtered = []
    unique_val, counts = np.unique(image[2, :, :], return_counts=True)

    # loop to remove saturated cells (>1% px values > 60000)
    for label in unique_val[1:]:
        label
        pixel_count = np.count_nonzero(image[2, :, :] == label)
        cell = np.where(image[2, :, :] == label, image[0, :, :], 0)
        saturated_count = np.count_nonzero(cell == 65535)

        if (saturated_count/pixel_count) < 0.05:
            labels_filtered.append(label)

    cells_filtered = np.where(
        np.isin(image[2, :, :], labels_filtered), image[2, :, :], 0)

    # stack the filtered masks
    cells_filtered_stack = np.stack(
        (image[0, :, :], image[1, :, :], cells_filtered))
    not_saturated[name] = cells_filtered_stack

# now collect puncta and cell features info
logger.info('collecting feature info')
feature_information_list = []
for name, image in not_saturated.items():
    # logger.info(f'Processing {name}')
    labels_filtered = []
    unique_val, counts = np.unique(image[2, :, :], return_counts=True)
    # find cell outlines for later plotting
    cell_binary_mask = np.where(image[2, :, :] !=0, 1, 0)
    contours = measure.find_contours(cell_binary_mask, 0.8)
    contour = [x for x in contours if len(x) >= 100]
    # loop to extract params from cells
    for num in unique_val[1:]:
        num
        cell = np.where(image[2, :, :] == num, image[0, :, :], 0)
        cell_std = np.std(cell[cell != 0])
        cell_mean = np.mean(cell[cell != 0])
        binary = (cell > (cell_mean+(cell_std*2))).astype(int)
        puncta_masks = measure.label(binary)
        puncta_masks = remove_small_objects(puncta_masks, 9)
        cell_properties = feature_extractor(puncta_masks)

        # make list for cov and skew, add as columns to properties
        granule_cov_list = []
        granule_skew_list = []
        granule_intensity_list = []
        for granule_num in np.unique(puncta_masks)[1:]:
            granule_num
            granule = np.where(puncta_masks == granule_num, image[0,:,:], 0)
            granule = granule[granule!=0]
            granule_cov = np.std(granule) / np.mean(granule)
            granule_cov_list.append(granule_cov)
            res = skewtest(granule)
            granule_skew = res.statistic
            granule_skew_list.append(granule_skew)
            granule_intensity_list.append(np.mean(granule))
        cell_properties['granule_cov'] = granule_cov_list
        cell_properties['granule_skew'] = granule_skew_list
        cell_properties['granule_intensity'] = granule_intensity_list
        
        if len(cell_properties) < 1:
            cell_properties.loc[len(cell_properties)] = 0

        properties = pd.concat([cell_properties])
        properties['image_name'] = name
        properties['cell_number'] = num
        properties['cell_size'] = np.size(cell[cell!=0])
        properties['cell_intensity_mean'] = cell_mean

        # add cell outlines to coords
        properties['cell_coords'] = [contour]*len(properties)

        feature_information_list.append(properties)
        
feature_information = pd.concat(feature_information_list)
logger.info('completed feature collection')

# add condition
feature_information['cell_type'] = feature_information['image_name'].str.split('_').str[3]
feature_information['treatment'] = feature_information['image_name'].str.split('_').str[4]
feature_information['biomarker'] = feature_information['image_name'].str.split('_').str[5]

# add aspect ratio and circularity
feature_information['aspect_ratio'] = feature_information['minor_axis_length'] / feature_information['major_axis_length']
feature_information['circularity'] = (12.566*feature_information['area'])/(feature_information['perimeter']**2)

# save data for plotting coords
feature_information.to_csv(f'{output_folder}puncta_detection_feature_info.csv')

# remove massive puncta
feature_information = feature_information[feature_information['area'] < 2000]

# # threshold bona fide puncta by dimmest puncta in sham treatment
# min_intensity = feature_information[feature_information['condition'] == 'Shammut'].granule_intensity.min()
# feature_information = feature_information[(feature_information['granule_intensity'] > min_intensity) | (feature_information['granule_intensity'] == 0)]

# make additional df for avgs per replicate
features_of_interest = ['granule_cov', 'granule_skew', 'granule_intensity', 'circularity', 'eccentricity','cell_intensity_mean', 'cell_size', 'area']
granule_summary_reps = []
for col in features_of_interest:
    # feature_information = feature_information.reset_index()
    reps_table = feature_information.groupby(['cell_type', 'treatment', 'biomarker', 'cell_number']).mean(numeric_only=True)[f'{col}']
    granule_summary_reps.append(reps_table)
granule_summary_reps_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['cell_type', 'treatment', 'biomarker', 'cell_number'], how='outer'), granule_summary_reps).reset_index()

pairs = [('hsrMDM2', 'ecMDM2')]
order = ['hsrMDM2', 'ecMDM2']
x = 'cell_type' 
plt.figure(figsize=(15, 12))
plt.subplots_adjust(hspace=0.5)
plt.suptitle("calculated parameters - per granule, not normalized", fontsize=18, y=0.99)
for n, parameter in enumerate(features_of_interest):
    # add a new subplot iteratively
    ax = plt.subplot(3, 3, n + 1)

    # filter df and plot ticker on the new subplot axis
   # sns.stripplot(data=feature_information, x=x, y=parameter, dodge='True', 
   #                 edgecolor='white', linewidth=1, size=8, alpha=0.4, order=order, ax=ax)
    sns.stripplot(data=granule_summary_reps_df.reset_index(drop=True), x=x, y=parameter, dodge='True', edgecolor='k', linewidth=1, size=8, order=order, ax=ax)
    sns.violinplot(data=feature_information.reset_index(drop=True), x=x, y=parameter,
                color='.9', order=order, ax=ax)
    
    # statannot stats
    annotator = Annotator(ax, pairs, data=granule_summary_reps_df, x=x, y=parameter, order=order)
    annotator.configure(test='t-test_ind', verbose=2)
    annotator.apply_test()
    annotator.annotate()

    # formatting
    sns.despine()
    ax.set_xlabel('')

plt.tight_layout()
plt.savefig(f'{output_folder}puncta-features_pergranule_raw_celltype.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)

#more analysis for treatments instead of cell_type
pairs = [('(-)Hygro', '(+)Hygro')]
order = ['(-)Hygro', '(+)Hygro']
x = 'treatment' 
plt.figure(figsize=(15, 10))
plt.subplots_adjust(hspace=0.5)
plt.suptitle("calculated parameters - per granule, not normalized", fontsize=18, y=0.99)
for n, parameter in enumerate(features_of_interest):
    # add a new subplot iteratively
    ax = plt.subplot(3, 3, n + 1)

    # filter df and plot ticker on the new subplot axis
   # sns.stripplot(data=feature_information, x=x, y=parameter, dodge='True', 
   #                 edgecolor='white', linewidth=1, size=8, alpha=0.4, order=order, ax=ax)
    sns.stripplot(data=granule_summary_reps_df.reset_index(drop=True), x=x, y=parameter, dodge='True', edgecolor='k', linewidth=1, size=8, order=order, ax=ax)
    sns.violinplot(data=feature_information.reset_index(drop=True), x=x, y=parameter,
                color='.9', order=order, ax=ax)
    
    # statannot stats
    annotator = Annotator(ax, pairs, data=granule_summary_reps_df, x=x, y=parameter, order=order)
    annotator.configure(test='t-test_ind', verbose=2)
    annotator.apply_test()
    annotator.annotate()

    # formatting
    sns.despine()
    ax.set_xlabel('')

plt.tight_layout()
plt.savefig(f'{output_folder}puncta-features_pergranule_raw_treatment.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)

#more analysis with biomarker comparisons
pairs = [('NPM1', 'H2AX'), ('NPM1', 'SON'), ('SON', 'H2AX')]
order = ['NPM1', 'SON', 'H2AX']
x = 'biomarker' 
plt.figure(figsize=(15, 10))
plt.subplots_adjust(hspace=0.5)
plt.suptitle("calculated parameters - per granule, not normalized", fontsize=18, y=0.99)
for n, parameter in enumerate(features_of_interest):
    # add a new subplot iteratively
    ax = plt.subplot(3, 3, n + 1)

    # filter df and plot ticker on the new subplot axis
   # sns.stripplot(data=feature_information, x=x, y=parameter, dodge='True', 
   #                 edgecolor='white', linewidth=1, size=8, alpha=0.4, order=order, ax=ax)
    sns.stripplot(data=granule_summary_reps_df.reset_index(drop=True), x=x, y=parameter, dodge='True', edgecolor='k', linewidth=1, size=8, order=order, ax=ax)
    sns.violinplot(data=feature_information.reset_index(drop=True), x=x, y=parameter,
                color='.9', order=order, ax=ax)
    
    # statannot stats
    annotator = Annotator(ax, pairs, data=granule_summary_reps_df, x=x, y=parameter, order=order)
    annotator.configure(test='t-test_ind', verbose=2)
    annotator.apply_test()
    annotator.annotate()

    # formatting
    sns.despine()
    ax.set_xlabel('')

plt.tight_layout()
plt.savefig(f'{output_folder}puncta-features_pergranule_raw_biomarker.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)


# --------------Grab major and minor_axis_length for punctas--------------
minor_axis = feature_information.groupby(
    ['image_name', 'cell_number'])['minor_axis_length'].mean()
major_axis = feature_information.groupby(
    ['image_name', 'cell_number'])['major_axis_length'].mean()

# --------------Calculate average size of punctas per cell--------------
puncta_avg_area = feature_information.groupby(
    ['image_name', 'cell_number'])['area'].mean().reset_index()

# --------------Calculate proportion of area in punctas--------------
cell_size = feature_information.groupby(
    ['image_name', 'cell_number'])['cell_size'].mean()
puncta_area = feature_information.groupby(
    ['image_name', 'cell_number'])['area'].sum()
puncta_proportion = ((puncta_area / cell_size) *
                   100).reset_index().rename(columns={0: 'proportion_puncta_area'})

# --------------Calculate number of 'punctas' per cell--------------
puncta_count = feature_information.groupby(
    ['image_name', 'cell_number'])['area'].count()

# --------------Calculate average size of punctas per cell--------------
avg_eccentricity = feature_information.groupby(
    ['image_name', 'cell_number'])['eccentricity'].mean().reset_index()

# --------------Grab cell intensity mean --------------
granule_cov_mean = feature_information.groupby(
    ['image_name', 'cell_number'])['granule_cov'].mean()

# --------------Grab cell intensity mean --------------
granule_skew_mean = feature_information.groupby(
    ['image_name', 'cell_number'])['granule_skew'].mean()

# --------------Grab cell intensity mean --------------
cell_intensity_mean = feature_information.groupby(
    ['image_name', 'cell_number'])['cell_intensity_mean'].mean()

# --------------Summarise, save to csv--------------
summary = functools.reduce(lambda left, right: pd.merge(left, right, on=['image_name', 'cell_number'], how='outer'), [cell_size.reset_index(), puncta_avg_area, puncta_proportion, puncta_count.reset_index(), minor_axis, major_axis, avg_eccentricity, granule_cov_mean, granule_skew_mean, cell_intensity_mean])
summary.columns = ['image_name', 'cell_number',  'cell_size', 'mean_puncta_area', 'puncta_area_proportion', 'puncta_count', 'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'avg_eccentricity', 'granule_cov_mean', 'granule_skew_mean', 'cell_intensity_mean']

# --------------tidy up dataframe--------------
# add columns for sorting
summary['cell_type'] = summary['image_name'].str.split('_').str[3]
summary['treatment'] = summary['image_name'].str.split('_').str[4]
summary['biomarker'] = summary['image_name'].str.split('_').str[5]

summary.to_csv(f'{output_folder}puncta_detection_summary.csv')

# make df where all puncta features are normalized to mean cell intensity
normalized_summary = summary.copy()
for column in normalized_summary.columns[3:-3]:
    column
    normalized_summary[column] = normalized_summary[column] / normalized_summary['cell_intensity_mean']

# --------------visualize calculated parameters - raw --------------
plt.figure(figsize=(15, 15))
plt.subplots_adjust(hspace=0.5)
plt.suptitle('calculated parameters - per cell, *not* normalized to cytoplasm intensity', fontsize=18, y=0.99)
# loop through the length of tickers and keep track of index
for n, parameter in enumerate(summary.columns.tolist()[3:-3]):
    # add a new subplot iteratively
    ax = plt.subplot(3, 4, n + 1)

    sns.stripplot(data=summary, x=x, y=parameter, dodge='True',
                    edgecolor='k', linewidth=1, size=8, order=order, ax=ax)
    sns.boxplot(data=summary, x=x, y=parameter,
                palette=['.9'], order=order, ax=ax)
    
    # statannot stats
    annotator = Annotator(ax, pairs, data=summary, x=x, y=parameter, order=order)
    annotator.configure(test='t-test_ind', verbose=2)
    annotator.apply_test()
    annotator.annotate()

    # formatting
    sns.despine()
    ax.set_xlabel('')

plt.tight_layout()
plt.savefig(f'{output_folder}puncta-features_percell_raw.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)

# --------------visualize calculated parameters - normalized --------------
plt.figure(figsize=(15, 15))
plt.subplots_adjust(hspace=0.5)
plt.suptitle('calculated parameters - per cell, normalized to cytoplasm intensity', fontsize=18, y=0.99)
# loop through the length of tickers and keep track of index
for n, parameter in enumerate(summary.columns.tolist()[3:-3]):
    # add a new subplot iteratively
    ax = plt.subplot(3, 4, n + 1)

    # filter df and plot ticker on the new subplot axis
    sns.stripplot(data=normalized_summary, x=x, y=parameter, dodge='True',
                    edgecolor='k', linewidth=1, size=8, order=order, ax=ax)
    sns.boxplot(data=normalized_summary, x=x, y=parameter,
                palette=['.9'], order=order, ax=ax)
    
    # statannot stats
    annotator = Annotator(ax, pairs, data=normalized_summary, x=x, y=parameter, order=order)
    annotator.configure(test='t-test_ind', verbose=2)
    annotator.apply_test()
    annotator.annotate()

    # formatting
    sns.despine()
    ax.set_xlabel('')

plt.tight_layout()
plt.savefig(f'{output_folder}puncta-features_percell_normalized.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)

# -------------- plotting proofs --------------
# get 80% max intensity
vmax = 0
for name, image in image_mask_dict.items():
    if image[0, :, :].max() > vmax:
        vmax = image[0, :, :].max()
vmax = vmax*0.8

# plot proofs
for name, image in image_mask_dict.items():
    name
    unique_val, counts = np.unique(image[2, :, :], return_counts=True)

    # extract coords
    cell = np.where(image[2, :, :] != 0, image[0, :, :], 0)
    image_df = feature_information[(feature_information['image_name'] == name)]
    if len(image_df) > 0:
        cell_contour = image_df['cell_coords'].iloc[0]
        coord_list = np.array(image_df.coords)

        # plot
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image[1,:,:], cmap=plt.cm.gray)
        ax1.imshow(image[0,:,:], alpha=0.60, vmax=vmax)
        ax2.imshow(cell, vmax=vmax)
        for cell_line in cell_contour:
            ax2.plot(cell_line[:, 1], cell_line[:, 0], linewidth=0.5, c='w')
        if len(coord_list) > 1:
            for puncta in coord_list:
                if isinstance(puncta, np.ndarray):
                    ax2.plot(puncta[:, 1], puncta[:, 0], linewidth=0.5)
        for ax in fig.get_axes():
            ax.label_outer()

        # Create scale bar
        scalebar = ScaleBar(0.0779907, 'um', location = 'lower right', pad = 0.3, sep = 2, box_alpha = 0, color='w', length_fraction=0.3)
        ax1.add_artist(scalebar)

        # title and save
        fig.suptitle(name, y=0.78)
        fig.tight_layout()
        fig.savefig(f'{plotting_folder}{name}_proof.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)
        plt.close()