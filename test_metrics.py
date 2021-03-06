#
# Testing script for the localization and detection metrics proposed in the paper
#
# Joint Measurement of Localization and Detection of Sound Events
# Annamaria Mesaros, Sharath Adavanne, Archontis Politis, Toni Heittola, Tuomas Virtanen
# WASPAA 2019
#
# To calculate the metrics we use the reference from the dataset - TAU Spatial Sound Events 2019 - Microphone Array.
# The refernce labels can be downloaded here - https://zenodo.org/record/2599196/files/metadata_dev.zip
#
# As the predictions we use the output of the SELD method, SELDnet at 5, 25 and 75 epochs. These predictions can be
# downloaded here - https://zenodo.org/record/3354709  More about SELDnet can be read here - https://arxiv.org/abs/1905.08546
#
# This script has MIT license
#

import os
import sys
import numpy as np
import SELD_evaluation_metrics

# This import works only for python3.x For python2.x just copy this test_metrics.py and SELD_evaluation_metrics.py
# inside seld_dcase2019 and use it
sys.path.insert(0, 'seld_dcase2019')
from seld_dcase2019.metrics import evaluation_metrics
from seld_dcase2019 import cls_feature_class


def segment_gt_labels(_gt_dict, _unique_classes, _max_frames, _frames_seg):
    '''
    Collects class-wise sound event location information in segments of length _frames_seg from reference dataset

    :param _gt_dict: Dictionary containing sound event time and location information. Reference labels from dataset.
    :param _unique_classes: Total number of classes in the dataset
    :param _max_frames: Total number of frames in the recording
    :param _frames_seg: Number of frames in one segment
    :return: Dictionary containing class-wise sound event location information in each segment of audio
            dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)

    '''
    nb_blocks = int(np.ceil(_max_frames/float(_frames_seg)))
    output_dict = {x: {} for x in range(nb_blocks)}

    for se_cnt, se_start in enumerate(_gt_dict['start']):
        class_ind = _unique_classes[_gt_dict['class'][se_cnt]]
        se_end = _gt_dict['end'][se_cnt]

        # first block of SE
        first_block = se_start // _frames_seg
        first_frame = se_start % _frames_seg

        # last block of SE
        se_end = (_max_frames-1) if se_end >= _max_frames else se_end
        last_block = se_end // _frames_seg
        last_frame = se_end % _frames_seg

        if last_block == first_block:

            ind_list = range(first_frame, last_frame)
            if len(ind_list):
                azi_list = np.ones(len(ind_list)) * _gt_dict['azi'][se_cnt]
                ele_list = np.ones(len(ind_list)) * _gt_dict['ele'][se_cnt]

                if class_ind not in output_dict[first_block]:
                    output_dict[first_block][class_ind] = []
                output_dict[first_block][class_ind].append([ind_list, azi_list, ele_list])

        else:
            # first block
            first_ind_list = range(first_frame, _frames_seg)
            first_azi_list = np.ones(len(first_ind_list)) * _gt_dict['azi'][se_cnt]
            first_ele_list = np.ones(len(first_ind_list)) * _gt_dict['ele'][se_cnt]

            if class_ind not in output_dict[first_block]:
                output_dict[first_block][class_ind] = []
            output_dict[first_block][class_ind].append([first_ind_list, first_azi_list, first_ele_list])

            # last block
            if last_frame:
                last_ind_list = range(last_frame)
                last_azi_list = np.ones(len(last_ind_list)) * _gt_dict['azi'][se_cnt]
                last_ele_list = np.ones(len(last_ind_list)) * _gt_dict['ele'][se_cnt]

                if class_ind not in output_dict[last_block]:
                    output_dict[last_block][class_ind] = []
                output_dict[last_block][class_ind].append([last_ind_list, last_azi_list, last_ele_list])

            # intermediate blocks of SE
            for block_cnt in range(first_block+1, last_block):
                intermediate_ind_list = range(_frames_seg)
                intermediate_azi_list = np.ones(len(intermediate_ind_list)) * _gt_dict['azi'][se_cnt]
                intermediate_ele_list = np.ones(len(intermediate_ind_list)) * _gt_dict['ele'][se_cnt]

                if class_ind not in output_dict[block_cnt]:
                    output_dict[block_cnt][class_ind] = []
                output_dict[block_cnt][class_ind].append([intermediate_ind_list, intermediate_azi_list, intermediate_ele_list])
    return output_dict


def segment_pred_labels(_pred_dict, _max_frames, _frames_seg):
    '''
        Collects class-wise sound event location information in segments of length _frames_seg from reference dataset

    :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
    :param _max_frames: Total number of frames in the recording
    :param _frames_seg: Number of frames in one segment
    :return: Dictionary containing class-wise sound event location information in each segment of audio
            dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
    '''
    nb_blocks = int(np.floor(_max_frames/float(_frames_seg)))
    output_dict = {x: {} for x in range(nb_blocks)}
    for frame_cnt in range(0, max_frames, _frames_seg):
        block_cnt = frame_cnt // _frames_seg
        loc_dict = {}
        for audio_frame in range(frame_cnt, frame_cnt+_frames_seg):
            if audio_frame not in _pred_dict:
                continue
            for value in _pred_dict[audio_frame]:
                if value[0] not in loc_dict:
                    loc_dict[value[0]] = {}

                block_frame = audio_frame - frame_cnt
                if block_frame not in loc_dict[value[0]]:
                    loc_dict[value[0]][block_frame] = []
                loc_dict[value[0]][block_frame].append([value[1], value[2]])

        for class_cnt in loc_dict:
            if class_cnt not in output_dict[block_cnt]:
                output_dict[block_cnt][class_cnt] = []

            keys = [k for k in loc_dict[class_cnt]]
            values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

            output_dict[block_cnt][class_cnt].append([keys, values])

    return output_dict


# # # --------------------------- MAIN SCRIPT STARTS HERE -------------------------------------------

# INPUT DIRECTORY
ref_desc_files = 'results/metadata_dev'   # reference description directory location
pred_output_format_files = 'results/mic_dev_75'  # predicted output format directory location
                                                # options: mic_dev_75, mic_dev_25, mic_dev_5
print('system prediction files inputs: {}'.format(pred_output_format_files))

# Load feature class
feat_cls = cls_feature_class.FeatureClass()
max_frames = feat_cls.get_nb_frames()
unique_classes = feat_cls.get_classes()
azi_list, ele_list = feat_cls.get_azi_ele_list()
hop_len_s = feat_cls.get_hop_len_sec()
seg_len_s = 1 # options: hop_len_s, 0.5 or 1
seg_frames = int(seg_len_s/hop_len_s)

print('Frame length in seconds: {}'.format(seg_len_s))
print('Number of frames in a segment: {}'.format(seg_frames))

# collect reference files info
ref_file_list = os.listdir(ref_desc_files)
nb_ref_files = len(ref_file_list)

# collect predicted files info
pred_file_list = os.listdir(pred_output_format_files)
nb_pred_files = len(pred_file_list)

if nb_ref_files != nb_pred_files:
    print('ERROR: Mismatch. Reference has {} and prediction has {} files'.format(nb_ref_files, nb_pred_files))
    exit()


# Load evaluation metric class
for avg_type in [True, False]:
    print('\nAveraging type: {}'.format('Average spatial error within segment (SE_seg)' if avg_type else 'Average location within segment (AL_seg)'))
    for doa_threshold in [10, 20, 30, 40]:
        eval = SELD_evaluation_metrics.SELDMetrics(seg_length_s=seg_len_s, hop_length_s=hop_len_s, doa_thresh=doa_threshold, avg_type=avg_type)
        for pred_cnt, pred_file in enumerate(pred_file_list):
            # Load predicted output format file
            pred_dict = evaluation_metrics.load_output_format_file(os.path.join(pred_output_format_files, pred_file))
            pred_blocks_dict = segment_pred_labels(pred_dict, max_frames, seg_frames)

            # Load reference description file
            gt_dict = feat_cls.read_desc_file(os.path.join(ref_desc_files, pred_file.replace('.npy', '.csv')))
            gt_blocks_dict = segment_gt_labels(gt_dict, unique_classes, max_frames, seg_frames)

            eval.update_seld_scores(pred_blocks_dict, gt_blocks_dict)

        # Overall SELD scores
        output = eval.compute_seld_scores()
        print('threshold {}: ER_LD {:0.2f}, F_LD {:0.1f}, DE_CL {:0.1f}, F_CL {:0.1f}'.format(doa_threshold, output['ER_LD'], output['F_LD'], output['DE_CL'], output['F_CL']))
