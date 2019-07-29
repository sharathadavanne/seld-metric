#
# Implements the localization and detection metrics proposed in the paper
#
# Joint Measurement of Localization and Detection of Sound Events
# Annamaria Mesaros, Sharath Adavanne, Archontis Politis, Toni Heittola, Tuomas Virtanen
# WASPAA 2019
#
#
# This script has MIT license
#

import numpy as np
eps = np.finfo(np.float).eps


class SELDMetrics(object):
    def __init__(self, seg_length_s=1, hop_length_s=0.02, doa_thresh=10, nb_classes=11, avg_type=True):
        '''
            This class implements both the class-sensitive localization and location-sensitive detection metrics.
            Additionally, based on the user input, the corresponding averaging is performed within the segment.


        :param seg_length_s: segment length in seconds. In the paper we use seg_length_s =0.02, 0.5, 1
        :param hop_length_s: hop length in seconds. In the paper we use hop_length_s=0.02
        :param doa_thresh: DOA threshold for location sensitive detection. In the paper, doa_thresh = 10, 20, 30, 40
        :param nb_classes: Number of sound classes. In the paper, nb_classes = 11
        :param avg_type: Averaging type within a segment.
                            True - average spatial error within segment
                            False - average location within segment
        '''

        self._is_avg_spatial_error = avg_type
        self._TP = 0
        self._FP = 0
        self._TN = 0
        self._FN = 0

        self._S = 0
        self._D = 0
        self._I = 0

        self._Nref = 0
        self._Nsys = 0

        self._total_DE = 0
        self._DE_TP = 0

        self._block_size = int(seg_length_s/hop_length_s)

        self._nb_classes = nb_classes
        self._spatial_T = doa_thresh

    def compute_seld_scores(self):
        '''
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores
        '''

        # Location-senstive detection performance
        ER = (self._S + self._D + self._I) / float(self._Nref + eps)

        prec = float(self._TP) / float(self._Nsys + eps)
        recall = float(self._TP) / float(self._Nref + eps)
        F = 2 * prec * recall / (prec + recall + eps)

        # Class-sensitive localization performance
        DE = self._total_DE / float(self._DE_TP + eps)

        DE_prec = float(self._DE_TP) / float(self._Nsys + eps)
        DE_recall = float(self._DE_TP) / float(self._Nref + eps)
        DE_F = 2 * DE_prec * DE_recall / (DE_prec + DE_recall + eps)

        return {'ER_LD': ER, 'F_LD': F * 100, 'DE_CL': DE, 'F_CL': DE_F * 100}

    def update_seld_scores(self, pred, gt):
        '''
        Calls the corresponding score averaging type based on self.avg_type flag

        :param pred: dictionary containing class-wise prediction results for each N-seconds segment block
        :param gt: dictionary containing class-wise groundtruth for each N-seconds segment block
        '''

        if self._is_avg_spatial_error:
            self.update_average_spatial_error_in_segment_score(pred, gt)
        else:
            self.update_average_location_in_segment_score(pred, gt)
        return

    def update_average_location_in_segment_score(self, pred, gt):
        '''
        Implements the location (position) averaging according to equation [4] in the paper

        :param pred: dictionary containing class-wise prediction results for each N-seconds segment block
        :param gt: dictionary containing class-wise groundtruth for each N-seconds segment block
        '''

        for block_cnt in range(len(gt.keys())):
            loc_FN, loc_FP = 0, 0
            for class_cnt in range(self._nb_classes):
                # Counting the number of ref and sys outputs should include the number of tracks for each class in the segment
                if class_cnt in gt[block_cnt]:
                    self._Nref += 1
                if class_cnt in pred[block_cnt]:
                    self._Nsys += 1

                if class_cnt in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # True positives or False negative case

                    # NOTE: For multiple tracks per class, identify multiple tracks using hungarian algorithm and then
                    # calculate the spatial distance using the following code. In the current code, we are assuming only
                    # one track per class.

                    gt_azi_list = np.array(gt[block_cnt][class_cnt][0][1]) * np.pi / 180
                    gt_ele_list = np.array(gt[block_cnt][class_cnt][0][2]) * np.pi / 180

                    pred_list = np.squeeze(np.array(pred[block_cnt][class_cnt][0][1]), 1) * np.pi / 180
                    pred_azi_list, pred_ele_list = pred_list[:, 0], pred_list[:, 1]

                    # IMPORTANT: calculate mean in cartesian format and NOT in spherical format. This can cause errors
                    # if the spatial location is around wrap around boundary

                    gt_x, gt_y, gt_z = sph2cart(gt_azi_list, gt_ele_list, 1)
                    pred_x, pred_y, pred_z = sph2cart(pred_azi_list, pred_ele_list, 1)

                    gt_azi, gt_ele, dum_d = cart2sph(np.mean(gt_x), np.mean(gt_y), np.mean(gt_z))
                    pred_azi, pred_ele , dum_d = cart2sph(np.mean(pred_x), np.mean(pred_y), np.mean(pred_z))

                    # Calculate distance between mean groundtruth and predicted locations
                    avg_spatial_dist = distance_between_spherical_coordinates_rad(gt_azi, gt_ele, pred_azi, pred_ele)

                    # DOA error and frame recall similar to journal paper
                    self._total_DE += avg_spatial_dist
                    self._DE_TP += 1

                    if avg_spatial_dist <= self._spatial_T:
                        self._TP += 1
                    else:
                        loc_FN += 1
                        self._FN += 1
                elif class_cnt in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                    # False negative
                    loc_FN += 1
                    self._FN += 1
                elif class_cnt not in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # False positive
                    loc_FP += 1
                    self._FP += 1
                elif class_cnt not in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                    # True negative
                    self._TN += 1

            self._S += np.minimum(loc_FP, loc_FN)
            self._D += np.maximum(0, loc_FN - loc_FP)
            self._I += np.maximum(0, loc_FP - loc_FN)
        return

    def update_average_spatial_error_in_segment_score(self, pred, gt):
        '''
        Implements the spatial error averaging according to equation [5] in the paper

        :param pred: dictionary containing class-wise prediction results for each N-seconds segment block
        :param gt: dictionary containing class-wise groundtruth for each N-seconds segment block
        '''

        for block_cnt in range(len(gt.keys())):
            loc_FN, loc_FP = 0, 0
            for class_cnt in range(self._nb_classes):
                # Counting the number of ref and sys outputs should include the number of tracks for each class in the segment
                if class_cnt in gt[block_cnt]:
                    self._Nref += 1
                if class_cnt in pred[block_cnt]:
                    self._Nsys += 1

                if class_cnt in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # True positives or False negative case

                    # NOTE: For multiple tracks per class, identify multiple tracks using hungarian algorithm and then
                    # calculate the spatial distance using the following code. In the current code, we are assuming only
                    # one track per class.

                    gt_azi_list = np.array(gt[block_cnt][class_cnt][0][1]) * np.pi / 180
                    gt_ele_list = np.array(gt[block_cnt][class_cnt][0][2]) * np.pi / 180

                    pred_list = np.squeeze(np.array(pred[block_cnt][class_cnt][0][1]), 1) * np.pi / 180
                    pred_azi_list, pred_ele_list = pred_list[:, 0], pred_list[:, 1]

                    total_spatial_dist = 0
                    total_framewise_matching_doa = 0
                    gt_ind_list = gt[block_cnt][class_cnt][0][0]
                    pred_ind_list = pred[block_cnt][class_cnt][0][0]
                    for gt_ind, gt_val in enumerate(gt_ind_list):
                        if gt_val in pred_ind_list:
                            total_framewise_matching_doa += 1
                            pred_ind = pred_ind_list.index(gt_val)
                            total_spatial_dist += distance_between_spherical_coordinates_rad(gt_azi_list[gt_ind], gt_ele_list[gt_ind], pred_azi_list[pred_ind], pred_ele_list[pred_ind])

                    if total_spatial_dist == 0 and total_framewise_matching_doa == 0:
                        loc_FN += 1
                        self._FN += 1
                    else:
                        avg_spatial_dist = (total_spatial_dist / total_framewise_matching_doa)

                        self._total_DE += avg_spatial_dist
                        self._DE_TP += 1

                        if avg_spatial_dist <= self._spatial_T:
                            self._TP += 1
                        else:
                            loc_FN += 1
                            self._FN += 1
                elif class_cnt in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                    # False negative
                    loc_FN += 1
                    self._FN += 1
                elif class_cnt not in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # False positive
                    loc_FP += 1
                    self._FP += 1
                elif class_cnt not in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                    # True negative
                    self._TN += 1

            self._S += np.minimum(loc_FP, loc_FN)
            self._D += np.maximum(0, loc_FN - loc_FP)
            self._I += np.maximum(0, loc_FP - loc_FN)
        return


def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
    """
    Angular distance between two spherical coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance

    :return: angular distance in degrees
    """
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def sph2cart(azimuth, elevation, r):
    '''
    Convert spherical to cartesian coordinates

    :param azimuth: in radians
    :param elevation: in radians
    :param r: in meters
    :return: cartesian coordinates
    '''

    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


def cart2sph(x, y, z):
    '''
    Convert cartesian to spherical coordinates

    :param x:
    :param y:
    :param z:
    :return: azi, ele in radians and r in meters
    '''

    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r
