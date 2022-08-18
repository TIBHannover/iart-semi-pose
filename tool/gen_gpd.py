import os
import sys
import re
import argparse
import logging
import math
import json
import msgpack

from shapely.geometry import LineString, Point, Polygon
import numpy as np

import multiprocessing as mp


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-p", "--poses_path", help="verbose output")
    parser.add_argument("-b", "--boxes_path", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    parser.add_argument("-t", "--threshold", type=float, default=0.9, help="verbose output")
    parser.add_argument("-m", "--min_keypoints", type=int, default=5, help="verbose output")
    args = parser.parse_args()
    return args


def dict_to_item_list(d, level=1):
    # Ressolves double list items k:[[a],[b]] -> [k:[a]],[k:[b]]
    items = []
    for k, item in d.items():
        if level == 1:
            if isinstance(item[0], list):
                for e in item:
                    items.append([k, e])
            else:
                items.append([k, item])
        if level == 0:
            if isinstance(item, list):
                for e in item:
                    items.append([k, e])
            else:
                items.append([k, item])
    return items


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        print("Info: lines do not intersect")
        return -1
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]


def angle(l1, l2):
    [p11, p12] = list(l1.coords)
    [p21, p22] = list(l2.coords)
    # print("coords1: ", list(l1.coords))
    # print("coords2: ", list(l2.coords))
    intersect = None
    # When line is a point (due to prediction overlay kpts) return 0
    if (p11[0] == p12[0] and p11[1] == p12[1]) or (p21[0] == p22[0] and p21[1] == p22[1]):
        return 0
    # Find intersection point for error prone angle computation
    if p11 == p21:
        j1 = np.subtract(p12, p11)
        j2 = np.subtract(p22, p21)
    elif p11 == p22:
        j1 = np.subtract(p12, p11)
        j2 = np.subtract(p21, p22)
    elif p12 == p21:
        j1 = np.subtract(p11, p12)
        j2 = np.subtract(p22, p21)
    elif p12 == p22:
        j1 = np.subtract(p11, p12)
        j2 = np.subtract(p21, p22)
    else:
        # Rare call to this computation
        # Don't using shapely cause no extended lines
        intersect = line_intersection(l1.coords, l2.coords)
        if intersect == -1:  # parallel lines
            return 0.0
        # print("1:",p12, intersect)
        # print("2:",p22, intersect)

        j1 = np.subtract(p12, intersect)
        j2 = np.subtract(p22, intersect)

        # If intersection is the endpoint, take the startpoint
        if np.linalg.norm(j1) < 10e-4:
            j1 = np.subtract(p11, intersect)
            print("Info: intersection is exactly on the line endpoint")

            assert np.linalg.norm(j1) > 10e-4
        if np.linalg.norm(j2) < 10e-4:
            j2 = np.subtract(p21, intersect)
            print("Info: intersection is exactly on the line endpoint")

            assert np.linalg.norm(j2) > 10e-4

    # print(l1,list(l1.coords),l2,list(l2.coords))
    # print(j1,j2)
    try:
        j1_norm = j1 / np.linalg.norm(j1)
        j2_norm = j2 / np.linalg.norm(j2)
    except:
        plotsave_linesintersect(
            [0, 0, j1[0], j1[1]],
            [0, 0, j2[0], j2[1]],
            [p11[0], p11[1], p12[0], p12[1]],
            [p21[0], p21[1], p22[0], p22[1]],
            [intersect],
        )
        raise ValueError()
    # python3.6 /home/althausc/master_thesis_impl/scripts/pose_descriptors/geometric_pose_descriptor.py -inputFile /home/althausc/master_thesis_impl/detectron2/out/art_predictions/train/11-24_15-33-23/maskrcnn_predictions.json -mode JcJLdLLa_reduced -target insert

    return np.arccos(np.clip(np.dot(j1_norm, j2_norm), -1.0, 1.0))


class GlobalPoseDescriptor:
    def __init__(self, threshold=0.9, min_keypoints=5):
        self.norm = True
        self.threshold = threshold
        self.min_keypoints = min_keypoints

        self.num_keypoints = 17

        self.body_part_mapping = {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
            4: "right_ear",
            5: "left_shoulder",
            6: "right_shoulder",
            7: "left_elbow",
            8: "right_elbow",
            9: "left_wrist",
            10: "right_wrist",
            11: "left_hip",
            12: "right_hip",
            13: "left_knee",
            14: "right_knee",
            15: "left_ankle",
            16: "right_ankle",
        }

        self.inv_body_part_mapping = dict((v, k) for k, v in self.body_part_mapping.items())
        # Pairs of keypoints that should be exchanged under horizontal flipping
        # _KPT_SYMMETRY = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]  # not used

        # Keypoint connections in accordance to https://images4.programmersought.com/935/c3/c3a73bf51c47252f4a33566327e30a87.png
        # Modified/Added Lines: - left-right shoulder
        #                      - left hipbone-left shoulder
        #                      - right hipbone-right shoulder
        #                      - left-right hipbone
        self.keypoints_lines = [
            (0, 1),
            (0, 2),
            (2, 4),
            (1, 3),
            (6, 8),
            (8, 10),
            (5, 7),
            (7, 9),
            (12, 14),
            (14, 16),
            (11, 13),
            (13, 15),
        ]
        self.keypoints_lines.extend([(6, 5), (6, 12), (5, 11), (12, 11)])

        self.refs = {5: "left_shoulder", 6: "right_shoulder"}  # {1: "left_shoulder"}

        self.end_joints = {
            3: "left_ear",
            4: "right_ear",
            9: "left_wrist",
            10: "right_wrist",
            15: "left_ankle",
            16: "right_ankle",
        }
        self.end_joints_depth_2 = [(0, 4), (0, 3), (6, 10), (5, 9), (12, 16), (11, 15)]

    def normalizevec(self, featurevector, rangemin=0, rangemax=1, mask=False):
        if mask:
            # Filter out unvalid features (=-1) for normalization
            # Then combine normalization and unvalid features in same ordering
            maskvalid = []
            for n in featurevector:
                if n == -1:
                    maskvalid.append(False)
                else:
                    maskvalid.append(True)
            subvec = [n for n, l in zip(featurevector, maskvalid) if l is True]
            if len(subvec) > 0:
                if max(subvec) != min(subvec):
                    normsubvec = [
                        (x - min(subvec)) * (rangemax - rangemin) / (max(subvec) - min(subvec)) + rangemin
                        for x in subvec
                    ]
                else:
                    # relation description not possible when same values or just on entry
                    print("Info: relation description not possible")
                    normsubvec = [-1 for _ in subvec]

            normvec = []
            c_valid = 0
            for l in maskvalid:
                if l is True:
                    normvec.append(normsubvec[c_valid])
                    c_valid += 1
                else:
                    normvec.append(-1)
            return normvec
        else:
            # Normalize aka rescale input vector to new range
            return [
                (x - min(featurevector)) * (rangemax - rangemin) / (max(featurevector) - min(featurevector)) + rangemin
                for x in featurevector
            ]

    def joint_coordinates_rel(self, keypoints, kptsvalid, imagesize, addconfidences=None):
        # Joint Coordinate: all keypoints in relative coordinate system + reference point in world coordinate system
        # Make all keypoints relative to a selected reference point
        ref_ids = list(self.refs.keys())
        if len(self.refs) == 2:
            p1 = keypoints[ref_ids[0]]
            p2 = keypoints[ref_ids[1]]
            pmid = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        elif len(self.refs) == 1:
            pmid = keypoints[ref_ids[0]]
        else:
            raise ValueError("self.refs not valid.")

        ref_point = np.array(pmid)
        keypoints = (keypoints - ref_point).tolist()

        # Dimension 17 x 2 + 2(reference point) [+ 17 (confidences)]
        joint_coordinates = []
        for i, k in enumerate(keypoints):
            if not kptsvalid[i]:
                joint_coordinates.extend([-1, -1])
            else:
                joint_coordinates.extend(k)
        # Normalization allows for scale invariance
        # Only normalize valid entries, since -1 entries would falsify the normalization result
        if self.norm:
            joint_coordinates = self.normalizevec(joint_coordinates, mask=True)

        # Add relative position of pose to descriptor
        height, width = imagesize[0], imagesize[1]
        relative_refpoint = [pmid[0] / width, pmid[1] / height]
        # print(pmid[0], width, pmid[1], height ,pmid[0]/width, pmid[1]/height)
        joint_coordinates.extend(relative_refpoint)

        if addconfidences is not None:
            # Clip values bcose sometime very large
            # Lower clip value if too much effect on search result
            confs = map(lambda y: max(y, 1), addconfidences)
            joint_coordinates.extend(confs)

        logging.debug(joint_coordinates)
        logging.debug("Dimension joint coordinates: {}".format(len(joint_coordinates)))
        return joint_coordinates

    def joint_joint_distances(self, keypoints, kptsvalid, indices_pairs=None):
        # Joint-Joint Distance
        joint_distances = []
        if indices_pairs is None:
            # Dimension 25 over 2 = 300 / 17 over 2 = 136
            # scipy.special.binom(len(keypoints), 2))
            # Only consider unique different joint pairs ,e.g. (1,2) == (2,1)
            for i1 in range(len(keypoints)):
                for i2 in range(len(keypoints)):
                    if i1 < i2:
                        if not kptsvalid[i1] or not kptsvalid[i2]:
                            joint_distances.append(-1)
                        else:
                            joint_distances.append(np.linalg.norm(keypoints[i2] - keypoints[i1]))
        else:
            for start, end in indices_pairs:
                if not kptsvalid[start] or not kptsvalid[end]:
                    joint_distances.append(-1)
                else:
                    joint_distances.append(np.linalg.norm(keypoints[start] - keypoints[end]))

        logging.debug("Dimension joint distances: {}".format(len(joint_distances)))
        if self.norm:
            joint_distances = self.normalizevec(joint_distances, mask=True)
        return joint_distances

    def joint_joint_orientations(self, keypoints, kptsvalid, indices_pairs=None):
        # Joint-Joint Orientation (vector orientation of unit vector from j1 to j2)
        joint_orientations = []
        if indices_pairs is None:
            # Dimension 25 over 2 = 300 / 17 over 2 = 136
            for i1 in range(len(keypoints)):
                for i2 in range(len(keypoints)):
                    if i1 < i2:
                        j1 = keypoints[i1]
                        j2 = keypoints[i2]
                        # Don't compute for unvalid points or points with same coordinates
                        if (j1 == j2).all():
                            joint_orientations.extend([0, 0])
                        elif not kptsvalid[i1] or not kptsvalid[i2]:
                            joint_orientations.extend([-1, -1])
                        else:
                            vec = np.subtract(j2, j1)
                            normvec = vec / np.linalg.norm(vec)
                            normvec = normvec.astype(float)
                            joint_orientations.extend(list(normvec))

        else:
            for start, end in indices_pairs:
                j1 = keypoints[start]
                j2 = keypoints[end]
                # Don't compute for unvalid points or points with same coordinates
                if (j1 == j2).all():
                    joint_orientations.extend([0, 0])
                elif not kptsvalid[start] or not kptsvalid[end]:
                    joint_orientations.extend([-1, -1])
                else:
                    vec = np.subtract(j2, j1)
                    normvec = vec / np.linalg.norm(vec)
                    normvec = normvec.astype(float)
                    # print("vec: ",vec)
                    # print("normvec: ",normvec)
                    joint_orientations.extend(list(normvec))
                    # plt.axis('equal')
                    # plt.plot([j1[0], j2[0]], [j1[1], j2[1]], 'k-', lw=1)
                    # plt.plot([0, vec[0]], [0, vec[1]], 'g-', lw=1)
                    # plt.plot([0, normvec[0]], [0, normvec[1]], 'r-', lw=1)
                    # plt.gca().invert_yaxis()
                    # print("(%s,%s)->(%s,%s)"%(_BODY_PART_MAPPING[k11], _BODY_PART_MAPPING[k12], _BODY_PART_MAPPING[k21], _BODY_PART_MAPPING[k22]))
                    # label = "%s-%s"%(_BODY_PART_MAPPING[start], _BODY_PART_MAPPING[end])
                    # print(label)
                    # plt.savefig("/home/althausc/master_thesis_impl/posedescriptors/out/query/11-09_12-42-08/.test/%s.jpg"%label)
                    # plt.clf()

                # plt.gca().invert_yaxis()
                # plt.savefig("/home/althausc/master_thesis_impl/posedescriptors/out/query/11-09_12-42-08/.test/entirebody.jpg")
        logging.debug("Dimension of joint orientations: {}".format(len(joint_orientations)))
        # if self.norm:
        #    joint_orientations = normalizevec(joint_orientations, mask=True)
        return joint_orientations

    def joint_line_distances(self, keypoints, lines, kptsvalid, kpt_line_mapping=None):
        # Joint-Line Distance
        # Modes: 1. calculate distance between lines and all keypoints (kpt_line_mapping = None)
        #       2. calculate distance only for specified lines and specified keypoints (kpt_line_mappints != None)
        joint_line_distances = []

        if kpt_line_mapping is None:
            # Approx. Dimension: 60 lines * (25-3) joints = 1320 / (16+6+15) lines * (17-3) joints = 518
            # print("lines.keys(): ", lines.keys())
            # print("keypoints: ", keypoints)
            for k, l in lines.items():
                coords = list(l.coords)
                # print("coords: ", coords)
                # print(k)
                for i, joint in enumerate(keypoints):
                    # if joint is the same as either start or end point of the line, continue
                    if i in k:
                        continue
                    if not kptsvalid[i] or not kptsvalid[k[0]] or not kptsvalid[k[1]]:
                        joint_line_distances.append(-1)
                        # print('%s->(%s,%s) %f'%(_BODY_PART_MAPPING[i],_BODY_PART_MAPPING[k[0]],_BODY_PART_MAPPING[k[1], -1]))
                    else:
                        joint_line_distances.append(Point(joint).distance(l))
                        # print('%s->(%s,%s) %f'%(_BODY_PART_MAPPING[i],_BODY_PART_MAPPING[k[0]],_BODY_PART_MAPPING[k[1]], Point(joint).distance(l)))
        else:
            for k, [(k1, k2), label] in dict_to_item_list(kpt_line_mapping):
                logging.debug(
                    "%s->(%s,%s)" % (self.body_part_mapping[k], self.body_part_mapping[k1], self.body_part_mapping[k2])
                )

                if not kptsvalid[k] or not kptsvalid[k1] or not kptsvalid[k2]:
                    joint_line_distances.append(-1)
                    continue
                if (k1, k2) in lines.keys():
                    joint_line_distances.append(Point(keypoints[k]).distance(lines[(k1, k2)]))
                    # print(Point(keypoints[k]).distance(lines[(k1,k2)]))
                elif (k2, k1) in lines.keys():
                    joint_line_distances.append(Point(keypoints[k]).distance(lines[(k2, k1)]))
                    # print(Point(keypoints[k]).distance(lines[(k2,k1)]))
                else:
                    logging.debug("Not found line: {}{}".format(k1, k2))
                    print("Not found")

        logging.debug("Dimension joint line distances: {}".format(len(joint_line_distances)))
        if self.norm:
            joint_line_distances = self.normalizevec(joint_line_distances, mask=True)
        return joint_line_distances

    def line_line_angles(self, lines, kptsvalid, line_line_mapping=None):
        # Line-Line Angle
        line_line_angles = []

        if line_line_mapping is None:
            # (25-1) over 2 = 276 / (17-1) over 2 = 120
            finished = []
            for (k11, k12), l1 in lines.items():
                for (k21, k22), l2 in lines.items():
                    if not kptsvalid[k11] or not kptsvalid[k12] or not kptsvalid[k21] or not kptsvalid[k22]:
                        line_line_angles.append(-1)
                        continue
                    # skip self-angle and already calculated angles of same lines
                    if (
                        (k21, k22) == (k11, k12)
                        or [(k11, k12), (k21, k22)] in finished
                        or [(k21, k22), (k11, k12)] in finished
                    ):
                        continue
                    line_line_angles.append(angle(l1, l2))
                    finished.append([(k11, k12), (k21, k22)])
        else:
            k = 0
            for (k11, k12), [(k21, k22), label] in dict_to_item_list(line_line_mapping):
                if not kptsvalid[k11] or not kptsvalid[k12] or not kptsvalid[k21] or not kptsvalid[k22]:
                    line_line_angles.append(-1)
                    continue
                l1 = lines[(k11, k12)] if (k11, k12) in lines.keys() else lines[(k12, k11)]
                l2 = lines[(k21, k22)] if (k21, k22) in lines.keys() else lines[(k22, k21)]
                # an= angle(l1,l2)
                # print(k,"angle between ", l1,l2, "is", an, math.degrees(an))
                # print("(%s,%s)->(%s,%s)"%(_BODY_PART_MAPPING[k11], _BODY_PART_MAPPING[k12], _BODY_PART_MAPPING[k21], _BODY_PART_MAPPING[k22]))
                # [p11,p12] = list(l1.coords)
                # [p21,p22] = list(l2.coords)
                # plt.axis('equal')
                # plt.plot([p11[0], p12[0]], [p11[1], p12[1]], 'k-', lw=1)
                # plt.plot([p21[0], p22[0]], [p21[1], p22[1]], 'k-', lw=1)
                # print(k,"angle between ", l1,l2, "is", an, math.degrees(an))
                # print((k11,k12), (k21,k22), (k11,k12) in lines.keys(), (k21,k22) in lines.keys())
                line_line_angles.append(angle(l1, l2))
                # plt.gca().invert_yaxis()
                # plt.savefig("/home/althausc/master_thesis_impl/posedescriptors/out/query/11-09_12-42-08/.test/angle%d.jpg"%k)
                # plt.clf()
                # k += 1

        logging.debug("Dimension line line angles: {}".format(len(line_line_angles)))
        # if self.norm: #not reasonable for angle vector
        #    #line_line_angles = normalizevec(line_line_angles, rangemin=0, rangemax=math.pi)
        #    line_line_angles = normalizevec(line_line_angles, mask=True)
        return line_line_angles

    def joint_plane_distances(self, keypoints, planes):
        # Joint-Plane Distance
        # Dimensions (25-3)*8 = 176
        joint_plane_distances = []
        for joint in keypoints:
            for plane in planes:
                if tuple(joint) not in list(plane.exterior.coords):
                    joint_plane_distances.append(Point(joint).distance(plane))
                # To provide static descriptor dimension
                else:
                    joint_plane_distances.append(0)
        logging.debug("Dimension joint plane distances: {}".format(len(joint_plane_distances)))
        return normalizevec(joint_plane_distances)

    def lines_direct_adjacent(self, keypoints):
        # Directly adjacent lines (linemapping of self.keypoints_lines), Dimesions: 25 - 1 / 17-1 = 16
        # Lines pointing outwards for ordering
        lines = {}
        for start, end in self.keypoints_lines:
            lines.update({(start, end): LineString([keypoints[start], keypoints[end]])})
        logging.debug("\tDimension lines between directly adjacent keypoints: {}".format(len(lines)))
        return lines

    def lines_endjoints_depth2(self, keypoints):
        # Kinetic chain from end joints of depth 2
        # Lines pointing outwards for ordering
        # Dimension (BODY_MODEL_25): 8 / 6
        lines = {}
        for (begin, end) in self.end_joints_depth_2:
            lines.update({(begin, end): LineString([keypoints[begin], keypoints[end]])})
        logging.debug("\tDimension lines from end-joints of depth 2: {}".format(len(lines)))
        return lines

    def lines_endjoints(self, keypoints, indices_pairs=None):
        # Lines only between end joints
        # Lines starting with highest point for ordering
        # when no such, then from left to right
        lines = {}
        if indices_pairs is None:
            # Dimensions: num(self.end_joints) over 2 , BODY_MODEL_25: 8 over 2 = 28 / 6 over 2 = 15
            # Add left-to right of same height first
            for (k1, k2) in [(4, 3), (10, 9), (16, 15)]:
                lines.update({(k1, k2): LineString([keypoints[k1], keypoints[k2]])})
            for (k1, label1) in self.end_joints.items():
                for (k2, label2) in self.end_joints.items():
                    if k1 != k2 and (k2, k1) not in lines and (k1, k2) not in lines:
                        lines.update({(k1, k2): LineString([keypoints[k1], keypoints[k2]])})
        else:
            for start, end in indices_pairs:
                if start in self.end_joints.keys() and end in self.end_joints.keys():
                    lines.update({(start, end): LineString([keypoints[start], keypoints[end]])})
                else:
                    raise ValueError("Given index is no end joint.")
        logging.debug("\tDimension lines between end-joints only: {}".format(len(lines)))
        return lines

    def lines_custom(self, keypoints):
        # Lines which are considered important and not computed by other line methods,
        # Important: some lines are needed for angle computations
        # Lines starting with highest point for ordering
        # when no such, then from left to right
        kpt_mapping = {
            14: 15,
            13: 16,  # foot knees x
            0: [5, 6],  # shoulders-nose
            5: 8,
            6: 7,  # elbows-shoulders x
            3: 5,
            4: 6,  # shoulders-ears
        }
        lines = {}
        for k1, k2 in dict_to_item_list(kpt_mapping, level=0):
            lines.update({(k1, k2): LineString([keypoints[k1], keypoints[k2]])})
        logging.debug("\tDimension lines custom: {}".format(len(lines)))
        return lines

    def filterKeypoints(self, pose_keypoints, mode, threshold=None):
        # Filter out keypoints above a treshold
        # Skip pose if too few keypoints or reference point is not contained (for keypoint coordinate descriptor)
        # Probability = confidence score for the keypoint (not visibility flag!)

        if threshold is None:
            threshold = self.threshold
        # print(pose_keypoints)
        if mode == "JcJLdLLa_reduced":  # TODO: plus Jc_rel?
            for ref in self.refs.keys():
                if pose_keypoints[ref * 3 + 2] <= threshold:
                    return False
        c = 0
        for idx in range(0, self.num_keypoints):
            x, y, prob = pose_keypoints[idx * 3 : idx * 3 + 3]
            # prob: 0 (no keypoint) or low value (not sure)
            if prob <= threshold:
                pose_keypoints[idx * 3 : idx * 3 + 3] = [-1, -1, -1]
            else:
                c = c + 1
        logging.debug("Not valid keypoints: {}".format(self.num_keypoints - c))
        if c >= self.min_keypoints:
            return True
        else:
            return False

    def __call__(self, keypoints, mode, metadata):

        # Get the geometric pose descriptor based on:
        #   -selected mode
        #   -slected reference point(s)
        xs = keypoints[::3]
        ys = keypoints[1::3]
        cs = keypoints[2::3]

        kpts_valid = [False if x == -1 or y == -1 else True for x, y in zip(xs, ys)]
        logging.debug("kpts_valid: {}".format(kpts_valid))

        # Delete background label if present & included in keypoint list
        if len(xs) == len(self.body_part_mapping) and len(ys) == len(self.body_part_mapping):
            if "Background" in self.inv_body_part_mapping:
                del xs[self.inv_body_part_mapping["Background"]]
                del ys[self.inv_body_part_mapping["Background"]]
                print("Info: Deleted background label.")

        keypoints = np.asarray(list(zip(xs, ys)), np.float32)

        # Construct different connection lines from keypoints
        l_direct_adjacent = self.lines_direct_adjacent(keypoints)
        l_end_depth2 = self.lines_endjoints_depth2(keypoints)
        l_endjoints = self.lines_endjoints(keypoints)
        l_custom = self.lines_custom(keypoints)
        # Merge all lines
        l_all = {}
        l_all.update(l_direct_adjacent)
        l_all.update(l_end_depth2)
        l_all.update(l_endjoints)
        l_all.update(l_custom)

        logging.debug("l_direct_adjacent:", l_direct_adjacent)
        logging.debug("l_end_depth2:", l_end_depth2)
        logging.debug("line_endjoints:", l_endjoints)
        logging.debug("l_custom:", l_custom)
        logging.debug("l_custom_all:", len(l_all))

        pose_descriptor = []

        if mode == "JcJLdLLa_reduced":
            # Descriptor contains all joint coordinates, selected joint-line distances and selected line-line angles
            # Result dimension: 64

            # Dimensions: 17 keypoints, 17x2 values (with or without visibility flag)
            #            + 1 reference keypoint, 1x2 values
            joint_coordinates = self.joint_coordinates_rel(
                keypoints, kpts_valid, metadata["imagesize"], addconfidences=None
            )
            # joint_coordinates = joint_coordinates_rel(keypoints, ref_point.tolist(), visiblities = vs , vclipping = True)
            pose_descriptor.append(joint_coordinates)

            # Dimensions: 18 distances
            kpt_line_mapping = {
                7: [(5, 9), "left_arm"],
                8: [(6, 10), "right_arm"],
                3: [(5, 0), "shoulder_head_left"],
                4: [(6, 0), "shoulder_head_right"],
                6: [[(8, 5), "shoulders_elbowr"], [(10, 4), "endpoints_earhand_shoulder_r"]],
                5: [[(6, 7), "shoulders_elbowsl"], [(3, 9), "endpoints_earhand_shoulder_l"]],
                13: [[(14, 15), "knees_foot_side"], [(11, 15), "left_leg"]],
                14: [[(13, 16), "knees_foot_side"], [(12, 16), "right_leg"]],
                10: [(5, 9), "arms_left_side"],
                9: [(6, 10), "arms_right_side"],
                0: [[(16, 12), "headpos_side"], [(15, 11), "headpos_side"]],
                11: [(15, 9), "endpoints_foodhand_hip_l"],
                12: [(10, 16), "endpoints_foodhand_hip_r"],
            }
            JL_d = self.joint_line_distances(keypoints, l_all, kpts_valid, kpt_line_mapping)
            pose_descriptor.append(JL_d)

            # Dimensions: 12 angles
            line_line_mapping = {
                (10, 9): [
                    [(9, 15), "rhand_lhandrfoot"],
                    [(9, 16), "rhand_lhandlfoot"],
                    [(10, 16), "lhand_rhandlfoot"],
                    [(10, 15), "lhand_rhandrfoot"],
                ],
                (5, 11): [(5, 9), "hand_shoulder_hip_l"],
                (6, 12): [(6, 10), "hand_shoulder_hip_r"],
                (6, 8): [(5, 7), "upper_arms"],
                (8, 10): [(7, 9), "lower_arms"],
                (12, 14): [(11, 13), "upper_legs"],
                (14, 16): [(13, 15), "lower_legs"],
                (0, 5): [(3, 5), "head_shoulder_l"],
                (0, 6): [(4, 6), "head_shoulder_r"],
            }
            LL_a = self.line_line_angles(l_all, kpts_valid, line_line_mapping)

            pose_descriptor.append(LL_a)

        elif mode == "JLd_all_direct":
            # Descriptor contains all joint-line distances
            # Specify which lines should be used, either:
            #   - all lines: Z * 17 keypoints = ..
            #   - adjacent lines: 14*17 = 238
            #   - endjoint connection lines: X*17 = ..
            #   - endjoint depth 2 connection lines: Y*17 = ..

            JL_d = self.joint_line_distances(keypoints, l_direct_adjacent, kpts_valid)
            JL_d = [entry for k, entry in enumerate(JL_d) if k % 2 == 0]
            pose_descriptor.append(JL_d)

        elif mode == "JJo_reduced":
            # Descriptor contains normalized joint-joint orientations
            # Used: all limbs in COCO keypoint model (direction facing outwards) = 16
            #       + lines between endjoints and depth 2 from endjoints = 6
            #       + custom lines
            custom_jj_mapping = [(9, 15), (10, 16), (0, 6), (0, 5)]  # endjoints  # head-shoulders
            kpt_kpt_mapping = []
            kpt_kpt_mapping.extend(self.keypoints_lines)
            kpt_kpt_mapping.extend(self.end_joints_depth_2)
            kpt_kpt_mapping.extend(custom_jj_mapping)

            JJ_o = self.joint_joint_orientations(keypoints, kpts_valid, kpt_kpt_mapping)
            pose_descriptor.append(JJ_o)

        elif mode == "Jc_rel":
            Jc_rel = self.joint_coordinates_rel(keypoints, kpts_valid, metadata["imagesize"], addconfidences=None)
            pose_descriptor.append(Jc_rel)
        else:
            print("Unknown gpd type")

        # indices_pairs = []
        # JJ_d = joint_joint_distances(keypoints,indices_pairs=None)
        # pose_descriptor.append(JJ_d)

        # indices_pairs = []
        # JJ_o = joint_joint_orientations(keypoints, indices_pairs=None)
        # pose_descriptor.append(JJ_o)

        # Add clipped score value
        # score = max(0,min(1,score))
        # pose_descriptor.append([score])

        # Planes for selected regions: 2 for each head, arms, leg & foot region
        # plane_points = [{1: 'Neck',0: 'Nose', 18: 'LEar'}, {1: 'Neck',0: 'Nose', 18: 'REar'}, {2: 'RShoulder', 3: 'RElbow', 4: 'RWrist'},
        #                {5: 'LShoulder', 6: 'LElbow',7: 'LWrist'}, {12: 'LHip', 13: 'LKnee', 14: 'LAnkle'}, { 9: 'RHip', 10: 'RKnee', 11: 'RAnkle'},
        #                {13: 'LKnee', 14: 'LAnkle', 19: 'LBigToe'}, {10: 'RKnee', 11: 'RAnkle', 22: 'RBigToe'}]

        # planes = get_planes(plane_points)
        # JP_d = joint_plane_distances(keypoints, planes)
        logging.debug(pose_descriptor)
        pose_descriptor = [item for sublist in pose_descriptor for item in sublist]
        for v in pose_descriptor:
            if math.isnan(v):
                print("Detected NaN value in final descriptor: ")
                print(pose_descriptor)
                exit(1)
        # logging.debug(pose_descriptor)
        logging.debug("\nDimension of pose descriptor: {}".format(len(pose_descriptor)))
        # flatten desriptor

        logging.debug("Dimension of pose descriptor flattened: {}".format(len(pose_descriptor)))
        logging.debug("\n")

        mask = "".join(["1" if entry != -1 else "0" for entry in pose_descriptor])

        return pose_descriptor, cs, mask


def flat_keypoints(keypoints, scores, labels, threshold):
    filtered_keypoints = np.zeros([17, 3], dtype=np.float)
    for keypoint, score, label in zip(keypoints, scores, labels):
        # check if we already found a better prediction
        if filtered_keypoints[label, 2] > score:
            continue
        if score < threshold:
            continue
        filtered_keypoints[label, :2] = keypoint
        filtered_keypoints[label, 2] = score

    filtered_keypoints = filtered_keypoints.flatten()

    return filtered_keypoints


def jsonl_data_iter(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def gdp_input_iter(data, args):
    for d in data:
        yield {"data": d, "args": args}


def compute_gpd(args):
    d = args["data"]
    gpd = args["args"]["gpd"]
    gpd_mapping = args["args"]["gpd_mapping"]
    features = []
    for keypoints, scores, labels, size in zip(d["keypoints"], d["scores"], d["labels"], d["origin_size"]):

        # print(f"{keypoints}  {scores}  {labels}")
        keypoints = flat_keypoints(keypoints, scores, labels, 0.9)
        feature_type = []

        for gpd_type, _ in gpd_mapping.items():
            if not gpd.filterKeypoints(keypoints, gpd_type):
                print("SKIP")
                continue
            # print(gpd_type)
            pose_descriptor, cs, mask = gpd(keypoints, gpd_type, {"imagesize": size})

            feature_type.append({"type": gpd_type, "feature": pose_descriptor, "mask": mask})
        # print(feature_type)
        # exit()
    features.append(feature_type)
    output = {**d, "features": features}
    return output


def generete_gdb(path, args):
    with mp.Pool(16) as p:
        for r in p.imap(compute_gpd, gdp_input_iter(jsonl_data_iter(path), args)):
            yield r


def main():
    args = parse_args()
    gpd = GlobalPoseDescriptor(args.threshold, args.min_keypoints)
    gpd_mapping = {"JcJLdLLa_reduced": 66, "JLd_all_direct": 0, "JJo_reduced": 0, "Jc_rel": 0}
    with open(args.output_path, "wb") as f_out:
        for i, gpd in enumerate(generete_gdb(args.poses_path, {"gpd": gpd, "gpd_mapping": gpd_mapping})):

            print(i)
            f_out.write(msgpack.packb(gpd))
        # exit()

    # type = "JcJLdLLa_reduced"
    # with open(args.output_path, "wb") as f_out:
    #     with open(args.poses_path, "r") as f:
    #         for i, line in enumerate(f):
    #             print(i)
    #             d = json.loads(line)
    #             # print(d.keys())
    #             # exit()

    #             features = []
    #             for keypoints, scores, labels, size in zip(d["keypoints"], d["scores"], d["labels"], d["origin_size"]):

    #                 # print(f"{keypoints}  {scores}  {labels}")
    #                 keypoints = flat_keypoints(keypoints, scores, labels, 0.9)
    #                 feature_type = []

    #                 for gpd_type, _ in gpd_mapping.items():
    #                     if not gpd.filterKeypoints(keypoints, gpd_type):
    #                         print("SKIP")
    #                         continue
    #                     # print(gpd_type)
    #                     pose_descriptor, cs, mask = gpd(keypoints, gpd_type, {"imagesize": size})

    #                     feature_type.append({"type": gpd_type, "feature": pose_descriptor, "mask": mask})
    #                 # print(feature_type)
    #                 # exit()
    #             features.append(feature_type)
    #             output = {**d, "features": features}
    #             # print(output)
    #             f_out.write(msgpack.packb(output))

    return 0


if __name__ == "__main__":
    sys.exit(main())