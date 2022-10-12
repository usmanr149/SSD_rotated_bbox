import glob
import os
import argparse

import numpy as np

from train_test_val import writeToFile

def EuclideanDistance(p1, p2):
    return np.sqrt( (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 )

def calculateSlope(x1, y1, x2, y2):
    if x1 == x2:
        return np.pi / 2
    elif y1 == y2:
        return 0
    m = (y1 - y2) / ( x1 - x2 )

    return np.arctan(m)

def convert8pointsto5(points):
    """
        Input:
            points: rectangle corner point in clockwise format
        Output:
            (c_x, c_y, w, h, theta)
    """
    A = (points[0], points[1])
    B = (points[2], points[3])
    D = (points[6], points[7])
    
    distance1, distance2 = EuclideanDistance(A, B), EuclideanDistance(A, D)
    
    if distance1 > distance2:
        height = distance1
        width = distance2
        angle = calculateSlope(A[0], A[1], D[0], D[1])
    else:
        width = distance1
        height = distance2
        angle = calculateSlope(A[0], A[1], B[0], B[1])

    c_x = ( max(points[0], points[2], points[4], points[6]) + min(points[0], points[2], points[4], points[6]) ) / 2.
    c_y = ( max(points[1], points[3], points[5], points[7]) + min(points[1], points[3], points[5], points[7]) ) / 2.

    return [c_x, c_y, width, height, angle]


if __name__ == '__main__':

    # python src/create_obb.py -l '/Users/usmanr/workspace/rotatedBoxes/input/train/*.txt'

    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--labels_dir', help='labels directory')

    args = vars(parser.parse_args())

    bbox_files = glob.glob(args['labels_dir'])

    for file in bbox_files:
        print(file)
        coco_theta_format = []
        with open(file) as f:
            data = f.readlines()

            for corners in data:
                corners = corners.strip().split(',')
                label = corners[0]

                corners = corners[1:]

                corners = list(map(float, corners))

                coco = convert8pointsto5(corners)
                coco_theta_format.append([label] + coco)

        path = file.replace('.txt', '.txt.coco_theta')
        writeToFile(path, coco_theta_format)

