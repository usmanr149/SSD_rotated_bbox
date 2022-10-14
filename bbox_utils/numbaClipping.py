import numpy as np

import numba
from numba import jit

from bbox_utils.bbox import generate_default_boxes

from src.config import *

@jit(nopython=True, parallel=False)
def isRight(x1, y1, x2, y2, x, y):
    R = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    
    return R >= 0

@jit(nopython=True, parallel=False)
def calcSlopeIntercept(x1, y1, x2, y2):
    
    if x2 - x1 == 0:
        return 2**32 - 1, x1
    
    m = ( y2 - y1 ) / ( x2 - x1 )
    
    b = y1 - m * x1
    
    return np.float32(m), np.float32(b)

@jit(nopython=True, parallel=False)
def findIntersectionInRange(m1, b1, m2, b2):
    if m1 == m2:
        return np.array([2**32 - 1, 2**32 - 1], dtype=np.float32)
    
    elif m1 == 2**32 - 1:
        x_intersect = b1
        y_intersect = m2 * x_intersect + b2
    elif m2 == 2**32 - 1:
        x_intersect = b2
        y_intersect = m1 * x_intersect + b1
    elif m1 == 0:
        y_intersect = b1
        x_intersect = ( y_intersect - b2 ) / m2
    elif m2 == 0:
        y_intersect = b2
        x_intersect = ( y_intersect - b1 ) / m1
    else:
        x_intersect = (b2 - b1) / (m1 - m2)
        y_intersect = m1 * x_intersect + b1
    
    return np.array([x_intersect, y_intersect], dtype=np.float32)

@jit(nopython=True, parallel=False)
def clippedPolygon(subjectPolygon, subjectPolygonSize, clipPolygon, clipPolygonSize):
        
    newPolygon = np.empty( (16, 2), dtype=np.float32 )

    inputPolygon = np.empty( (16, 2), dtype=np.float32 )

    # copy subject polygon to new polygon and set its size
    for i in numba.prange(subjectPolygonSize):
        newPolygon[i] = subjectPolygon[i]

    newPolygonSize = subjectPolygonSize

    for j in numba.prange(clipPolygonSize):
        # copy new polygon to input polygon & set counter to 0
        for k in numba.prange(newPolygonSize):
            inputPolygon[k] = newPolygon[k]
        counter = 0

        # get clipping polygon edge
        cp1 = clipPolygon[j];
        cp2 = clipPolygon[(j - 1) % clipPolygonSize];

        for i in numba.prange(newPolygonSize):
            # get subject polygon edge
            s = inputPolygon[i];
            e = inputPolygon[(i - 1) % newPolygonSize];


            m1, b1 = calcSlopeIntercept(e[0], e[1], s[0], s[1])
            m2, b2 = calcSlopeIntercept(cp2[0], cp2[1], cp1[0], cp1[1])

            interscecting_point = findIntersectionInRange(m1, b1, m2, b2)

            if isRight(cp1[0], cp1[1], cp2[0], cp2[1], s[0], s[1]):
                if not isRight(cp1[0], cp1[1], cp2[0], cp2[1], e[0], e[1]):
                    newPolygon[counter] = interscecting_point
                    counter+=1
                newPolygon[counter] = [s[0], s[1]]
                counter+=1
            elif isRight(cp1[0], cp1[1], cp2[0], cp2[1], e[0], e[1]):
                newPolygon[counter] = interscecting_point
                counter+=1

        newPolygonSize = counter;
    return newPolygon[:newPolygonSize]

@jit(nopython=True, parallel=False)
def shoelaceFormula(coordinates):
    if len(coordinates) < 3:
        return np.float32(0)
    
    area = 0

    for i in range(len(coordinates)):
        i_plus_1 = (i + 1) % len(coordinates)
        area += (coordinates[i_plus_1][0] * coordinates[i][1] - coordinates[i][0] * coordinates[i_plus_1][1])

    return np.float32(abs(area) / 2)

@jit(nopython=True, parallel=False)
def getIOU(default_box, gt_box, gt_box_area):
    default_box = default_box.reshape(4, 2)

    
    clipped_polygon = clippedPolygon(default_box, len(default_box), gt_box, len(gt_box))
        
    if len(clipped_polygon) == 0:
        return 0
        
    default_box_area = shoelaceFormula(default_box)
    clipped_polygon_area = shoelaceFormula(clipped_polygon)
    
    IOU = clipped_polygon_area / ( default_box_area + gt_box_area - clipped_polygon_area )
    
    return np.float32(IOU)

# Parallel version
@jit(nopython=True, parallel=True)
def getIOUOverallDefaultBox(prior_boxes, gt_box, prior_boxes_coco, gt_boxes_coco, prior_box_area):

    c_x = gt_boxes_coco[0]
    c_y = gt_boxes_coco[1]
    rot_angle = gt_boxes_coco[4]

    if rot_angle >= -0.7853981633974483 and rot_angle <= 0.7853981633974483: # 45 deg in rad
        width = gt_boxes_coco[2]
        height = gt_boxes_coco[3]
    else:
        width = gt_boxes_coco[3]
        height = gt_boxes_coco[2]

    # need to take care of the edge case where width is very small and thus the right and left limit are very small
    # so that no default box overlaps
    if 1.5 * width < (30 / 300):
        l = (30 / 300)
    else:
         l = 1.5 * width
    left_limit = c_x - l
    right_limit = c_x + l
    c_x_in_bound = np.where(
        (prior_boxes_coco[:,0 ] >= left_limit) & (prior_boxes_coco[:,0 ] <= right_limit), True, False)

    if 1.5 * height < (30 / 300):
        l = (30 / 300)
    else:
         l = 1.5 * height
    up_limit = c_y - l
    down_limit = c_y + l
    c_y_in_bound = np.where(
        (prior_boxes_coco[:,1 ] >= up_limit) & (prior_boxes_coco[:,1 ] <= down_limit), True, False)

    c_in_bound = np.logical_and(c_x_in_bound, c_y_in_bound)

    gt_box = gt_box.reshape(4, 2)
    gt_area = width * height

    # if the gt_area is smaller than the smallest anchorbox
    min_d_area =  np.min(prior_box_area)
    if gt_area < min_d_area:
        low = 0
        high = (min_d_area / gt_area) * 2
    else:
        low = 0.4
        high = 2.5

    area_bound = np.where( (prior_box_area >= low * gt_area) & (prior_box_area <= high * gt_area), True, False)

    in_bound = np.logical_and(c_in_bound, area_bound)

    idx = np.argwhere(in_bound)

    iou = np.zeros(len(prior_boxes), dtype = np.float32)

    for i in numba.prange(len(idx)):
        iou[idx[i][0]] = getIOU(prior_boxes[idx[i][0]], gt_box, gt_area)
        
    return iou

if __name__ == '__main__':

    prior_boxes, prior_boxes_coco, prior_box_area = generate_default_boxes(feature_map_shapes, 6, aspect_ratios, angles)

    gt_box = [0.03333333, 0.16666667, 0.83333333, 0.16666667, 0.83333333,
        0.96666667, 0.03333333, 0.96666667]

    gt_box = np.array(gt_box, dtype=np.float32)
    prior_boxes= np.array(prior_boxes, dtype=np.float32)

    result = getIOUOverallDefaultBox(prior_boxes, gt_box, prior_boxes_coco, gt_boxes_coco, prior_box_area)

    print(result)