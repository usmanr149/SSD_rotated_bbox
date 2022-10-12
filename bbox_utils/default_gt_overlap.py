import tensorflow as tf
import numpy as np

from bbox_utils.numbaClipping import getIOUOverallDefaultBox

def calculate_offset_from_gt(gt_boxes_mapped_to_prior, prior_boxes):
    prior_boxes = tf.expand_dims(prior_boxes, axis=0)
    g_j_cx = 10 * (gt_boxes_mapped_to_prior[:, :, 0] - prior_boxes[:, :, 0]) / prior_boxes[:, :, 2]
    g_j_cy = 10 * (gt_boxes_mapped_to_prior[:, :, 1] - prior_boxes[:, :, 1]) / prior_boxes[:, :, 3]
    g_j_w = 5 * tf.math.log(gt_boxes_mapped_to_prior[:, :, 2] / prior_boxes[:, :, 2])
    g_j_h = 5 * tf.math.log(gt_boxes_mapped_to_prior[:, :, 3] / prior_boxes[:, :, 3])
    g_theta = tf.math.tan(gt_boxes_mapped_to_prior[:, :, 4] - prior_boxes[:, :, 4])

    offset = tf.concat( [ g_j_cx, g_j_cy, g_j_w, g_j_h, g_theta ] , axis = 0)

    return tf.transpose(tf.expand_dims(offset, axis = 0), perm=[0,2,1])

def match_priors_with_gt(prior_boxes, prior_boxes_coco, gt_boxes, gt_boxes_coco, gt_labels, number_of_labels, threshold = 0.5):
    """
        Input:
            prior_boxes: default boxes in form (x1, y1, x2, y2, x3, y3, x4, y4)
            gt_boxes: ground truth boxes in form (1, number of labels, x1, y1, x2, y2, x3, y3, x4, y4)
        Output:

    """

    IOU_map = np.zeros((1, len(prior_boxes), len(gt_boxes)))

    for i in range(len(gt_boxes)):
        IOU_map[:, :, i] = getIOUOverallDefaultBox(prior_boxes, np.array(gt_boxes[i]))

    # select the box with the highest IOU
    highest_overlap_idx = tf.math.argmax(IOU_map, axis = 1)
    highest_overlap_idx = tf.cast(highest_overlap_idx, tf.int32)
    idx = tf.range(IOU_map.shape[1])
    highest_overlap_idx_map = tf.expand_dims(tf.equal(idx, tf.transpose(highest_overlap_idx)), axis = 0)
    # set the highest overlap to 1
    IOU_map = tf.where(tf.transpose(highest_overlap_idx_map, perm=[0,2,1]), tf.constant(1.0), IOU_map)

    # find the column idx with the highest IOU at each row
    max_IOU_idx_per_row = tf.math.argmax(IOU_map, axis = 2)
    # find the max value per row
    max_IOU_per_row = tf.reduce_max(IOU_map, axis = 2)

    # threshold IOU
    max_IOU_above_threshold = tf.greater(max_IOU_per_row, threshold)

    # map the gt boxes to the prior boxes with the highest overlap
    gt_box_label_map = tf.gather(gt_boxes_coco, max_IOU_idx_per_row, batch_dims = 0)

    gt_box_label_map_offsets = calculate_offset_from_gt(gt_box_label_map, tf.constant(prior_boxes_coco, np.float32))

    # remove from gt_boxes_map where overlap with prior boxes is less than 0.5
    gt_boxes_map_offset_suppressed = tf.where( tf.expand_dims(max_IOU_above_threshold, -1),  
                                        gt_box_label_map_offsets, tf.zeros_like(gt_box_label_map))
    # add a positive condition column for the localization loss
    max_IOU_above_threshold_expand = tf.expand_dims(max_IOU_above_threshold, -1)
    max_IOU_above_threshold_expand = tf.cast(max_IOU_above_threshold_expand, tf.float32)
    gt_boxes_map_offset_suppressed_with_pos_cond = tf.concat([  gt_boxes_map_offset_suppressed, 
                                                                max_IOU_above_threshold_expand ], axis = 2)

                                         
    gt_labels_map = tf.gather(gt_labels, max_IOU_idx_per_row, batch_dims = 0)
    # suppress the label where IOU with the gt boxes is < 0.5
    gt_labels_map_suppressed = tf.where( max_IOU_above_threshold, 
                                        gt_labels_map, tf.zeros_like(gt_labels_map))
    gt_labels_one_hot_encoded = tf.one_hot(gt_labels_map_suppressed, number_of_labels)

    return gt_boxes_map_offset_suppressed_with_pos_cond, gt_labels_one_hot_encoded
    
