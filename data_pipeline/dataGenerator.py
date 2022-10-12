import numpy as np
import keras
import tensorflow as tf

from bbox_utils.default_gt_overlap import match_priors_with_gt
from data_pipeline.getData import read_data, resize_images_and_labels

def label_dimensions_normalized(labels, label_type, image_height = 300, image_width = 300):
    """
    input:
        labels: in [[category_label, c_x, c_y, width, height, theta]]
        image_height: height of image
        image_width: width of image
    output:
        label with dimesion divided by with width and height: in [[category_label, c_x, c_y, width, height, theta]]
    """
    if label_type == 'edge':
        return [ [    
                            label, 
                            x1 / image_width, 
                            y1 / image_height, 
                            x2 / image_width, 
                            y2 / image_height, 
                            x3 / image_width, 
                            y3 / image_height, 
                            x4 / image_width, 
                            y4 / image_height, 
                        ] 
                        for label, x1, y1, x2, y2, x3, y3, x4, y4 in labels ]
    elif label_type == 'coco':
        return [ [ label, c_x / image_width, c_y / image_height, width / image_width, height /image_height, theta ] 
                    for label, c_x, c_y, width, height, theta in labels ]

class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs,
                label_folder_path,
                image_folder_path, 
                prior_boxes,
                prior_boxes_coco_form,
                batch_size = 8, 
                n_classes = 5, 
                image_height = 300,
                image_width = 300,
                normalize = True,
                shuffle = True,
                image_extension = '.png',
                training = True):
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.label_folder_path = label_folder_path
        self.image_folder_path = image_folder_path
        self.prior_boxes = prior_boxes
        self.prior_boxes_coco_form = prior_boxes_coco_form
        self.image_height = image_height
        self.image_width = image_width
        self.normalize = normalize
        self.image_extension = image_extension
        self.training = training
        self.on_epoch_end()

        """
        Inputs:
            list_IDs: name of files used to look data in label_folder_path and image_folder_path
            label_folder_path: path to where labels are stored, need to be in .txt format
            image_folder_path: path to where images are stored, need to be in png
            prior_boxes: precalculated prior boxes in (x1, y1, x2, y2, x3, y3, x4, y4)
            prior_boxes_coco_form: precomputed prior boxes in (c_x, c_y, width, height, theta)
            batch_size: int
            n_classes: number of classes in the dataset, don't include background
        """

    def __len__(self):
        return int( np.floor( len(self.list_IDs) / self.batch_size ) )

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle:
            np.random.shuffle(self.indexes)
        
    def __data_generation(self, list_IDs_temp):

        X = np.empty([self.batch_size, self.image_height, self.image_width, 3])
        y_label = None
        y_loc = None

        for i, file_name in enumerate(list_IDs_temp):
            image, labelled_gt_box_coords, labelled_gt_coco = read_data(  file_name, 
                                        self.image_folder_path, 
                                        self.label_folder_path,
                                        self.image_extension
                                     )

            # take care of images with no labels
            # if no label then the whole image is a background
            if len(labelled_gt_box_coords) == 0:
                labelled_gt_box_coords = [[0, 0, 0 , image.shape[1], 0, image.shape[1], image.shape[0], 0, image.shape[0]]]
                labelled_gt_coco = [[0, image.shape[1] / 2, image.shape[0] / 2, image.shape[1], image.shape[0]]]

            image, labelled_gt_box_coords, labelled_gt_coco = resize_images_and_labels(image, labelled_gt_box_coords, 
                                                                    labelled_gt_coco, self.image_height, self.image_width)
            if self.normalize:
                X[i,] = image / 255

            labelled_gt_box_coords_normallized = label_dimensions_normalized(labelled_gt_box_coords, 
                                                    'edge', 
                                                    self.image_height, 
                                                    self.image_width)

            labelled_gt_box_coco_coords_normallized = label_dimensions_normalized(labelled_gt_coco, 
                                                    'coco', 
                                                    self.image_height, 
                                                    self.image_width)

            labels_map = {1: 1, 3: 2, 17: 3, 18: 4}
            gt_labels = [labels_map[int(l[0])] for l in labelled_gt_box_coords_normallized]
            gt_boxes_normalized = [l[1:] for l in labelled_gt_box_coords_normallized]
            gt_boxes_coco = [l[1:] for l in labelled_gt_box_coco_coords_normallized]
            
            offset, one_hot_encoded_label = match_priors_with_gt(   
                                                            self.prior_boxes, 
                                                            self.prior_boxes_coco_form, 
                                                            gt_boxes_normalized, 
                                                            gt_boxes_coco,
                                                            tf.constant(gt_labels), 
                                                            number_of_labels = self.n_classes + 1, 
                                                            threshold = 0.5)

            if y_label == None:
                y_label = one_hot_encoded_label
                y_loc = offset
            else:
                y_label = tf.concat([y_label, one_hot_encoded_label], axis = 0)
                y_loc = tf.concat([y_loc, offset], axis = 0)

        return X, [y_loc, y_label]

if __name__ == '__main__':
    # dg = DataGenerator(list_IDs, 
    #                label_path,
    #                image_path, 
    #                prior_boxes,
    #                boxes)
    pass