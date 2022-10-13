import glob
import json
import numpy as np
import cv2
import os
import argparse
import csv

def writeToFile(file, listToWrite):
    with open(file, "w") as f:
        writer = csv.writer(f)
        writer.writerows(listToWrite)

def getImageId(data, file_name):
    for i in range(len(data['images'])):
        if data['images'][i]['file_name'] == file_name:
            return data['images'][i]['id']
    return None

def getSegments(data, image_id, desired_category = [1,3,17,18]):

    segments = []

    for i in range(len(data['annotations'])):
        if data['annotations'][i]['image_id'] == image_id and data['annotations'][i]['category_id'] in desired_category:
            if type(data['annotations'][i]['segmentation']) == list:
                for pts in data['annotations'][i]['segmentation']:
                    pts = np.array(pts, np.int32).reshape( len(pts) // 2, 2 )

                    pts = pts.reshape((-1, 1, 2))

                    rect = cv2.minAreaRect(pts)
                    box = cv2.boxPoints(rect)
                    label = data['annotations'][i]['category_id']
                    segments.append( [ label ] + list(box.reshape(-1)))

    return segments

def writeBoxDimensions(images_dir, labels_files, output_folder):

    with open(labels_files, 'r') as f:
        data = json.load(f)

    images_list = glob.glob(images_dir)

    images_list = [file_name.split('/')[-1] for file_name in images_list]

    for file_name in images_list:
        # get image_id
        image_id = getImageId(data, file_name)
        if image_id is None:
            return 'filename: {0} not found'.format(file_name)
        
        # get segments of interest for image_id
        segments = getSegments(data, image_id)
        
        path = os.path.join(output_folder, file_name.replace('.jpg', '') + '.txt')

        writeToFile(path, segments)


if __name__ == '__main__':
    # images_dir  = '/Users/usmanr/fiftyone/coco-2014/train/data/*jpg'
    # labels_files = '/Users/usmanr/fiftyone/coco-2014/train/labels.json'

    # categories to keep
    # person: 1
    # car: 3
    # cat: 17
    # dog: 18

    # python src/train_test_val.py -i '/home/usman/coco-2014/train/data/*.jpg' -l /home/usman/coco-2014/train/labels.json -o /home/usman/workspace/SSD_rotated_bbox_input/input/train

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image_dir_path', help='image directory path')
    parser.add_argument('-l', '--label_file_path', help='label files')
    parser.add_argument('-o', '--output_folder', help='output path for outputs')

    args = vars(parser.parse_args())

    writeBoxDimensions(args['image_dir_path'], args['label_file_path'], args['output_folder'])

