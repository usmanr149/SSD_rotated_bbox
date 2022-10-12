import os
import cv2

def readLabels(file_name):
    file = open(file_name)
    read_lines = file.readlines()

    labels = []
    for line in read_lines:
        line = line.strip().split(',')
        # 0 is reserved for background
        labels.append([ int(line[0]) ] + list(map(float, line[1:])) )
    
    return labels


def read_data(file_name, image_path, label_path, image_extension='.png'):
    """
    inputs:
        file_name: name of file without type. Image should be in JPG on PNG format
        image_path: image will be read from this path
        label_path: labels for the image will be read from here
    output:
        image: as a numpy array in rgb format
        label: in [[category_label, x1, y1, x2, y2, x3, y3, x4, y4]]
        label_coco: [[category_label, c_x, c_y, width, height, theta]]
    """
    image_path = os.path.join(image_path, file_name + image_extension)
    label_file = os.path.join(label_path, file_name + '.txt')
    label_file_coco = os.path.join(label_path, file_name + '.txt.coco_theta')

    assert os.path.exists(image_path), "{0} image not found".format(image_path)
    assert os.path.exists(label_file), "{0} label not found".format(label_file)
    assert os.path.exists(label_file_coco), "{0} label not found".format(label_file_coco)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # read the 8 point labels
    labels = readLabels(label_file)

    # read the coco labels
    labels_coco = readLabels(label_file_coco)

    return image, labels, labels_coco

def resize_images_and_labels(image, labels, labels_coco, image_height = 300, image_width = 300):
    """
    inputs:
        image: image as numpy array RGB
        label: in [[category_label, x_min, y_min, x_max, y_max]]
        image_height: desired height of image
        image_width: desired width of image
    """

    original_image_height, original_image_width, _ = image.shape

    image = cv2.resize(image, (image_width, image_height))

    labels = [ [    label, 
                    float(x1 * image_width / original_image_width), 
                    float(y1 * image_height / original_image_height), 
                    float(x2 * image_width / original_image_width), 
                    float(y2 * image_height / original_image_height),
                    float(x3 * image_width / original_image_width), 
                    float(y3 * image_height / original_image_height), 
                    float(x4 * image_width / original_image_width), 
                    float(y4 * image_height / original_image_height)
                     ] 
                for label, x1, y1, x2, y2, x3, y3, x4, y4 in labels ]

    labels_coco = [ [label, 
                    c_x * image_width / original_image_width,
                    c_y * image_height / original_image_height,
                    width * image_width / original_image_width,
                    height * image_height / original_image_height,
                    theta
                    ]
                for label, c_x, c_y, width, height, theta in labels_coco]

    return image, labels, labels_coco