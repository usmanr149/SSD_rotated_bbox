import numpy as np

from src.config import s_max, s_min

def rotateTheta(x, y, theta, theta_type='rad'):
    """
        Input:
            x: x-coordinate of the point
            y: y-coordinate of the point
            theta: rotation angle in degrees 
    """

    assert theta_type in ['deg', 'rad'], "theta_type needs to be in 'deg' or 'rad' "

    if theta_type == 'deg':
        theta = theta * np.pi / 180

    x_rot_theta  = x * np.cos(theta) - y * np.sin(theta)
    y_rot_theta = y * np.cos(theta) + x * np.sin(theta)
    
    return x_rot_theta, y_rot_theta

def rotateRectangle(c_x, c_y, width, height, theta, theta_type='rad'):
    """
        Input:
            c_x: x-center of rectangle
            c_y: y-center of rectangle
            width: width of rectangle
            height: height of rectangle
            theta: rotation angle in degrees
        Return:
            [
                x1, y1m x2, y2, x3, y3, x4, y4
            ]
            The corners of the rotated rectangle in clockwise fashion
            x1, y1----------------x2, y2
            |                          |
            |                          |
            |                          |
            x4, y4----------------x3, y3

    """

    rotRect = []
    
    corners = [
        [-width / 2, -height / 2],
        [width / 2, -height / 2],
        [width / 2, height / 2],
        [-width / 2, height / 2]
    ]
    
    for corner in corners:
        x_prime, y_prime = rotateTheta(corner[0], corner[1], theta, theta_type)
        rotRect.append(c_x + x_prime)
        rotRect.append(c_y + y_prime)
    
    return rotRect


def calculate_scale_of_default_boxes(k, m, s_max = 0.8, s_min = 0.2):
    """
    m = number_of_feature_maps
    s_k = s_min + (s_max - s_min) * (k - 1)/(m - 1)
    width_k = s_k * sqrt(aspect_ratio)
    height_k = s_k / sqrt(aspect_ratio)
    """
    return s_min + (s_max - s_min) * (k - 1) / (m - 1)

def generate_default_boxes(feature_map_shapes, number_of_feature_maps, aspect_ratios, angles):
    """
    Input:
        feature map shapes for VGG: [38, 19, 10, 5, 3, 1]
        number_of_feature_maps: len(number_of_feature_maps)
        aspect_ratios: an array of asepect ratio for each feature map
                for VGG16
                [
                    [1/2, 1,3],
                    [1/2, 1/3],
                    [1/2, 1/3],
                    [1/2, 1/3],
                    [1/2],
                    [1/2]
                ]
        angles: angle for rectangle (non-squares)
    Output:

    """

    assert len(feature_map_shapes) == number_of_feature_maps, 'number of feature maps needs to be {0}'.format(len(feature_map_shapes))
    assert len(feature_map_shapes) == len(aspect_ratios), 'Need aspect ratios for all feature maps'

    prior_boxes = []
    prior_boxes_coco = []
    prior_box_area = []

    for k, f_k in enumerate(feature_map_shapes):
        s_k = calculate_scale_of_default_boxes(k, m = number_of_feature_maps, s_max=s_max, s_min=s_min)
        s_k_prime = np.sqrt(s_k * calculate_scale_of_default_boxes(k + 1, m = 6, s_max=s_max, s_min=s_min))
        for i in range(f_k):
            for j in range(f_k):
                cx = (i + 0.5) / f_k
                cy = (j + 0.5) / f_k

                 # for the square box don't rotate by 90
                for angle in angles[:-1]:
                    prior_boxes_coco.append([cx, cy, s_k, s_k, angle * np.pi / 180])
                    prior_boxes.append(rotateRectangle(cx, cy, s_k, s_k, angle, 'deg'))
                    prior_box_area.append(s_k * s_k)

                # aspect ratio 1
                for angle in angles:
                    prior_boxes_coco.append([cx, cy, s_k, s_k_prime, angle * np.pi / 180])
                    prior_boxes.append(rotateRectangle(cx, cy, s_k, s_k_prime, angle, 'deg'))
                    prior_box_area.append(s_k * s_k_prime)

                for ar in aspect_ratios[k]:
                    for angle in angles:
                        prior_boxes_coco.append([cx, cy, s_k*np.sqrt(ar), s_k/np.sqrt(ar), angle * np.pi / 180])
                        prior_boxes.append(rotateRectangle(cx, cy, s_k*np.sqrt(ar), s_k/np.sqrt(ar), angle, 'deg'))
                        prior_box_area.append( s_k*np.sqrt(ar) * s_k/np.sqrt(ar) )

    return np.array(prior_boxes), np.array(prior_boxes_coco), np.array(prior_box_area)