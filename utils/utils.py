import numpy as np
import cv2


def overlap_ratio(rect1, rect2):
    """Compute overlap ratio (IoU) between two rects

    Args:
        rect1 (ndarray): 1d array of [x,y,w,h] or 2d array of N x [x,y,w,h]
        rect2 (ndarray): 1d array of [x,y,w,h] or 2d array of N x [x,y,w,h]

    Returns:
        iou (ndarray): overlap ratio (IoU) between two input rects
    """

    # 1d array transform to 2d array by adding a new aixs. [x,y,w,h]->[[x,y,w,h]]
    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)  # The intersection of 'rect1' and 'rect2'
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect  # The union of 'rect1' and 'rect2'
    iou = np.clip(intersect / union, 0, 1)  # if iou lager than 1 become 1, and smaller than 0 become 0
    
    return iou


def crop_image2(img, bbox, img_size=107, padding=16, flip=False, rotate_limit=0, blur_limit=0):
    """crop image and perform some transformations such as rotation, flip, blur image

    @Args:
        img (ndarray): origin image.
        bbox (ndarray): Size of bounding box.
        img_size (int, optional): Size of cropped image. Defaults to 107.
        padding (int, optional): padding. Defaults to 16.
        flip (bool, optional): Whether to flip the image. Defaults to False.
        rotate_limit (int, optional): Max number of rotation angle. Defaults to 0.
        blur_limit (int, optional): Max number of blured size. Defaults to 0.

    Returns:
        patch (ndarray): Image after cropping and other processing transformation
    """
    
    x, y, w, h = np.array(bbox, dtype='float32')
    # [x, y, w, h] -> [center_x, center_y, w, h]
    cx, cy = x+w/2, y+h/2  # center postion

    if padding > 0:
        w += 2 * padding * w / img_size
        h += 2 * padding * h / img_size

    # List of transformation matrix
    matrix = []

    # Translation matrix to move patch center to origin
    translation_matrix = np.asarray([[1, 0, -cx],
                                     [0, 1, -cy],
                                     [0, 0, 1]], dtype=np.float32)
    matrix.append(translation_matrix)

    # Scaling matrix according to image size
    scaling_matrix = np.asarray([[img_size / w, 0, 0],
                                 [0, img_size / h, 0],
                                 [0, 0, 1]], dtype=np.float32)
    matrix.append(scaling_matrix)

    # Define flip matrix
    if flip and np.random.binomial(1, 0.5):
        flip_matrix = np.eye(3, dtype=np.float32)
        flip_matrix[0, 0] = -1
        matrix.append(flip_matrix)

    ''' Define rotation matrix.
        Transformation matrix for the rotation transformation is as follows: 
        [[cosθ, -sinθ, 0], 
         [sinθ, cosθ,  0], 
         [0,    0,     1]]
        and θ is rotation angle(radians).
    '''
    if rotate_limit and np.random.binomial(1, 0.5):
        # Generate a float number randomly in the range of negative and positive rotate_limit as the rotation angle
        angle = np.random.uniform(-rotate_limit, rotate_limit)        
        alpha = np.cos(np.deg2rad(angle))  # Convert angles from degrees to radians
        beta = np.sin(np.deg2rad(angle))
        rotation_matrix = np.asarray([[alpha, -beta, 0], 
                                      [beta, alpha, 0], 
                                      [0, 0, 1]], dtype=np.float32)
        matrix.append(rotation_matrix)

    # Translation matrix to move patch center from origin
    revert_t_matrix = np.asarray([[1, 0, img_size / 2],
                                  [0, 1, img_size / 2],
                                  [0, 0, 1]], dtype=np.float32)
    matrix.append(revert_t_matrix)

    # Aggregate all transformation matrix
    matrix_t = np.eye(3)  # generate a 3*3 matrix
    for m_ in matrix:
        matrix_t = np.matmul(m_, matrix_t)

    # Warp image, padded value is set to 128
    patch = cv2.warpPerspective(img, matrix_t, (img_size, img_size), borderValue=128)

    if blur_limit and np.random.binomial(1, 0.5):
        blur_size = np.random.choice(np.arange(1, blur_limit + 1, 2))
        patch = cv2.GaussianBlur(patch, (blur_size, blur_size), 0)

    return patch