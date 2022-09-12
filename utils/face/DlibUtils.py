import cv2
import numpy as np
# import dlib
from utils.face.Umeyama import umeyama

predefined_color = [128, 64, 160, 244, 35, 220, 80, 200]

mean_face_x = np.array([
    0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
    0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
    0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
    0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
    0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
    0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
    0.553364, 0.490127, 0.42689])

mean_face_y = np.array([
    0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
    0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
    0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
    0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
    0.784792, 0.824182, 0.831803, 0.824182])

landmarks_2D = np.stack([mean_face_x, mean_face_y], axis=1)

FOREHEAD_POINTS = list(range(68, 81))
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

EYE_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
              RIGHT_BROW_POINTS)
NOSE_MOUTH_POINTS = NOSE_POINTS + MOUTH_POINTS

#
# def get_face_detector():
#     return dlib.get_frontal_face_detector()
#
#
# def get_face_predictor(point_num=68):
#     if point_num == 68:
#         model_path = './shape_predictor_68_face_landmarks.dat'
#     else:
#         model_path = './shape_predictor_81_face_landmarks.dat'
#     return dlib.shape_predictor(model_path)


def shape_to_np(shape, point_num=68):
    coords = np.zeros((point_num, 2), dtype=int)
    for i in range(0, point_num):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def get_trans_matrix(points):
    """
    compute face area align matrix
    :param points: landmarks points
    :return:
    """
    return umeyama(points[FACE_POINTS], landmarks_2D, True)[0:2]


def get_rectangle(points):
    """
    :param points: [N,2]
    :return:
    """
    x, y, w, h = cv2.boundingRect(points)
    return np.array([(x, y), (x + w, y), (x, y + h), (x + w, y + h)], dtype=np.int32)


def get_min_area_rectangle(points):
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    return box.astype(np.int32)


def get_mask(img_size, points, value=1):
    """
    Get face mask by landmarks
    :param img_size: [W,H]
    :param points: [N,2]
    :return:
    """
    points = cv2.convexHull(points)
    mask = np.zeros(img_size, dtype=np.float32)
    mask = cv2.fillConvexPoly(mask, points, value)
    return mask


def draw_face_bbox(img, bbox, color=(0, 255, 0)):
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    return img


def draw_convex_hull(img, points, color=1):
    """
    Draw convex hull on original image.
    """
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(img, points, color=color)
    return img


def draw_landmarks(img, shape):
    for (x, y) in shape:
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    return img


def line_landmarks(img, points):
    lm_img = np.ones(img.shape, dtype=np.uint8) * 255
    # chin
    for i in range(17 - 1):
        cv2.line(lm_img, (points[i, 0], points[i, 1]), (points[i + 1, 0], points[i + 1, 1]),
                 [predefined_color[0], 250, 10], 2)
    # left eyebrow
    for i in range(17, 22 - 1):
        cv2.line(lm_img, (points[i, 0], points[i, 1]), (points[i + 1, 0], points[i + 1, 1]),
                 [predefined_color[1], 10, 250], 2)
    # right eyebrow
    for i in range(22, 27 - 1):
        cv2.line(lm_img, (points[i, 0], points[i, 1]), (points[i + 1, 0], points[i + 1, 1]),
                 [predefined_color[2], 150, 150], 2)
    # nose
    for i in range(27, 31 - 1):
        cv2.line(lm_img, (points[i, 0], points[i, 1]), (points[i + 1, 0], points[i + 1, 1]),
                 [predefined_color[3], 0, 0], 2)
    for i in range(31, 36 - 1):
        cv2.line(lm_img, (points[i, 0], points[i, 1]), (points[i + 1, 0], points[i + 1, 1]),
                 [predefined_color[4], 50, 0], 2)
    # left eye
    for i in range(36, 42 - 1):
        cv2.line(lm_img, (points[i, 0], points[i, 1]), (points[i + 1, 0], points[i + 1, 1]),
                 [predefined_color[5], 0, 50], 2)
    cv2.line(lm_img, (points[36, 0], points[36, 1]), (points[41, 0], points[41, 1]),
             [predefined_color[5], 0, 50], 2)
    # right eye
    for i in range(42, 48 - 1):
        cv2.line(lm_img, (points[i, 0], points[i, 1]), (points[i + 1, 0], points[i + 1, 1]),
                 [predefined_color[6], 180, 30], 2)
    cv2.line(lm_img, (points[42, 0], points[42, 1]), (points[47, 0], points[47, 1]),
             [predefined_color[6], 180, 30], 2)
    # outer and inner lip
    for i in range(48, 60 - 1):
        cv2.line(lm_img, (points[i, 0], points[i, 1]), (points[i + 1, 0], points[i + 1, 1]),
                 [predefined_color[7], 20, 180], 2)
    cv2.line(lm_img, (points[48, 0], points[48, 1]), (points[59, 0], points[59, 1]),
             [predefined_color[7], 20, 180], 2)

    return lm_img
