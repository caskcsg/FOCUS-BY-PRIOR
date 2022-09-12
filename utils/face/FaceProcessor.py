import os
import cv2
import pickle
from tqdm import tqdm
from utils.face.DlibUtils import get_face_predictor, get_face_detector, shape_to_np, get_trans_matrix


class FaceProcessor(object):
    def __init__(self, images_path, cache_path):
        self.detector = get_face_detector()
        self.predictor = get_face_predictor(68)
        self.cache_path = cache_path
        face_caches = self.load_cache()
        if face_caches is None:
            face_caches = {}
            count = 0
            for img_pth in tqdm(images_path):
                img = cv2.imread(img_pth)

                splits = img_pth.split('/')
                img_key = os.path.join(splits[-5], splits[-4], splits[-3], splits[-2], splits[-1])

                faces = self.detector(img, 0)
                if len(faces) == 0:
                    faces = [None, None]
                    count += 1
                else:
                    points = shape_to_np(self.predictor(img, faces[0]))
                    trans_matrix = get_trans_matrix(points)
                    faces = [trans_matrix, points]
                face_caches[img_key] = faces
            print('Total images :{}, Dont detect faces images:{}'.format(len(images_path), count))
            self.save_cache(face_caches)
        self.face_caches = face_caches

    def get_faces(self, img_pth):
        splits = img_pth.split('/')
        img_key = os.path.join(splits[-5], splits[-4], splits[-3], splits[-2], splits[-1])
        faces = self.face_caches[img_key]
        return faces

    def load_cache(self):
        face_caches = None
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                face_caches = pickle.load(f)
        return face_caches

    def save_cache(self, face_caches):
        # Save face and matrix to cache
        with open(self.cache_path, 'wb') as f:
            pickle.dump(face_caches, f)
