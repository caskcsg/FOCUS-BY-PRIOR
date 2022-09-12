import os
import cv2
from torch.utils.data import Dataset
import pickle
from simulate.fs_simulator import SimulatorHomdogy

DATA_ROOT = '/workspace/data/CelebA'
ANNO_FILE = 'annotation_faces_13.pkl'
FACE_ROOT = 'faces_13'


class CelebA(Dataset):
    def __init__(self, data_root, transform, input_size=320, mode='train', out_mask=False, nums=None, sim=False):
        self.data_root = DATA_ROOT
        self.transform = transform
        self.mode = mode
        self.input_size = input_size
        self.out_mask = out_mask
        self.nums = nums
        self.sim = sim
        self.all_imgs, self.all_landms = self._make_dataset()
        self.simulator = SimulatorHomdogy()

    def __getitem__(self, index):
        img_pth = self.all_imgs[index]
        face = cv2.imread(img_pth)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        if self.sim:
            face_fake, face_mask = self.simulator(face, self.all_landms[index])
            face_fake = cv2.resize(face_fake, (self.input_size, self.input_size))
            return self.transform(face_fake), 1
        else:
            face_real = cv2.resize(face, (self.input_size, self.input_size))
            return self.transform(face_real), 0

    def __len__(self):
        return len(self.all_imgs)

    def _make_dataset(self):
        all_imgs = []
        all_landms = []

        with open(os.path.join(self.data_root, ANNO_FILE), 'rb') as f:
            dict_face = pickle.load(f)

        for img_name in dict_face.keys():
            face_pth = os.path.join(self.data_root, FACE_ROOT, img_name)
            points = dict_face[img_name]['landms']

            all_imgs.append(face_pth)
            all_landms.append(points)
            if self.nums is not None:
                if len(all_imgs) == self.nums:
                    break
        print('Faces:{}'.format(len(all_imgs)))
        return all_imgs, all_landms
