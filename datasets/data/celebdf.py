import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import pickle


class CelebDF(Dataset):
    def __init__(self, data_root, transform,transform_canny, input_size=320, mode='train'):
        self.data_root = data_root
        self.transform = transform
        self.transform_canny = transform_canny

        self.mode = mode
        self.input_size = input_size
        self.all_imgs, self.all_labels = self._make_dataset()

    def __getitem__(self, index):
        img_pth = self.all_imgs[index]
        label = self.all_labels[index]
        img = cv2.imread(img_pth)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.mode == 'train':
            face = cv2.resize(img, (self.input_size, self.input_size))
            guass = cv2.GaussianBlur(face, (9, 9), 0)
            canny = cv2.Canny(guass, 50, 100)
            return self.transform(face), label, self.transform_canny(canny)
        else:
            face = cv2.resize(img, (self.input_size, self.input_size))
            #guass = cv2.GaussianBlur(face, (9, 9), 0)
            #canny = cv2.Canny(guass, 50, 100)
            guass = cv2.GaussianBlur(face, (7, 7), 0)
            canny = cv2.Canny(guass, 100, 0)
            return self.transform(face), label, img_pth, self.transform_canny(canny)
        
        
        #img = cv2.resize(img, (self.input_size, self.input_size))
        #return self.transform(img), label, img_pth

    def __len__(self):
        return len(self.all_imgs)

    def _make_dataset(self):
        all_imgs = []
        all_labels = []
        with open(os.path.join(self.data_root, 'sample/annos/annotation_CelebDF_{}.pkl').format(self.mode), 'rb') as f:
            dict = pickle.load(f)

        for video in dict.keys():
            frame_dir = os.path.join(self.data_root, 'sample/faces_13', video.split('.')[0])
            frames = dict[video]
            keys = list(frames.keys())
            for item in keys:
                img_pth = os.path.join(frame_dir, item + '.png')
                all_imgs.append(img_pth)
                if 'synthesis' in video:
                    label = 1
                else:
                    label = 0
                all_labels.append(label)
        print(len(all_imgs))
        return all_imgs, all_labels


# data_root = '/workspace/data/Celeb-DFv2'
# dataset = CelebDF(data_root, None, 224, 'train')

