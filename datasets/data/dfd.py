import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import pickle

COMPRESSION = 'c0'
ANNO_FILE = 'sample/annos/annotation_DeepFakeDetection_{}.pkl'
DATASETS = {
    'Original': 'sample/faces_13/original_sequences/actors',
    'DFD': 'sample/faces_13/manipulated_sequences/DeepFakeDetection'
}


class DFD(Dataset):
    def __init__(self, data_root, transform, input_size=224, mode='train'):
        self.data_root = data_root
        self.transform = transform

        self.mode = mode
        self.input_size = input_size
        self.all_imgs, self.all_labels = self._make_dataset()

    def __getitem__(self, index):
        img_pth = self.all_imgs[index]
        label = self.all_labels[index]
        img = cv2.imread(img_pth)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        return self.transform(img), label, img_pth

    def __len__(self):
        return len(self.all_imgs)

    def _make_dataset(self):
        all_imgs = []
        all_labels = []
        with open(os.path.join(self.data_root, ANNO_FILE.format(self.mode)), 'rb') as f:
            dict = pickle.load(f)

        for video in dict.keys():
            splits = video.split('__')
            if len(splits[0]) > 2:
                category = 'DFD'
                label = 1
            else:
                category = 'Original'
                label = 0

            frames = dict[video]
            for item in frames.keys():
                img_name = os.path.join(self.data_root, DATASETS[category], COMPRESSION, video, item + '.png')
                if not os.path.exists(img_name):
                    continue
                all_imgs.append(img_name)
                all_labels.append(label)
        print('Category:{},Images:{}'.format(category, len(all_imgs)))
        return all_imgs, all_labels
