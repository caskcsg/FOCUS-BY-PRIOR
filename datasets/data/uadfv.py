import os
import cv2
from torch.utils.data import Dataset
import pickle

ANNO_FILE = 'sample/annos/annotation_UADFV_{}.pkl'


class UADFV(Dataset):
    def __init__(self, data_root, transform, input_size=224, mode='test'):
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
            if 'fake' in video:
                label = 1
            else:
                label = 0

            frames = dict[video]
            for item in frames.keys():
                img_name = os.path.join(self.data_root, 'sample/faces_13', video, item + '.png')
                if not os.path.exists(img_name):
                    continue
                all_imgs.append(img_name)
                all_labels.append(label)
        print('Category:UADFV,Images:{}'.format(len(all_imgs)))
        return all_imgs, all_labels