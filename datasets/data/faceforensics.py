import os

import cv2
from torch.utils.data import Dataset
import pickle

COMPRESSION = 'c40'
FACE_ROOT = 'faces_13'

# for faces_margin16
DATASETS_fix = {
    'DF': 'DF',
    'F2F': 'F2F',
    'FS': 'FS',
    'NT': 'NT',
    'Pristine': 'youtube'
}
#
ANNO_FILES_fix = {
    'DF': '../annos/annotation_Deepfakes_{}.pkl',
    'F2F': '../annos/annotation_Face2Face_{}.pkl',
    'FS': '../annos/annotation_FaceSwap_{}.pkl',
    'NT': '../annos/annotation_NeuralTextures_{}.pkl',
    'Pristine': '../annos/annotation_youtube_{}.pkl'
}

DATASETS = {
    'DF': 'sample/{}/manipulated_sequences/Deepfakes'.format(FACE_ROOT),
    'F2F': 'sample/{}/manipulated_sequences/Face2Face'.format(FACE_ROOT),
    'FS': 'sample/{}/manipulated_sequences/FaceSwap'.format(FACE_ROOT),
    'NT': 'sample/{}/manipulated_sequences/NeuralTextures'.format(FACE_ROOT),
    'Pristine': 'sample/{}/original_sequences/youtube'.format(FACE_ROOT)
}

ANNO_FILES = {
    'DF': 'sample/annos/annotation_Deepfakes_{}.pkl',
    'F2F': 'sample/annos/annotation_Face2Face_{}.pkl',
    'FS': 'sample/annos/annotation_FaceSwap_{}.pkl',
    'NT': 'sample/annos/annotation_NeuralTextures_{}.pkl',
    'Pristine': 'sample/annos/annotation_youtube_{}.pkl'
}

"""
class FaceForensics(Dataset):
    def __init__(self, data_root, transform, transform_canny, input_size=224, mode='train', dataset='DF', tsne=False):
        self.data_root = data_root
        self.transform = transform
        self.transform_canny = transform_canny
        self.mode = mode
        self.input_size = input_size
        self.dataset = dataset
        self.tsne = tsne
        self.all_imgs, self.all_labels = self._make_dataset()

    def __getitem__(self, index):
        img_pth = self.all_imgs[index]
        label = self.all_labels[index]
        face = cv2.imread(img_pth)
        
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        if self.mode == 'train':
            face = cv2.resize(face, (self.input_size, self.input_size))
            guass = cv2.GaussianBlur(face, (9, 9), 0)
            canny = cv2.Canny(guass, 50, 100)
            return self.transform(face), label, self.transform_canny(canny)
        else:
            face = cv2.resize(face, (self.input_size, self.input_size))
            guass = cv2.GaussianBlur(face, (9, 9), 0)
            canny = cv2.Canny(guass, 50, 100)
            return self.transform(face), label, img_pth, self.transform_canny(canny)

    def __len__(self):
        return len(self.all_imgs)

    def _make_dataset(self):
        all_imgs = []
        all_labels = []
        # all_landms = []

        count = 0

        with open(os.path.join(self.data_root, ANNO_FILES_fix[self.dataset].format(self.mode)), 'rb') as f:
            dict = pickle.load(f)
        # videos_path =  os.path.join(self.data_root, COMPRESSION, DATASETS[self.dataset])
        # video_files = os.listdir(videos_path)

        for video in dict.keys():
            # print ('video',video)

            frames_path = os.path.join(self.data_root, COMPRESSION, DATASETS_fix[self.dataset], video)
            images_path = os.listdir(frames_path)
            for index, image_name in enumerate(images_path):

                img_name = os.path.join(frames_path, image_name)  # m + '.png')
                # print (img_name)
                if not os.path.exists(img_name):
                    continue
                all_imgs.append(img_name)

                if self.dataset == 'Pristine':
                    label = 0
                else:
                    if not self.tsne:
                        label = 1
                    else:
                        if self.dataset == 'DF':
                            label = 1
                        elif self.dataset == 'F2F':
                            label = 2
                        elif self.dataset == 'FS':
                            label = 3
                        elif self.dataset == 'NT':
                            label = 4
                all_labels.append(label)
                count += 1
        print("!!!!!!!!!!!!!!!!!!!!!!!!!all_imgs", len(all_imgs), "all_labels", len(all_labels))
        print('Category:{},Images:{}'.format(self.dataset, count))

        return all_imgs, all_labels
"""

class FaceForensics_faces13_nocanny(Dataset):
    def __init__(self, data_root, transform,  input_size=320, mode='train', dataset='DF', tsne=False):
        self.data_root = data_root
        self.transform = transform
        
        self.mode = mode
        self.input_size = input_size
        self.dataset = dataset
        self.tsne = tsne
        self.all_imgs, self.all_labels = self._make_dataset()

    def __getitem__(self, index):
        img_pth = self.all_imgs[index]
        label = self.all_labels[index]
        face = cv2.imread(img_pth)
        
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        if self.mode == 'train':
            face = cv2.resize(face, (self.input_size, self.input_size))
            
            return self.transform(face), label
        else:
            face = cv2.resize(face, (self.input_size, self.input_size))
            
            return self.transform(face), label, img_pth

    def __len__(self):
        return len(self.all_imgs)

    def _make_dataset(self):
        all_imgs = []
        all_labels = []
        # all_landms = []

        count = 0

        with open(os.path.join(self.data_root, ANNO_FILES[self.dataset].format(self.mode)), 'rb') as f:
            dict = pickle.load(f)
        print("---------------", self.dataset, len(dict.keys()))
        for video in dict.keys():
            # print (video)
            if (dict[video]) == None:
                continue
            # rate = len(dict[video].keys()) // 10
            # print ("rate",rate)
            # length=0
            # index=0
            for item in dict[video].keys():

                # if (index%rate==0):
                img_name = os.path.join(self.data_root, DATASETS[self.dataset], COMPRESSION, video, item + '.png')
                if not os.path.exists(img_name):
                    continue
                # all_imgs.append(img_name)
                all_imgs.append(img_name)
                # print("-------imagpath",img_name)
                # length += 1
                if self.dataset == 'Pristine':
                    label = 0
                else:
                    if not self.tsne:
                        label = 1
                    else:
                        if self.dataset == 'DF':
                            label = 1
                        elif self.dataset == 'F2F':
                            label = 2
                        elif self.dataset == 'FS':
                            label = 3
                        elif self.dataset == 'NT':
                            label = 4
                all_labels.append(label)
                count += 1

            #
            # index +=1
            # if length == 10:
            #      break
        print("!!!!!!!!!!!!!!!!!!!!!!!!!all_imgs", len(all_imgs), "all_labels", len(all_labels))

        print('Category:{},Images:{}'.format(self.dataset, count))

        return all_imgs, all_labels


class FaceForensics_faces13(Dataset):
    def __init__(self, data_root, transform, transform_canny, input_size=320, mode='train', dataset='DF', tsne=False):
        self.data_root = data_root
        self.transform = transform
        self.transform_canny = transform_canny
        self.mode = mode
        self.input_size = input_size
        self.dataset = dataset
        self.tsne = tsne
        self.all_imgs, self.all_labels = self._make_dataset()

    def __getitem__(self, index):
        img_pth = self.all_imgs[index]
        label = self.all_labels[index]
        face = cv2.imread(img_pth)
        
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        if self.mode == 'train':
            face = cv2.resize(face, (self.input_size, self.input_size))
            guass = cv2.GaussianBlur(face, (7, 7), 0)
            canny = cv2.Canny(guass, 50, 50)
            
            return self.transform(face), label, self.transform_canny(canny)
        else:
            face = cv2.resize(face, (self.input_size, self.input_size))
            guass = cv2.GaussianBlur(face, (7, 7), 0)
            canny = cv2.Canny(guass, 50, 50)
            #guass = cv2.GaussianBlur(face, (7, 7), 0)
            #canny = cv2.Canny(guass, 100, 0)
            return self.transform(face), label, img_pth, self.transform_canny(canny)

    def __len__(self):
        return len(self.all_imgs)

    def _make_dataset(self):
        all_imgs = []
        all_labels = []
        # all_landms = []

        count = 0

        with open(os.path.join(self.data_root, ANNO_FILES[self.dataset].format(self.mode)), 'rb') as f:
            dict = pickle.load(f)
        print("---------------", self.dataset, len(dict.keys()))
        for video in dict.keys():
            # print (video)
            if (dict[video]) == None:
                continue
            # rate = len(dict[video].keys()) // 10
            # print ("rate",rate)
            # length=0
            # index=0
            for item in dict[video].keys():

                # if (index%rate==0):
                img_name = os.path.join(self.data_root, DATASETS[self.dataset], COMPRESSION, video, item + '.png')
                if not os.path.exists(img_name):
                    continue
                # all_imgs.append(img_name)
                all_imgs.append(img_name)
                # print("-------imagpath",img_name)
                # length += 1
                if self.dataset == 'Pristine':
                    label = 0
                else:
                    if not self.tsne:
                        label = 1
                    else:
                        if self.dataset == 'DF':
                            label = 1
                        elif self.dataset == 'F2F':
                            label = 2
                        elif self.dataset == 'FS':
                            label = 3
                        elif self.dataset == 'NT':
                            label = 4
                all_labels.append(label)
                count += 1

            #
            # index +=1
            # if length == 10:
            #      break
        print("!!!!!!!!!!!!!!!!!!!!!!!!!all_imgs", len(all_imgs), "all_labels", len(all_labels))

        print('Category:{},Images:{}'.format(self.dataset, count))

        return all_imgs, all_labels
