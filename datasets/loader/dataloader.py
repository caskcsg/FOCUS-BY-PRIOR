import torch
from torch.utils.data import DataLoader,ConcatDataset
from datasets.data.celebdf import CelebDF
import torchvision.transforms as transforms
from datasets.data.faceforensics import FaceForensics_faces13,FaceForensics_faces13_nocanny#FaceForensics,
# from datasets.data.uadfv import UADFV
# from datasets.data.dfd import DFD
# from datasets.data.celeba import CelebA
import numpy as np


# ??? what mask for
def mask_collate(batch):
    real_imgs = torch.stack([item[0] for item in batch], 0)
    fake_imgs = torch.stack([item[3] for item in batch], 0)
    all_imgs = torch.cat((real_imgs, fake_imgs), 0)

    real_masks = torch.stack([item[1] for item in batch], 0)
    fake_masks = torch.stack([item[4] for item in batch], 0)
    all_masks = torch.cat((real_masks, fake_masks), 0)

    real_targets = torch.from_numpy(np.array([item[2] for item in batch]))
    fake_targets = torch.from_numpy(np.array([item[5] for item in batch]))
    targets = torch.cat((real_targets, fake_targets), 0)
    return all_imgs, all_masks, targets.float()

# data_root=/data6/caiyu/data/FF++/face_margin16'
def get_data_loader(data_root, batch_size, num_workers, input_size=320, mode='train',
                    data_source='DF', tsne=False):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_canny = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_canny = transforms.Compose([
            transforms.ToTensor()
        ])

    if mode == 'train':
        if data_source in ['DF', 'F2F', 'FS', 'NT', 'Pristine', 'P-Sim']:
            # if input_size == 224:
            #     data_root = '/workspace/data/FF++/face_margin16'
            #     dataset = FaceForensics(data_root, transform, input_size, mode, data_source)
            # else:
                data_root = '/workspace/data/FaceForensics++/'
                dataset = FaceForensics_faces13(data_root, transform, transform_canny,input_size, mode, data_source)
                #dataset = FaceForensics_faces13_nocanny(data_root, transform, input_size, mode, data_source)

        # if data_source in ['CelebA', 'C-Sim']:
        #     if data_source == 'C-Sim':
        #         dataset = CelebA(data_root, transform, input_size, mode, sim=True)
        #     else:
        #         dataset = CelebA(data_root, transform, input_size, mode)
        shuffle = True
        # if out_mask:
        #     data_loader = DataLoader(dataset=dataset,
        #                              batch_size=batch_size,
        #                              num_workers=num_workers,
        #                              collate_fn=mask_collate,
        #                              shuffle=shuffle)
        # else:
        #     data_loader = DataLoader(dataset=dataset,
        #                              batch_size=batch_size,
        #                              num_workers=num_workers,
        #                              shuffle=shuffle)

        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=shuffle)
    # test_mode
    else:
        if data_source in ['DF_F2F_FS_Pristine', 'DF_F2F_NT_Pristine', 'DF_FS_NT_Pristine', 'F2F_FS_NT_Pristine',
                           'DF_Pristine', 'F2F_Pristine', 'FS_Pristine', 'NT_Pristine','NT',
                           'DF_F2F_FS_NT_Pristine']:
            all_datasets = []
            for category in data_source.split('_'):
                # #print("kind", category)
                # if input_size == 224:
                #     data_root = '/workspace/data/FF++/face_margin16'
                #     all_datasets.append(FaceForensics(data_root, transform, input_size, mode, category, tsne=tsne))
                # else:
                    data_root = '/workspace/data/FaceForensics++/'
                    all_datasets.append(FaceForensics_faces13(data_root, transform, transform_canny,input_size, mode, category, tsne=tsne))
            merge_dataset = ConcatDataset(all_datasets)
        

        elif data_source == 'CelebDF':
            merge_dataset = CelebDF(data_root, transform, transform_canny,input_size, mode)

        data_loader = DataLoader(dataset=merge_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=False)
    return data_loader
