import _init_paths
import os
import argparse
import torch
from  models.ResNoise import ResNoise
from  models.ResCanny import ResCanny
from  models.ResCanny_BN import ResCanny_BN
from  models.ResCannyNoise import ResCannyNoise
from  models.CannyNoiseNorm_ShareTrm import CannyNoiseNorm_ShareTrm
from  models.Cannyadd1_ShareTrm import Cannyadd1_ShareTrm
from  models.NormNoise_ShareTrm import NormNoise_ShareTrm 
from models.resnet34 import ResNet34
from models.fca34 import FcaNet34

from datasets.loader.dataloader import get_data_loader

from utils.evaluation.metric import evaluate, confusion_matrix, get_acc, get_eer
import torch.nn.functional as F

import timm
import numpy as np
from utils.Net import load_model,load_pretrained
#import cudnn
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#cudnn.benchmark = False
# 'vit_base_resnet50_224_in21k'


def ViT():
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = torch.nn.Linear(768, 2)
    # print(model)
    return model


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Test a classifier network')

    # Model hyper-parameters
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--pre_model', type=str, default='resnet34')
    parser.add_argument('--save_dir', dest='save_dir', help='directory to save models', default="../output", type=str)
    parser.add_argument('--norm', type=bool, default=False)

    # Model setting
    parser.add_argument('--img_size', dest='img_size', help='inpput img_size', default=224, type=int)
    parser.add_argument('--pre_checkpoint_pth', dest='pre_checkpoint_pth', help='directory to data',
                        default='/workspace/code/vit_deepfake_detection/models/trained_ckpt/resnet34/model_36000.pth',
                        type=str)
    parser.add_argument('--data_root', dest='data_root', help='directory to data',
                        default='/workspace/data/FF++/face_margin16', type=str)
    parser.add_argument('--model', type=str, default='efb0')
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--out_mask', type=bool, default=False, help='model with mask or not')
    parser.add_argument('--train_dataset', type=str, default='Original', help='dataset')
    parser.add_argument('--test_dataset', type=str, default='NT_Pristine', help='dataset')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'val'])
    parser.add_argument('--test_interval', dest='test_interval', help='number of iterations to test', default=1000,
                        type=int)
    parser.add_argument('--start', dest='checkpoint_start', help='checkpoint to start load model',
                        default=1000, type=int)
    parser.add_argument('--end', dest='checkpoint_end', help='checkpoint to end load model',
                        default=60001,
                        type=int)
    return parser.parse_args()


#def premodel(args):
 #   pre_model = ResNet34(2)
  #  pre_model = load_model(pre_model, args.pre_checkpoint_pth,args.gpu_id)
   # pre_model = pre_model.cuda(args.gpu_id)
    #return pre_model


def main():
    args = parse_args()
    #pre_model = premodel(args)
    model_root = os.path.join(args.save_dir, args.model )
    if args.version is not None:
        model_root = model_root + '_'  + args.version
    if args.out_mask:
        model_root = model_root + '_mask'
    print(model_root)
   
    val_loader = get_data_loader(data_root=args.data_root,
                                 batch_size=128,
                                 num_workers=4,
                                 input_size=args.img_size,
                                 mode=args.mode,
                                 data_source=args.test_dataset)

    results = []
    for i in range(args.checkpoint_start, args.checkpoint_end, args.test_interval):
        checkpoint_pth = os.path.join(model_root, 'ckpt', 'model_{}.pth'.format(i))
        erro_list = []

        # build model
        # if args.model == 'resnet50_mask':
        #     model = ResNet50_Mask(num_class=args.num_class)
        if args.model == 'NormNoise_ShareTrm':
                model = NormNoise_ShareTrm() 
        elif args.model == 'CannyNoiseNorm_ShareTrm':
                model = CannyNoiseNorm_ShareTrm()
        elif args.model == 'Cannyadd1_ShareTrm':
                model = Cannyadd1_ShareTrm()
        elif args.model == 'resnet34':
                model = ResNet34(num_class=args.num_class)
        elif args.model == 'ResCanny':
                model = ResCanny() 
        elif args.model == 'ResCanny_BN':
                model = ResCanny_BN() 
        elif args.model == 'ResNoise':
                model = ResNoise()
        elif args.model == 'ResCannyNoise':
                model = ResCannyNoise()
        elif args.model == 'fca34':
                model = FcaNet34(num_class=args.num_class)
      
        else:
            print("no model")
            # model, *_ = model_selection(modelname='xception', num_out_classes=2)
        model = load_model(model, checkpoint_pth,args.gpu_id)
        #model = load_pretrained(model, checkpoint_pth)
        model = model.cuda(args.gpu_id)
        model.eval()

        all_preds = []
        all_labels = []
        all_pos_scores = []
        with torch.no_grad():
            for imgs, labels, img_pths,cannys in val_loader:
                # for imgs, labels, img_pths in val_loader:
                imgs = imgs.cuda(args.gpu_id)
                labels = labels.cuda(args.gpu_id)
                cannys = cannys.cuda(args.gpu_id)
                if args.model == 'resnet50_mask':
                    outputs, _, _ = model(imgs)
                elif args.model == 'ResVit_Fusion':
                    output1, output2  = model(imgs)  
                    outputs= (output1+output2)/2
                elif args.model == 'resnet34':
                    outputs = model(imgs)
                elif args.model == 'fca34' or args.model == 'WOcanny_noiseShareTrm' or args.model == 'ResNoise' or args.model=='NormNoise_ShareTrm':
                    outputs  = model(imgs)
                elif args.model == 'CannySuper_Sideway' or args.model == 'CannySuper_Backbone' or args.model == 'CNN_CannySuper':
                    outputs, _, _  = model(imgs,cannys)
                else:
                    outputs = model(imgs,cannys)
                _, predicted = torch.max(outputs, 1)
                outputs = F.softmax(outputs, dim=1)
                outputs = outputs[:, -1].view(-1)
                all_pos_scores.extend(outputs.detach().cpu().numpy().tolist())
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

                for idx in range(len(labels)):
                    if labels.cpu()[idx] != predicted.cpu()[idx]:
                        erro_list.append(img_pths[idx] + '\n')
        if args.test_dataset == 'CelebA':
            acc = get_acc(all_labels, all_preds)
            print('Acc:{}'.format(acc))
        else:
            acc = get_acc(all_labels, all_preds)
            bacc, roc_auc = evaluate(all_labels, all_preds, all_pos_scores)
            TN, FP, FN, TP, = confusion_matrix(all_labels, all_preds).ravel()
            real_recall = TN / (TN + FP)
            fake_recall = TP / (TP + FN)
            eer = get_eer(all_labels, all_pos_scores)

            far = FP / (FP + TN)
            frr = FN / (FN + TP)
            hter = (far + frr) / 2
            result = 'model:{},Total images:{},acc:{:.6f},bACC:{:.6f},RR:{:.6f},FR:{:.6f},ROC_AUC:{:.6f},EER:{:.6f},' \
                     'HTER:{:.6f},TN:{},FN:{},TP:{},FP:{}' \
                .format(str(i), len(all_labels), acc, bacc, real_recall, fake_recall, roc_auc, eer, hter, TN, FN, TP, FP)
            print(result)
            results.append(result + '\n')

        if args.train_dataset == args.test_dataset:
            with open(os.path.join(model_root, 'error_face_{}.txt'.format(i)), 'w') as f:
                f.writelines(erro_list)

    with open(os.path.join(model_root, 'results_face_{}_{}.txt'.format(args.train_dataset, args.test_dataset)),
              'w') as f:
        f.writelines(results)


if __name__ == '__main__':
    main()
