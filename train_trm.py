import _init_paths
import os
import sys
import argparse
import logging
import time
import datetime
import traceback
import timm
import torch
import numpy as np
from  models.ResCanny import ResCanny
from  models.ResCanny_BN import ResCanny_BN
from  models.ResCannyNoise import ResCannyNoise
from  models.CannyNoiseNorm_ShareTrm import CannyNoiseNorm_ShareTrm
from  models.Cannyadd1_ShareTrm import Cannyadd1_ShareTrm
from  models.NormNoise_ShareTrm import NormNoise_ShareTrm 
from utils.Net import load_model
from datasets.loader.dataloader import get_data_loader
from torch.utils.tensorboard import SummaryWriter
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings

warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)


# 'vit_base_resnet50_224_in21k'
def ViT():
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = torch.nn.Linear(768, 2)
    print(model)
    return model


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a classifier network')

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='efb0')
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'val'])
    parser.add_argument('--img_size', dest='img_size', help='inpput img_size', default=320, type=int)
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--norm', type=bool, default=False)

    # Path
    parser.add_argument('--data_root', dest='data_root', help='directory to data',
                        default='/workspace/data/FF++/face_margin16', type=str)
    parser.add_argument('--save_dir', dest='save_dir', help='directory to save models', default="../output", type=str)
    parser.add_argument('--log_dir', dest='log_dir', help="directory to log", default='log', type=str)

    # Training settings
    parser.add_argument('--num_workers', dest='num_workers', help='number of worker to load data', default=4, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='batch_size', default=24, type=int)
    parser.add_argument('--max_iter', dest='max_iter', help='max_iter', default=50000, type=int)
    parser.add_argument('--lr', dest='lr', help='starting learning rate', type=float, default=0.0001)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', help='learning rate decay ratio', type=float,
                        default=0.9)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step', help='learning rate decay step', type=int,
                        default=2000)
    parser.add_argument('--o', dest='optimizer', help='Training optimizer.', default='Adam')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=0.0001, type=float, help='Weight decay for SGD')
    parser.add_argument('--resume', default=False,
                        type=bool, help='resume training from latest checkpoint')

    # Step size
    parser.add_argument('--disp_interval', dest='disp_interval', help='number of iterations to display', default=100,
                        type=int)
    parser.add_argument('--save_interval', dest='save_interval', help='number of iterations to save', default=1000,
                        type=int)

    parser.add_argument('--dataset', type=str, default='Pristine+DF_F2F_FS', help='dataset')
    #

    # Misc
    parser.add_argument('--seed', type=int, default=2018, help='random seed (default: 1)')
    parser.add_argument('--use_tensorboard', dest='use_tensorboard',
                        help='whether use tensorflow tensorboard',
                        default=True, type=bool)
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    return parser.parse_args()


class Solver(object):
    def __init__(self, config):
        super(Solver, self).__init__()
        self.config = config
        self.start_step = 0
        # self.num_real_domain = num_real_domain
        # self.num_fake_domain = num_fake_domain
        self.build_model()

        # Build tensorboard if use
        if self.config.use_tensorboard and self.config.mode == 'train':
            self.build_tensorboard()

    def build_tensorboard(self):
        print(self.config.log_dir)
        self.tb_logger = SummaryWriter(self.config.log_dir)

    def build_model(self):
    #CannyNoiseNorm_ShareTrm_320_c23.pth  NormNoise_ShareTrm_320_c23.pth Cannyadd1_ShareTrm_320_c23.pth

        if self.config.model == 'CannyNoiseNorm_ShareTrm':
            self.model = CannyNoiseNorm_ShareTrm()
            pre_checkpoint_pth ="/workspace/code/c40_cannyvit/pretrain_ckpt/CannyNoiseNorm_ShareTrm_320_c23.pth"
            self.model = load_model(self.model, pre_checkpoint_pth, self.config.gpu_id)
        elif self.config.model == 'Cannyadd1_ShareTrm':
            self.model = Cannyadd1_ShareTrm()
            pre_checkpoint_pth ="/workspace/code/c40_cannyvit/pretrain_ckpt/Cannyadd1_ShareTrm_320_c23.pth"
            self.model = load_model(self.model, pre_checkpoint_pth, self.config.gpu_id)
        elif self.config.model == 'ResCanny':
            self.model = ResCanny() 
        elif self.config.model == 'ResCanny_BN':
            self.model = ResCanny_BN() 
            #pre_checkpoint_pth ="/workspace/code/c40_cannyvit/pretrain_ckpt/ResCanny_320_c23_bs32.pth"
            #self.model = load_model(self.model, pre_checkpoint_pth, self.config.gpu_id) 
        elif self.config.model == 'ResCannyNoise':
            self.model = ResCannyNoise() 
            pre_checkpoint_pth ="/workspace/code/c40_cannyvit/pretrain_ckpt/ResCannyNoise_320_c23_bs32.pth"
            self.model = load_model(self.model, pre_checkpoint_pth, self.config.gpu_id) 
              
       
        else:
            print("no model !!!!!!!!")
        # print(self.model)
        self.ce = torch.nn.CrossEntropyLoss()
        # self.mask_criterion = torch.nn.BCELoss()
        transformer_params_id = list(map(id, self.model.transformer.parameters())) 
        transformer_params = filter(lambda p: id(p) in transformer_params_id and p.requires_grad,
                                    self.model.parameters())
        base_params = filter(lambda p: id(p) not in transformer_params_id and p.requires_grad, self.model.parameters())

        if self.config.mode == 'train':
            optimizer_dict = [

                {"params": transformer_params, "lr": self.config.lr*10},
                {"params": base_params, "lr": self.config.lr}
            ]

            if self.config.optimizer == 'Adam':
                self.optimizer = torch.optim.Adam(optimizer_dict,
                                                  lr=self.config.lr, betas=(self.config.beta1, self.config.beta2),
                                                  weight_decay=self.config.weight_decay)
            elif self.config.optimizer == "SGD":
                self.optimizer = torch.optim.SGD(optimizer_dict,
                                                 lr=self.config.lr, momentum=self.config.momentum,
                                                 weight_decay=self.config.weight_decay)
            # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=0,
            # last_epoch=-1)

            # anneal
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                step_size=self.config.lr_decay_step,
                                                                gamma=self.config.lr_decay_gamma)

        if self.config.resume:
            resume_pth = os.path.join(self.config.save_dir, 'ckpt/checkpoint.pth')
            checkpoint = torch.load(resume_pth, map_location=lambda storage, loc: storage.cuda(self.config.gpu_id))
            self.start_step = 28800  # checkpoint['step']
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self.config.gpu_id)
            print("=> loaded checkpoint '{}' (step {})"
                  .format(self.config.resume, self.start_step))
            del checkpoint
            torch.cuda.empty_cache()

        if torch.cuda.is_available():
            self.model = self.model.cuda(self.config.gpu_id)

    def train(self, dataloader_reals, dataloader_fakes):
        # setting to train mode
        self.model.train()
        start_time = time.time()
        iter_per_epoch_reals = []
        for dataloader in dataloader_reals:
            iter_per_epoch_reals.append(len(dataloader))
        iter_per_epoch_fakes = []
        for dataloader in dataloader_fakes:
            iter_per_epoch_fakes.append(len(dataloader))

        data_iter_reals = {}
        for i in range(len(dataloader_reals)):
            data_iter_reals[str(i)] = None
        data_iter_fakes = {}
        for i in range(len(dataloader_fakes)):
            data_iter_fakes[str(i)] = None

        try:
            for step in range(self.start_step, self.config.max_iter):

                for idx, iter_per_epoch in enumerate(iter_per_epoch_reals):
                    # print (step,"step")
                    # print("iter_per_epoch",iter_per_epoch)
                    if step % iter_per_epoch == 0:
                        # print("index",idx)
                        data_iter_reals[str(idx)] = iter(dataloader_reals[idx])

                for idx, iter_per_epoch in enumerate(iter_per_epoch_fakes):
                    # print (step,"step")
                    # print("iter_per_epoch",iter_per_epoch)
                    if step % iter_per_epoch == 0:
                        data_iter_fakes[str(idx)] = iter(dataloader_fakes[idx])

                all_imgs = []
                all_cannys = []
                all_labels = []
                shape_reals = []
                for key in data_iter_reals.keys():
                    img, label, canny = data_iter_reals[key].next()
                    #print (img)
                    img = img.cuda(self.config.gpu_id)
                    canny = canny.cuda(self.config.gpu_id)
                    label = label.cuda(self.config.gpu_id)
                    all_imgs.append(img)
                    all_cannys.append(canny)
                    all_labels.append(label)
                    shape_reals.append(img.shape[0])

                shape_fakes = []
                for key in data_iter_fakes.keys():
                    img, label, canny = data_iter_fakes[key].next()
                    img = img.cuda(self.config.gpu_id)
                    canny = canny.cuda(self.config.gpu_id)
                    label = label.cuda(self.config.gpu_id)
                    all_imgs.append(img)
                    all_cannys.append(canny)
                    all_labels.append(label)
                    shape_fakes.append(img.shape[0])

                imgs = torch.cat(all_imgs, dim=0)
                cannys = torch.cat(all_cannys, dim=0)
                labels = torch.cat(all_labels, dim=0)
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                # pred_cls, features = self.model(imgs)
                pred_cls = self.model(imgs, cannys)
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                # classification loss
                cls_loss = self.ce(pred_cls, labels.long())
                # print("cls_loss!!",cls_loss)
                all_losses = [cls_loss]

                total_loss = 0
                for loss in all_losses:
                    total_loss += loss

                # backward
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
               
                
                        # display results
                if (step + 1) % self.config.disp_interval == 0:
                    loss = {}

                    loss['S/cls_loss'] = cls_loss.item()
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    log = 'time cost: {} iter: {} / {}' \
                        .format(elapsed, step + 1, self.config.max_iter)
                    for tag, value in loss.items():
                        log += ", {}: {:.5f}".format(tag, value)
                    log += ", {}: {:.6f}".format(
                        'lr', self.optimizer.param_groups[0]["lr"])
                    print(log)

                    # Tensorboard logger
                    if self.config.use_tensorboard:
                        for tag, value in loss.items():
                            self.tb_logger.add_scalar(tag, value, step + 1)

                if (step + 1) % self.config.save_interval == 0:
                    state = {
                        'step': step + 1,
                        'optimizer': self.optimizer.state_dict(),
                        'model': self.model.state_dict(),
                        'lr_scheduler': self.lr_scheduler.state_dict()
                    }
                    ckpt_dir = os.path.join(self.config.save_dir, 'ckpt')
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    save_name = os.path.join(ckpt_dir, 'model_{}.pth'.format(step + 1))
                    torch.save(state, save_name)

        except(RuntimeError, KeyboardInterrupt):
            stack_trace = traceback.format_exc()
            print(stack_trace)
            state = {
                'step': step + 1,
                'optimizer': self.optimizer.state_dict(),
                'model': self.model.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict()
            }
            ckpt_dir = os.path.join(self.config.save_dir, 'ckpt')
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            save_name = os.path.join(ckpt_dir, 'checkpoint.pth')
            torch.save(state, save_name)
        finally:
            if self.config.use_tensorboard:
                self.tb_logger.close()


def main():
    args = parse_args()
    logger.info('\t Called with args:')
    logger.info(args)

    if not torch.cuda.is_available():
        sys.exit('Need a CUDA device to run the code.')

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    model_tag = args.model + '_' + str(args.img_size)
    if args.version is not None:
        model_tag = model_tag + '_' + args.version
    save_dir = os.path.join(args.save_dir, model_tag)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger.info('\t output will be saved to {}'.format(save_dir))

    log_dir = os.path.join(save_dir, args.log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger.info('\t logs will be saved to {}'.format(log_dir))

    args.save_dir = save_dir
    args.log_dir = log_dir

    source_reals = args.dataset.split('+')[0].split('_')  # ['Pristine']
    source_fakes = args.dataset.split('+')[1].split('_')  # ['DF', 'F2F', 'FS']
    # ??? why only 3 fake dataset means source dataset
    '''scale the number of real data is expanded to equal the number of false data sets'''
    real_scale = int(len(source_fakes) / len(source_reals))
    print(real_scale)
    dataloader_reals = []
    for source in source_reals:
        dataloader = get_data_loader(data_root=args.data_root,
                                     batch_size=args.batch_size * real_scale,
                                     num_workers=args.num_workers,
                                     input_size=args.img_size,
                                     mode=args.mode,
                                     data_source=source)
        dataloader_reals.append(dataloader)

    dataloader_fakes = []
    for source in source_fakes:
        dataloader = get_data_loader(data_root=args.data_root,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     input_size=args.img_size,
                                     mode=args.mode,
                                     data_source=source)
        dataloader_fakes.append(dataloader)

    trainer = Solver(args)
    trainer.train(dataloader_reals, dataloader_fakes)


if __name__ == '__main__':
    main()
