#!/bin/bash



#python ../train_trm.py --model=Cannyadd1_ShareTrm --img_size=320 --batch_size=24 --version=c40_bs24_lr_1e10 \
#--dataset=Pristine+DF_NT_F2F_FS --gpu_id=0 \


#python ../train_wo_canny.py --model=NormNoise_ShareTrm --img_size=320 --batch_size=24 --version=c40_bs24_lr_1e10  \
#--dataset=Pristine+DF_NT_F2F_FS --gpu_id=1 \

#python ../train_trm.py --model=CannyNoiseNorm_ShareTrm --img_size=320 --batch_size=24 --version=c40_bs24_lr_1e10 \
#--dataset=Pristine+DF_NT_F2F_FS --gpu_id=1 \

#python ../train_wo_canny.py --model=NormNoise_ShareTrm --img_size=320 --batch_size=24 --version=wopretrain_c40_bs24_lr_1e10  \
#--dataset=Pristine+DF_NT_F2F_FS --gpu_id=1 \


#python ../train_ff.py --model=resnet34 --img_size=320 --batch_size=16 --version=c40_bs16_lr_1e10  \
#--dataset=Pristine+DF_NT_F2F_FS --gpu_id=1 \


#python ../train_ff.py --model=fca34 --img_size=320 --batch_size=16 --version=c40_bs16_lr_1e10  \
#--dataset=Pristine+DF_NT_F2F_FS --gpu_id=1 \

#python ../train_wo_canny.py --model=ResNoise --img_size=320 --batch_size=16 --version=c40_bs16_lr_1e10  \
#--dataset=Pristine+DF_NT_F2F_FS --gpu_id=0 \

#python ../train_trm.py --model=ResCanny --img_size=320 --batch_size=16 --version=c40_bs16_lr_1e10  \
#--dataset=Pristine+DF_NT_F2F_FS --gpu_id=0 \

#python ../train_trm.py --model=ResCannyNoise --img_size=320 --batch_size=16 --version=c40_bs16_lr_1e10  \
#--dataset=Pristine+DF_NT_F2F_FS --gpu_id=0 \

#python ../train_trm.py --model=ResCanny --img_size=320 --batch_size=16 --version=c40_bs16_lr_1e10  \
#--dataset=Pristine+DF_NT_F2F_FS --gpu_id=0 \

python ../train_trm.py --model=ResCanny_BN --img_size=320 --batch_size=16 --version=c40_bs16_lr_1e10  \
--dataset=Pristine+DF_NT_F2F_FS --gpu_id=3 \

