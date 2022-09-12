#!/bin/bash



#python ../test.py --model=Cannyadd1_ShareTrm  --img_size=320 --version=320_c40_bs24_lr_1e10 \
#--test_dataset=CelebDF --gpu_id=0 \
#--start=30000 --end=30001 \
#--data_root=/workspace/data/Celeb-DFv2

#python ../test.py --model=NormNoise_ShareTrm --img_size=320 --version=320_c40_bs24_lr_1e10 \
#--test_dataset=CelebDF --gpu_id=1 \
#--start=8000 --end=8001 \
#--data_root=/workspace/data/Celeb-DFv2

#
#python ../test.py --model=CannyNoiseNorm_ShareTrm --img_size=320 --version=320_c40_bs24_lr_1e10 \
#--test_dataset=CelebDF --gpu_id=0 \
#--start=12000 --end=12001 \
#--data_root=/workspace/data/Celeb-DFv2

#python ../test.py --model=resnet34 --img_size=320 --version=320_c40_bs16_lr_1e10  \
#-test_dataset=CelebDF --gpu_id=0 \
#--start=1000 --end=60001 \
#--data_root=/workspace/data/Celeb-DFv2

#python ../test.py --model=fca34 --img_size=320 --version=320_c40_bs16_lr_1e10  \
#--test_dataset=CelebDF --gpu_id=0 \
#--start=1000 --end=60001 \
#--data_root=/workspace/data/Celeb-DFv2


#python ../test.py --model=ResCanny --img_size=320 --version=320_c40_bs16_lr_1e10  \
#--test_dataset=CelebDF --gpu_id=0 \
#--start=1000 --end=50001 \
#--data_root=/workspace/data/Celeb-DFv2


#python ../test.py --model=ResNoise --img_size=320 --version=320_c40_bs16_lr_1e10  \
#--test_dataset=CelebDF --gpu_id=0 \
#--start=1000 --end=50001 \
#--data_root=/workspace/data/Celeb-DFv2

python ../test.py --model=ResCannyNoise --img_size=320 --version=320_c40_bs16_lr_1e10  \
--test_dataset=CelebDF --gpu_id=1 \
--start=1000 --end=50001 \
--data_root=/workspace/data/Celeb-DFv2

