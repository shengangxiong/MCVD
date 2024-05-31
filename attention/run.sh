nohup python cifar.py --save logs/resnet_40_1_teacher --depth 40 --width 1 > mylogs/my_1.log 2>&1 &
nohup python cifar.py --save logs/resnet_16_2_teacher --depth 16 --width 2 > mylogs/my_2.log 2>&1 &
nohup python cifar.py --save logs/resnet_40_2_teacher --depth 40 --width 2 > mylogs/my_3.log 2>&1 &