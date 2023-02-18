# python training.py --scratch -j 16 --model clf_CE --image_size 96x96 -b 64

# For multiclass
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --scratch --arch multiclass -j 32 --model multiclass_clf_4 --image_size 224x224 -b 384