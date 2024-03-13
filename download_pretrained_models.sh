wget https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl -P resources/model_binaries/pretrained
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth -P resources/model_binaries/pretrained
python msc/convert-pretrained-swin-model-to-d2.py resources/model_binaries/pretrained/swin_tiny_patch4_window7_224.pth resources/model_binaries/pretrained/swin_tiny_patch4_window7_224.pkl
rm resources/model_binaries/pretrained/swin_tiny_patch4_window7_224.pth
