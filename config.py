##################################################
# Training Config
##################################################
GPU = '0'
workers = 4                 
epochs = 50                
batch_size = 8
learning_rate = 1e-3        # 学习率
num_attn_layers=2       #注意力层数
drop_rate=0.1
n_cluster=10
num_classes=2
D_xml=512
##################################################
# Model Config
##################################################
image_size = (448, 448)     # 调整图像大小
net = 'inception_mixed_6e'  # feature extractor
num_attentions = 32         # 每一张图对应的局部特征区间
beta = 5e-2                 # 特征重心
##################################################
# Dataset/Path Config
##################################################
tag = 'bird'             

#存储.ckpt文件位置
save_dir = './IIMD/'
model_name = 'best_model.pth'
log_name = 'train.log'

# checkpoint model for resume training
ckpt = False
# ckpt = save_dir + model_name

##################################################
# Eval Config
##################################################
visualize = False
eval_ckpt = save_dir + model_name
eval_savepath = './IIMD/'