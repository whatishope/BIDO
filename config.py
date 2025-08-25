##################################################
# Training Config
##################################################
GPU = '0'
workers = 4                 
epochs = 50                
batch_size = 8
learning_rate = 1e-3        
num_attn_layers=2     
drop_rate=0.1
n_cluster=10
num_classes=2
D_xml=512
##################################################
# Model Config
##################################################
image_size = (448, 448)     
net = 'inception_mixed_6e'  
num_attentions = 32         
beta = 5e-2                 
##################################################
# Dataset/Path Config
##################################################
tag = 'bird'             


save_dir = './IIMD/'
model_name = 'best_model.pth'
log_name = 'train.log'


ckpt = False
# ckpt = save_dir + model_name

##################################################
# Eval Config
##################################################
visualize = False
eval_ckpt = save_dir + model_name
eval_savepath = './IIMD/'
