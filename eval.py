import os
import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models.bido import BIDO
from datasets import get_trainval_datasets
# from utils import TopKAccuracyMetric, batch_augment
from utils import TopKAccuracyMetric
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets.TestDataset import TestDataset


# GPU settings
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# visualize
visualize = config.visualize
savepath = config.eval_savepath
if visualize:
    os.makedirs(savepath, exist_ok=True)

ToPILImage = transforms.ToPILImage()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)



def main():
    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    try:
        ckpt = config.eval_ckpt
    except:
        logging.info('Set ckpt for evaluation in config.py')
        return

    ##################################
    # Dataset for testing
    ##################################
    # _, test_dataset = get_trainval_datasets(config.tag, resize=config.image_size)
    # test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
    #                          num_workers=2, pin_memory=True)
    test_dataset = TestDataset(
        dex_dir='./android_software_2016/2019/test',
        xml_dir='./android_software_2016/2019/xml_images_test',
        resize=config.image_size
    )

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    ##################################
    # Initialize model
    ##################################
    net = BIDO(num_classes=2, M=config.num_attentions, net=config.net, pretrained=True, num_attn_layers=config.num_attn_layers, D_xml=config.D_xml)


    # Load ckpt and get state_dict
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    net.load_state_dict(state_dict)
    logging.info('Network loaded from {}'.format(ckpt))

    ##################################
    # use cuda
    ##################################
    # net.to(device)
    # if torch.cuda.device_count() > 1:
    #     net = nn.DataParallel(net)



    raw_accuracy = Accuracy(task='binary')
    acc_p3 = Accuracy(task='binary')

    raw_accuracy.reset()
    acc_p3.reset()

    prec_metric = Precision(task='binary')
    recall_metric = Recall(task='binary')
    f1_metric = F1Score(task='binary')
    prec_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    all_preds = []
    all_labels = []

    net.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), unit=' batches')
        pbar.set_description('Validation')
        for i, (X, xml_feats, y) in enumerate(test_loader):
            # X = X.to(device)
            # xml_feats = xml_feats.to(device)
            # y = y.to(device)


            _, y_pred_attn, _ , fusion_pred, xml_pred= net(X, xml_feats)

            preds = torch.argmax(y_pred_attn, dim=1)
            raw_accuracy.update(preds, y)


            fusion_preds = torch.argmax(fusion_pred, dim=1)
            acc_p3.update(fusion_preds, y)
            prec_metric.update(fusion_preds, y)
            recall_metric.update(fusion_preds, y)
            f1_metric.update(fusion_preds, y)

            final_prob = (
                    1 * F.softmax(y_pred_attn, dim=1) +
                    1 * F.softmax(fusion_pred, dim=1) +
                    0.1 * F.softmax(xml_pred, dim=1)
            )
            final_pred = torch.argmax(final_prob, dim=1)

            all_preds.append(final_pred.cpu())
            all_labels.append(y.cpu())

            # end of this batch
            acc1_batch = raw_accuracy.compute().item() * 100
            acc3_batch = acc_p3.compute().item() * 100
            batch_info = f'Val Acc: Raw Top-1 {acc1_batch:.2f}%, Fusion Top-1 {acc3_batch:.2f}%'
            pbar.update()
            pbar.set_postfix_str(batch_info)

        pbar.close()

        acc3 = acc_p3.compute().item() * 100

        fusion_prec = prec_metric.compute().item() * 100
        fusion_recall = recall_metric.compute().item() * 100
        fusion_f1 = f1_metric.compute().item() * 100

        # compute final metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        final_acc = accuracy_score(all_labels, all_preds) * 100
        final_f1 = f1_score(all_labels, all_preds, average='binary') * 100
        final_prec = precision_score(all_labels, all_preds, average='binary') * 100
        final_recall = recall_score(all_labels, all_preds, average='binary') * 100

        print("\n=== Evaluation Results ===")
        print(f"Accuracy p3 (fusion)   : Top-1 {acc3:.2f}%")
        print(f"Fusion Metrics         : Precision {fusion_prec:.2f}%, Recall {fusion_recall:.2f}%, F1 {fusion_f1:.2f}%")
        print(f"Final Metrics         : accuracy {final_acc:.2f}%, Precision {final_prec:.2f}%, Recall {final_recall:.2f}%, F1 {final_f1:.2f}%")


if __name__ == '__main__':
    main()
