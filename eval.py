import os
import logging
import warnings
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models.bido import BIDO
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets.TestDataset import TestDataset


# GPU settings
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

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

    test_dataset = TestDataset(
        dex_dir='/home/user/android_obf_t_v/test',
        xml_dir='/home/user/android_obf_t_v/xml_images',
        resize=config.image_size
    )

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    net = BIDO(num_classes=2, M=config.num_attentions, net=config.net, pretrained=True, num_attn_layers=config.num_attn_layers, D_xml=config.D_xml)

    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    net.load_state_dict(state_dict)
    logging.info('Network loaded from {}'.format(ckpt))

    raw_accuracy = Accuracy(task='binary')
    acc_p3 = Accuracy(task='binary')

    # 初始化
    raw_accuracy.reset()
    acc_p3.reset()

    prec_metric = Precision(task='binary')
    recall_metric = Recall(task='binary')
    f1_metric = F1Score(task='binary')
    prec_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    # GC 头的指标
    gc_acc_metric = Accuracy(task='binary')
    gc_prec_metric = Precision(task='binary')
    gc_recall_metric = Recall(task='binary')
    gc_f1_metric = F1Score(task='binary')

    gc_acc_metric.reset()
    gc_prec_metric.reset()
    gc_recall_metric.reset()
    gc_f1_metric.reset()

    all_preds = []
    all_labels = []

    fi_agree_sum = 0
    fi_total = 0

    net.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), unit=' batches')
        pbar.set_description('Validation')
        for i, (X, xml_feats, y) in enumerate(test_loader):
            # X = X.to(device)
            # xml_feats = xml_feats.to(device)
            # y = y.to(device)

            _, y_pred_attn, _ , fusion_pred, xml_pred, p_gc, z_gc, log_det = net(X, xml_feats)

            preds = torch.argmax(y_pred_attn, dim=1)
            raw_accuracy.update(preds, y)

            fusion_preds = torch.argmax(fusion_pred, dim=1)
            acc_p3.update(fusion_preds, y)
            prec_metric.update(fusion_preds, y)
            recall_metric.update(fusion_preds, y)
            f1_metric.update(fusion_preds, y)

            gc_preds = torch.argmax(p_gc, dim=1)
            gc_acc_metric.update(gc_preds, y)
            gc_prec_metric.update(gc_preds, y)
            gc_recall_metric.update(gc_preds, y)
            gc_f1_metric.update(gc_preds, y)

            final_prob = (
                    1 * F.softmax(p_gc, dim=1) +
                    1 * F.softmax(fusion_pred, dim=1)
            )
            final_pred = torch.argmax(final_prob, dim=1)

            fi_agree_sum += (fusion_preds == gc_preds).sum().item()
            fi_total += y.size(0)

            all_preds.append(final_pred.cpu())
            all_labels.append(y.cpu())

            acc3_batch = acc_p3.compute().item() * 100
            batch_info = f'Val Acc: Raw Top-1 {acc1_batch:.2f}%, Fusion Top-1 {acc3_batch:.2f}%'
            pbar.update()
            pbar.set_postfix_str(batch_info)

        pbar.close()

        acc3 = acc_p3.compute().item() * 100

        fusion_prec = prec_metric.compute().item() * 100
        fusion_recall = recall_metric.compute().item() * 100
        fusion_f1 = f1_metric.compute().item() * 100

        gc_acc = gc_acc_metric.compute().item() * 100
        gc_prec = gc_prec_metric.compute().item() * 100
        gc_recall = gc_recall_metric.compute().item() * 100
        gc_f1 = gc_f1_metric.compute().item() * 100

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        final_acc = accuracy_score(all_labels, all_preds) * 100
        final_f1 = f1_score(all_labels, all_preds, average='binary') * 100
        final_prec = precision_score(all_labels, all_preds, average='binary') * 100
        final_recall = recall_score(all_labels, all_preds, average='binary') * 100

        fi = 100.0 * fi_agree_sum / max(1, fi_total)

        print(f"FI (Fusion vs GC agreement): {fi:.2f}%")
        logging.info(f"[EVAL] FI_agree(fusion,gc)={fi:.2f}%")

        print("\n=== Evaluation Results ===")
        print(f"Fusion Metrics         : Acc {acc3:.2f}%, Precision {fusion_prec:.2f}%, Recall {fusion_recall:.2f}%, F1 {fusion_f1:.2f}%")
        print(f"GC Metrics             : Acc {gc_acc:.2f}%, Precision {gc_prec:.2f}%, Recall {gc_recall:.2f}%, F1 {gc_f1:.2f}%")
        print(f"Final Metrics          : Acc {final_acc:.2f}%, Precision {final_prec:.2f}%, Recall {final_recall:.2f}%, F1 {final_f1:.2f}%")


if __name__ == '__main__':
    main()
