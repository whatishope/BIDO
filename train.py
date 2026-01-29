import os
import time
import config
import logging
import warnings

os.environ["WANDB_MODE"] = "offline"
import wandb
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.bido import BIDO
from datasets import get_trainval_datasets
from utils import CenterLoss, AverageMeter, con_loss_mahalanobis, MahalanobisMetric
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# GPU settings
assert torch.cuda.is_available()
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

cross_entropy_loss = nn.CrossEntropyLoss()
center_loss = CenterLoss()

loss_container = AverageMeter(name='loss')

def main(attn_weight=1.0, fusion_weight=1.0, con_weight=0.1, xml_weight=0.1,  gc_ce_weight=0.5,  batch_size=None):

    if batch_size is not None:
        config.batch_size = batch_size

    wandb.init(project="BIDO-opt", config={
        "attn_weight": attn_weight,
        "fusion_weight": fusion_weight,
        "con_weight": con_weight,
        "xml_weight": xml_weight,
        "gc_ce_weight": gc_ce_weight,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size
    })
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    logging.basicConfig(
        filename=os.path.join(config.save_dir, config.log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    train_dataset, validate_dataset = get_trainval_datasets(config.tag, config.image_size)

    train_loader, validate_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=config.workers, pin_memory=True), \
        DataLoader(validate_dataset, batch_size=config.batch_size * 4, shuffle=False,
                   num_workers=config.workers, pin_memory=True)
    num_classes = train_dataset.num_classes

    logs = {}
    start_epoch = 0

    net = BIDO(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=True,
               num_attn_layers=config.num_attn_layers, D_xml=config.D_xml)

    feature_center = torch.zeros(num_classes, config.num_attentions * net.num_features).to(device)

    if config.ckpt:
        checkpoint = torch.load(config.ckpt)
        logs = checkpoint['logs']
        start_epoch = int(logs['epoch'])
        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict)
        logging.info('Network loaded from {}'.format(config.ckpt))

        # load feature center
        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].to(device)
            logging.info('feature_center loaded from {}'.format(config.ckpt))

    logging.info('Network weights save to {}'.format(config.save_dir))

    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    metric = MahalanobisMetric(dim=768).to(device)  # 初始化一次
    learning_rate = logs['lr'] if 'lr' in logs else config.learning_rate
    optimizer = torch.optim.SGD(
        list(net.parameters()) + list(metric.parameters()),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(config.epochs, config.batch_size, len(train_dataset), len(validate_dataset)))
    logging.info('')

    for epoch in range(start_epoch, config.epochs):
        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.param_groups[0]['lr']

        logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))

        pbar = tqdm(total=len(train_loader), unit=' batches')
        pbar.set_description('Epoch {}/{}'.format(epoch + 1, config.epochs))

        train(logs=logs,
              data_loader=train_loader,
              net=net,
              feature_center=feature_center,
              optimizer=optimizer,
              pbar=pbar,
              metric=metric)
        validate(logs=logs,
                 data_loader=validate_loader,
                 net=net,
                 pbar=pbar)

        last_ckpt_path = os.path.join(config.save_dir, "last_model.pth")
        torch.save({
            'epoch': epoch + 1,
            'state_dict': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
            'logs': logs,
            'feature_center': feature_center
        }, last_ckpt_path)
        logging.info(f"Last model saved at epoch {epoch + 1}")

        if 'best_final_acc' not in logs or logs['val_final_acc'] > logs['best_final_acc']:
            logs['best_final_acc'] = logs['val_final_acc']
            best_final_ckpt_path = os.path.join(config.save_dir, "best_final_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
                'logs': logs,
                'feature_center': feature_center
            }, best_final_ckpt_path)
            logging.info(f"Best final model updated! New best final acc = {logs['val_final_acc']:.2f}%")

            wandb.log({
                "epoch": logs["epoch"],
                "lr": logs["lr"],
                "train_loss": logs.get("train_loss", 0),
                "train_attn_acc": logs.get("train_attn_acc", 0),
                "val_loss": logs.get("val_loss", 0),
                "val_fusion_acc": logs.get("val_fusion_acc", 0),
                "val_fusion_f1": logs.get("val_fusion_f1", 0),
                "val_final_acc": logs.get("val_final_acc", 0),
                "val_final_f1": logs.get("val_final_f1", 0),
            })
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(logs['val_loss'])
        else:
            scheduler.step()

        pbar.close()


def gc_joint_nll(z_gc, log_det, mu_gc, labels, class_prior=None):
    B, D = z_gc.size()
    K = mu_gc.size(0)

    z_expand = z_gc.unsqueeze(1)
    mu_expand = mu_gc.unsqueeze(0)
    dist_sq = torch.sum((z_expand - mu_expand) ** 2, dim=-1)

    dist_sq_y = torch.gather(dist_sq, dim=1, index=labels.unsqueeze(1)).squeeze(1)

    log_p_z_given_y = -0.5 * dist_sq_y

    if class_prior is None:
        log_p_y = 0.0
    else:
        log_prior = torch.log(class_prior + 1e-12)
        log_p_y = log_prior[labels]

    log_p_h_given_y = log_p_z_given_y + log_det

    log_p_hy = log_p_y + log_p_h_given_y

    loss_gc_nll = -log_p_hy.mean()
    return loss_gc_nll


def train(**kwargs):
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    optimizer = kwargs['optimizer']
    pbar = kwargs['pbar']
    metric = kwargs['metric']
    fusion_metric = Accuracy(task='binary').to(device)
    attn_metric = Accuracy(task='binary').to(device)

    loss_container.reset()
    attn_metric.reset()
    fusion_metric.reset()

    start_time = time.time()

    attn_weight = wandb.config.attn_weight
    fusion_weight = wandb.config.fusion_weight
    con_weight = wandb.config.con_weight
    xml_weight = wandb.config.xml_weight
    try:
        gc_ce_weight = wandb.config.gc_ce_weight
    except AttributeError:
        gc_ce_weight = 0.5

    net.train()
    for i, (X, xml_feats, y) in enumerate(data_loader):
        optimizer.zero_grad()

        X = X.to(device)
        xml_feats = xml_feats.to(device)
        y = y.to(device)

        y_pred_raw, y_pred_attn, embeddings, fusion_pred, xml_cls, p_gc, z_gc, log_det = net(X, xml_feats)

        cls_features = embeddings[:, -1, :]  # [B, 768]
        loss_con = con_loss_mahalanobis(cls_features, y, metric)
        loss_fusion = cross_entropy_loss(fusion_pred, y)
        loss_xml = cross_entropy_loss(xml_cls, y)
        loss_gc_ce = cross_entropy_loss(p_gc, y)
        loss_gc_nll = gc_joint_nll(z_gc, log_det, net.mu_gc, y)

        loss_attn = cross_entropy_loss(F.normalize(y_pred_attn, dim=-1), y)
        gc_nll_weight = 0.0005

        batch_loss = (
                gc_ce_weight * loss_gc_ce +
                gc_nll_weight * loss_gc_nll +
                attn_weight * loss_attn +
                fusion_weight * loss_fusion +
                con_weight * loss_con +
                xml_weight * loss_xml
        )


        batch_loss.backward()
        optimizer.step()

        with torch.no_grad():
            epoch_loss = loss_container(batch_loss.item())
            y_pred_attns = torch.argmax(y_pred_attn, dim=1)
            attn_metric.update(y_pred_attns, y)

        attn_acc_value = attn_metric.compute().item() * 100
        batch_info = 'Loss {:.4f}, Attention Acc {:.2f}, Contrastive loss ({:.2f})'.format(
            epoch_loss, attn_acc_value, loss_con)
        pbar.update()
        pbar.set_postfix_str(batch_info)

    logs['train_{}'.format(loss_container.name)] = epoch_loss
    logs['train_attn_acc'] = attn_acc_value
    logs['train_info'] = batch_info
    end_time = time.time()
    logging.info('Train: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))


def validate(**kwargs):
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    pbar = kwargs['pbar']
    all_preds = []
    all_labels = []

    fusion_metric = Accuracy(task='binary').to(device)
    fusion_prec_metric = Precision(task='binary').to(device)
    fusion_recall_metric = Recall(task='binary').to(device)
    fusion_f1_metric = F1Score(task='binary').to(device)

    gc_metric = Accuracy(task='binary').to(device)
    gc_prec_metric = Precision(task='binary').to(device)
    gc_recall_metric = Recall(task='binary').to(device)
    gc_f1_metric = F1Score(task='binary').to(device)

    gc_prec_metric.reset()
    gc_recall_metric.reset()
    gc_f1_metric.reset()
    gc_metric.reset()

    loss_container.reset()
    fusion_metric.reset()

    fusion_prec_metric.reset()
    fusion_recall_metric.reset()
    fusion_f1_metric.reset()
    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, xml_feats, y) in enumerate(data_loader):

            X = X.to(device)
            y = y.to(device)
            xml_feats = xml_feats.to(device)

            _, y_pred_attn, _, y_pred_fusion, xml_cls, p_gc, z_gc, log_det = net(X, xml_feats)

            final_prob = (
                    0.6 * F.softmax(p_gc, dim=1) +
                    0.6 * F.softmax(y_pred_fusion, dim=1)
            )
            final_pred = torch.argmax(final_prob, dim=1)

            all_preds.append(final_pred.cpu())
            all_labels.append(y.cpu())


            y_pred_fusions = torch.argmax(y_pred_fusion, dim=1)
            fusion_metric.update(y_pred_fusions, y)
            fusion_prec_metric.update(y_pred_fusions, y)
            fusion_recall_metric.update(y_pred_fusions, y)
            fusion_f1_metric.update(y_pred_fusions, y)

            y_pred_gcs = torch.argmax(p_gc, dim=1)
            gc_metric.update(y_pred_gcs, y)
            gc_prec_metric.update(y_pred_gcs, y)
            gc_recall_metric.update(y_pred_gcs, y)
            gc_f1_metric.update(y_pred_gcs, y)


    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    final_acc = accuracy_score(all_labels, all_preds) * 100
    final_f1 = f1_score(all_labels, all_preds, average='binary') * 100
    final_prec = precision_score(all_labels, all_preds, average='binary') * 100
    final_recall = recall_score(all_labels, all_preds, average='binary') * 100

    fusion_acc = fusion_metric.compute().item() * 100
    fusion_prec = fusion_prec_metric.compute().item() * 100
    fusion_recall = fusion_recall_metric.compute().item() * 100
    fusion_f1 = fusion_f1_metric.compute().item() * 100

    gc_acc = gc_metric.compute().item() * 100
    gc_prec = gc_prec_metric.compute().item() * 100
    gc_recall = gc_recall_metric.compute().item() * 100
    gc_f1 = gc_f1_metric.compute().item() * 100

    logs['val_gc_acc'] = gc_acc
    logs['val_gc_prec'] = gc_prec
    logs['val_gc_recall'] = gc_recall
    logs['val_gc_f1'] = gc_f1

    logs['val_final_acc'] = final_acc
    logs['val_final_prec'] = final_prec
    logs['val_final_recall'] = final_recall
    logs['val_final_f1'] = final_f1

    logs['val_fusion_acc'] = fusion_acc
    logs['val_fusion_prec'] = fusion_prec
    logs['val_fusion_recall'] = fusion_recall
    logs['val_fusion_f1'] = fusion_f1
    end_time = time.time()

    batch_info = '[fusion] acc {:.2f}, Prec {:.2f}, Recall {:.2f}, F1 {:.2f}'.format(
        fusion_acc, fusion_prec, fusion_recall, fusion_f1)
    pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))

    logging.info('Valid: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))
    logging.info('       [final] Acc {:.2f}, Prec {:.2f}, Recall {:.2f}, F1 {:.2f}'.format(
        final_acc, final_prec, final_recall, final_f1))
    logging.info('       [GC ] Acc {:.2f}, Prec {:.2f}, Recall {:.2f}, F1 {:.2f}'.format(
        gc_acc, gc_prec, gc_recall, gc_f1))


    logging.info('')

if __name__ == '__main__':
    main()

    # search_space = {
    #
    #     'batch_size': [8]
    # }
    #
    # keys, values = zip(*search_space.items())
    # for combo in itertools.product(*values):
    #     params = dict(zip(keys, combo))
    #     print(f"Grid Search - Trying {params}")
    #
    #     main(**params)
    #     wandb.finish()


