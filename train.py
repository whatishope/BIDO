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
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, con_loss_mahalanobis, MahalanobisMetric
from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# GPU settings
assert torch.cuda.is_available()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# General loss functions
cross_entropy_loss = nn.CrossEntropyLoss()
center_loss = CenterLoss()

# loss and metric
loss_container = AverageMeter(name='loss')


# raw_metric = TopKAccuracyMetric(topk=(1, 5))
# attn_metric = TopKAccuracyMetric(topk=(1, 5))


def main(attn_weight=1.0, fusion_weight=1.0, con_weight=0.1, xml_weight=0.1):
    wandb.init(project="BIDO-opt", config={
        "attn_weight": attn_weight,
        "fusion_weight": fusion_weight,
        "con_weight": con_weight,
        "xml_weight": xml_weight,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size
    })
    ##################################
    # Initialize saving directory
    ##################################
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    ##################################
    # Logging setting
    ##################################
    logging.basicConfig(
        filename=os.path.join(config.save_dir, config.log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    ##################################
    # Load dataset
    ##################################
    # print(config.batch_size)
    train_dataset, validate_dataset = get_trainval_datasets(config.tag, config.image_size)

    train_loader, validate_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=config.workers, pin_memory=True), \
        DataLoader(validate_dataset, batch_size=config.batch_size * 4, shuffle=False,
                   num_workers=config.workers, pin_memory=True)
    num_classes = train_dataset.num_classes

    ##################################
    # Initialize model
    ##################################
    logs = {}
    start_epoch = 0
    net = BIDO(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=True,
               num_attn_layers=config.num_attn_layers, D_xml=config.D_xml)

    # feature_center: size of (#classes, #attention_maps * #channel_features)
    feature_center = torch.zeros(num_classes, config.num_attentions * net.num_features).to(device)

    if config.ckpt:
        # Load ckpt and get state_dict
        checkpoint = torch.load(config.ckpt)

        # Get epoch and some logs
        logs = checkpoint['logs']
        start_epoch = int(logs['epoch'])

        # Load weights
        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict)
        logging.info('Network loaded from {}'.format(config.ckpt))

        # load feature center
        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].to(device)
            logging.info('feature_center loaded from {}'.format(config.ckpt))

    logging.info('Network weights save to {}'.format(config.save_dir))

    ##################################
    # Use cuda
    ##################################
    # device = torch.device("cuda:1")
    # net = net.to(device)
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    ##################################
    # Optimizer, LR Scheduler
    ##################################
    metric = MahalanobisMetric(dim=768).to(device) 
    learning_rate = logs['lr'] if 'lr' in logs else config.learning_rate
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.SGD(
        list(net.parameters()) + list(metric.parameters()),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-5
    )

    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # TRAINING
    ##################################
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
        logging.info(f" Last model saved at epoch {epoch + 1}")


        if 'best_fusion_acc' not in logs or logs['val_fusion_acc'] > logs['best_fusion_acc']:
            logs['best_fusion_acc'] = logs['val_fusion_acc']
            best_ckpt_path = os.path.join(config.save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
                'logs': logs,
                'feature_center': feature_center
            }, best_ckpt_path)
            logging.info(f" Best model updated! New best fusion acc = {logs['val_fusion_acc']:.2f}%")


        if 'best_final_acc' not in logs or logs['val_final_acc'] > logs['best_final_acc']:
            logs['best_final_acc'] = logs['val_final_acc']
            best_final_ckpt_path = os.path.join(config.save_dir, "best_final_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
                'logs': logs,
                'feature_center': feature_center
            }, best_final_ckpt_path)
            logging.info(f" Best final model updated! New best final acc = {logs['val_final_acc']:.2f}%")


            wandb.log({
                "epoch": logs["epoch"],
                "lr": logs["lr"],
                "train_loss": logs.get("train_loss", 0),
                "train_attn_acc": logs.get("train_attn_acc", 0),
                "val_loss": logs.get("val_loss", 0),
                "val_fusion_acc": logs.get("val_fusion_acc", 0),
                "val_fusion_f1": logs.get("val_fusion_f1", 0),
                "val_xml_acc": logs.get("val_xml_acc", 0),
                "val_raw_acc": logs.get("val_raw_acc", 0),
                "val_xml_f1": logs.get("val_xml_f1", 0),
                "val_raw_f1": logs.get("val_raw_f1", 0),
                "val_final_acc": logs.get("val_final_acc", 0),
                "val_final_f1": logs.get("val_final_f1", 0),
            })
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(logs['val_loss'])
        else:
            scheduler.step()

        pbar.close()


def train(**kwargs):
    # Retrieve training configuration
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    feature_center = kwargs['feature_center']
    optimizer = kwargs['optimizer']
    pbar = kwargs['pbar']
    metric = kwargs['metric']
    num_classes = net.module.num_classes if hasattr(net, 'module') else net.num_classes
    fusion_metric = Accuracy(task='binary').to(device)
    attn_metric = Accuracy(task='binary').to(device)

    # metrics initialization
    loss_container.reset()
    # raw_metric.reset()
    attn_metric.reset()
    fusion_metric.reset()  

    # begin training
    start_time = time.time()

    attn_weight = wandb.config.attn_weight
    fusion_weight = wandb.config.fusion_weight
    con_weight = wandb.config.con_weight
    xml_weight = wandb.config.xml_weight

    net.train()
    for i, (X, xml_feats, y) in enumerate(data_loader):
        optimizer.zero_grad()

        # obtain data for training
        X = X.to(device)
        xml_feats = xml_feats.to(device)
        y = y.to(device)

        # raw images forward
        y_pred_raw, y_pred_attn, embeddings, fusion_pred, xml_cls = net(X, xml_feats)

        # contrastive loss
        # y_pred_con = con_loss(embeddings, y)

        cls_features = embeddings[:, -1, :]  # [B, 768]
        loss_con = con_loss_mahalanobis(cls_features, y, metric)

        loss_fusion = cross_entropy_loss(fusion_pred, y)

        loss_xml = cross_entropy_loss(xml_cls, y)


      
        loss_attn = cross_entropy_loss(F.normalize(y_pred_attn, dim=-1), y)
        batch_loss = (
                attn_weight * loss_attn +
                fusion_weight * loss_fusion +
                con_weight * loss_con +
                xml_weight * loss_xml
        )

        # backward
        batch_loss.backward()
        optimizer.step()

        # metrics: loss and top-1,5 error
        with torch.no_grad():
            epoch_loss = loss_container(batch_loss.item())
            y_pred_attns = torch.argmax(y_pred_attn, dim=1)
            attn_metric.update(y_pred_attns, y)


        # end of this batch
        attn_acc_value = attn_metric.compute().item() * 100
        batch_info = 'Loss {:.4f}, Attention Acc {:.2f}, Contrastive loss ({:.2f})'.format(
            epoch_loss, attn_acc_value, loss_con)

        pbar.update()
        pbar.set_postfix_str(batch_info)

    # end of this epoch
    logs['train_{}'.format(loss_container.name)] = epoch_loss
    logs['train_attn_acc'] = attn_acc_value
    # logs['train_raw_{}'.format(raw_metric.name)] = epoch_raw_acc
    logs['train_info'] = batch_info
    end_time = time.time()

    # write log for this epoch
    logging.info('Train: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))


def validate(**kwargs):
    # Retrieve training configuration
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    pbar = kwargs['pbar']
    num_classes = net.module.num_classes if hasattr(net, 'module') else net.num_classes
    all_preds = []
    all_labels = []

    fusion_metric = Accuracy(task='binary').to(device)
    fusion_prec_metric = Precision(task='binary').to(device)
    fusion_recall_metric = Recall(task='binary').to(device)
    fusion_f1_metric = F1Score(task='binary').to(device)


    # metrics initialization
    loss_container.reset()
    fusion_metric.reset()
    fusion_prec_metric.reset()
    fusion_recall_metric.reset()
    fusion_f1_metric.reset()


    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, xml_feats, y) in enumerate(data_loader):
            # obtain data
            X = X.to(device)
            y = y.to(device)
            xml_feats = xml_feats.to(device)

            ##################################
            # Raw Image
            ##################################
            # _, y_pred_attn, _ = net(X)
            _, y_pred_attn, _, y_pred_fusion, xml_cls = net(X, xml_feats)



            final_prob = (
                    0.6 * F.softmax(y_pred_attn, dim=1) +
                    0.6 * F.softmax(y_pred_fusion, dim=1) +
                    0.1 * F.softmax(xml_cls, dim=1)
            )
            final_pred = torch.argmax(final_prob, dim=1)

            all_preds.append(final_pred.cpu())
            all_labels.append(y.cpu())

            # loss
            batch_loss = cross_entropy_loss(y_pred_fusion, y)
            epoch_loss = loss_container(batch_loss.item())


            y_pred_fusions = torch.argmax(y_pred_fusion, dim=1)
            fusion_metric.update(y_pred_fusions, y)
            fusion_prec_metric.update(y_pred_fusions, y)
            fusion_recall_metric.update(y_pred_fusions, y)
            fusion_f1_metric.update(y_pred_fusions, y)


   
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    final_acc = accuracy_score(all_labels, all_preds) * 100
    final_f1 = f1_score(all_labels, all_preds, average='macro') * 100
    final_prec = precision_score(all_labels, all_preds, average='macro') * 100
    final_recall = recall_score(all_labels, all_preds, average='macro') * 100

    fusion_acc = fusion_metric.compute().item() * 100
    fusion_prec = fusion_prec_metric.compute().item() * 100
    fusion_recall = fusion_recall_metric.compute().item() * 100
    fusion_f1 = fusion_f1_metric.compute().item() * 100

    logs['val_final_acc'] = final_acc
    logs['val_final_prec'] = final_prec
    logs['val_final_recall'] = final_recall
    logs['val_final_f1'] = final_f1

    logs['val_{}'.format(loss_container.name)] = epoch_loss

    logs['val_fusion_acc'] = fusion_acc
    logs['val_fusion_prec'] = fusion_prec
    logs['val_fusion_recall'] = fusion_recall
    logs['val_fusion_f1'] = fusion_f1
    end_time = time.time()

    batch_info = 'Val Loss {:.4f},  Val fusion acc {:.2f}, Prec {:.2f}, Recall {:.2f}, F1 {:.2f}'.format(
        epoch_loss, fusion_acc, fusion_prec, fusion_recall, fusion_f1)
    pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))
    # pbar.set_postfix_str('{}'.format( batch_info))

   
    logging.info('       [final] Acc {:.2f}, Prec {:.2f}, Recall {:.2f}, F1 {:.2f}'.format(
        final_acc, final_prec, final_recall, final_f1))
    logging.info('Valid: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))

    logging.info('')


import itertools

if __name__ == '__main__':
    main()

    # search_space = {
    #     'attn_weight': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #     'con_weight': [0.1, 0.3],
    #     'fusion_weight': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    #     'xml_weight': [0.1, 0.3]
    # }
    #
    # keys, values = zip(*search_space.items())
    # for combo in itertools.product(*values):
    #     params = dict(zip(keys, combo))
    #     print(f"Grid Search - Trying {params}")
    #
    #     wandb.init(
    #         project="BIDO-opt",
    #         name=f"attn_{params['attn_weight']}_con_{params['con_weight']}_fusion_{params['fusion_weight']}_xml_{params['xml_weight']}",
    #         config=params,
    #         reinit=True
    #     )
    #
    #     main(**params)
    #     wandb.finish()
