from .img_dataset import ImgDataset


def get_trainval_datasets(tag, resize):
    return ImgDataset(phase='train', resize=resize), ImgDataset(phase='val', resize=resize)