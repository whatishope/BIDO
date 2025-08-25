import os
import pdb
from PIL import Image
from torch.utils.data import Dataset
from utils import get_transform
from torchvision import transforms




DATAPATH = './android_software1'
image_path = {}
image_label = {}


class ImgDataset(Dataset):
    """
    # Description:
        Dataset for retrieving CUB-200-2011 images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', resize=500, xml_dir='xml_images'):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize
        self.xml_dir = os.path.join(DATAPATH, xml_dir)
        self.image_id = []
        self.num_classes = 2

        # get image path from images.txt
        with open(os.path.join(DATAPATH, 'images.txt')) as f:
            for line in f.readlines():
                id, path = line.strip().split(' ')
                image_path[id] = path

        # get image label from image_class_labels.txt
        with open(os.path.join(DATAPATH, 'image_class_labels.txt')) as f:
            for line in f.readlines():
                id, label = line.strip().split(' ')
                image_label[id] = int(label)

        # get train/test image id from train_test_split.txt
        with open(os.path.join(DATAPATH, 'train_test_split.txt')) as f:
            for line in f.readlines():
                image_id, is_training_image = line.strip().split(' ')
                is_training_image = int(is_training_image)

                if self.phase == 'train' and is_training_image:
                    self.image_id.append(image_id)
                if self.phase in ('val', 'test') and not is_training_image:
                    self.image_id.append(image_id)

        # transform
        self.transform = get_transform(self.resize, self.phase)
        self.transform_xml = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        # get image id
        image_id = self.image_id[item]

        # dex_image
        image = Image.open(os.path.join(DATAPATH, 'images', image_path[image_id])).convert('RGB')  # (C, H, W)
        image = self.transform(image)



        # Load XML image
        xml_path = os.path.join(self.xml_dir, os.path.basename(image_path[image_id]))
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML image not found: {xml_path}")
        xml_image = Image.open(xml_path).convert('RGB')
        xml_image = self.transform_xml(xml_image)

        # return image and label
        return image,  xml_image, image_label[image_id] - 1

    def __len__(self):
        return len(self.image_id)


if __name__ == '__main__':
    ds = ImgDataset('train')
    print(len(ds))
    for i in range(0, 10):
        image, label = ds[i]
        print(image.shape, label)
