import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import get_transform

class TestDataset(Dataset):
    def __init__(self, dex_dir, xml_dir, resize=500, phase='test'):
        self.dex_dir = dex_dir
        self.phase = phase
        self.xml_dir = xml_dir
        self.label_map = {'L': 0, 'M': 1}
        self.resize = resize
        self.num_classes = 2

        self.dex_images = sorted([
            f for f in os.listdir(dex_dir)
            if f.endswith('.jpg') and f[0] in self.label_map
        ])

        self.transform_dex = get_transform(self.resize, self.phase)
        self.transform_xml = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        # self.transform_xml = transforms.Compose([
        #     transforms.Resize((256, 256)),  # 保证输入尺寸
        #     transforms.CenterCrop(224),  # 或 RandomResizedCrop(224) 用于训练
        #     transforms.ToTensor(),
        #     transforms.Normalize(  # ImageNet 上的标准 mean/std
        #         mean=[0.485, 0.456, 0.406],  # R, G, B 三通道均值
        #         std=[0.229, 0.224, 0.225]
        #     )
        # ])

    def __len__(self):
        return len(self.dex_images)

    def __getitem__(self, idx):
        dex_name = self.dex_images[idx]
        label = self.label_map[dex_name[0]]

        # 加载 dex 图像
        dex_path = os.path.join(self.dex_dir, dex_name)
        dex_image = Image.open(dex_path).convert('RGB')
        dex_image = self.transform_dex(dex_image)

        # 加载 xml 图像（同名）
        xml_path = os.path.join(self.xml_dir, dex_name)
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML 图像不存在: {xml_path}")
        xml_image = Image.open(xml_path).convert('RGB')
        xml_image = self.transform_xml(xml_image)

        return dex_image, xml_image, label
