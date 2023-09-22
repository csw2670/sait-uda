import cv2
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer
        self.absolute_path = "/shared/s2/lab01/dataset/sait_uda/data"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.absolute_path + self.data.iloc[idx, 1][1:]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("image shape :", image.shape)
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image
        
        mask_path = self.absolute_path + self.data.iloc[idx, 2][1:]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 12
        print("mask sahep :", mask.shape)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        print("final_shape", image.shape, mask.shape)
        return image, mask