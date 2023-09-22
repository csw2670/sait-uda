import cv2
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, csv_file, transform=None, pre_train=False, infer=False):
        self.csv_file_path = data_path + csv_file
        self.data = pd.read_csv(self.csv_file_path)
        self.data_path = data_path
        self.transform = transform
        self.pre_train = pre_train
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data.iloc[idx, 1][:2] == '..' :
            img_path = self.data_path[:len(self.data_path)-6] + 'fish' + self.data.iloc[idx, 1][2:]
        else:
            img_path = self.data_path + self.data.iloc[idx, 1][1:]
            
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.pre_train or self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
                image = image.permute(0, 2, 1)
            return image
        
        mask_path = self.data_path + self.data.iloc[idx, 2][1:]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 12

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask