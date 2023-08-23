import torch
import numpy as np
import albumentations as A

from models import *
from data import *
from utils import *

from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

def train(model, device, trainloader, optimizer, criterion):
	for epoch in range(20):
		model.train()
		epoch_loss = 0
		for images, masks in tqdm(trainloader):
			images = images.float().to(device)
			masks = masks.long().to(device)

			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, masks.squeeze(1))
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()

		print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(trainloader)}')
		
def test(model, device, testloader):
    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(testloader):
            images = images.float().to(device)
            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1).cpu()
            outputs = torch.argmax(outputs, dim=1).numpy()
            
            for pred in outputs:
                pred = pred.astype(np.uint8)
                pred = Image.fromarray(pred)
                pred = pred.resize((960, 540), Image.NEAREST)
                pred = np.array(pred)
                
                for class_id in range(12):
                    class_mask = (pred == class_id).astype(np.uint8)
                    if np.sum(class_mask) > 0:
                        mask_rle = rle_encode(class_mask)
                        result.append(mask_rle)
                    else:
                        result.append(-1)
    return result

def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = UNet().to(device)

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	transform = A.Compose(
		[
			A.Resize(224, 224),
			A.Normalize(),
			ToTensorV2()
		]
	)
    
	trainset = CustomDataset(csv_file='/shared/s2/lab01/dataset/sait_uda/data/train_source.csv', transform=transform)
	trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)
	testset = CustomDataset(csv_file='/shared/s2/lab01/dataset/sait_uda/data/test.csv', transform=transform, infer=True)
	testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)

	train(model, device, trainloader, optimizer, criterion)
	result = test(model, device, testloader)
        
	submit = pd.read_csv('/shared/s2/lab01/dataset/sait_uda/data/sample_submission.csv')
	submit['mask_rle'] = result
	submit.to_csv('./baseline_submit.csv', index=False)


if __name__ == '__main__':
	if 1:
		main()
                
