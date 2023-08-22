from models import *
from data import *
import torch
import albumentations as A
from torch.utils.data import DataLoader
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2

def train():
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

	dataset = CustomDataset(csv_file='./train_source.csv', transform=transform)
	dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

	for epoch in range(20):
		model.train()
		epoch_loss = 0
		for images, masks in tqdm(dataloader):
			images = images.float().to(device)
			masks = masks.long().to(device)

			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, masks.squeeze(1))
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()

		print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')

if __name__ == '__main__':
	train()
