import torch
import argparse

import numpy as np
import albumentations as A

from models import *
from data import *
from util import *

from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from torch import optim

def get_args_parser():
	parser = argparse.ArgumentParser('train the model', add_help=False)

	parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
	
	parser.add_argument('--epochs', default=1, type=int)
	
	parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
	
	parser.add_argument('--determine_pretrain', default='y', type=str,
                        help='Decide whether to train the mae model or not')
	
	parser.add_argument('--pretrain_data_path', default='/shared/s2/lab01/dataset/myfish', type=str,
                        help='pre-train dataset path')
	
	parser.add_argument('--train_data_path', default='../../data/', type=str,
                        help='train dataset path')
	
	parser.add_argument('--valid_data_path', default='../../data/', type=str,
                        help='validation dataset path')

	return parser

def get_transform(target_h, target_w):
	transform = A.Compose(
		[
			A.Resize(target_h, target_w),
			A.Normalize(),
			ToTensorV2()
		]
	)
	
	return transform

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae_resize_network, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    return model

def pretrain_mae(args):
	chkpt_dir = '../mae_visualize_vit_large.pth'
	mae_model = prepare_model(chkpt_dir, args.model).to(args.device)
	transform = get_transform(480, 270)

	mae_train_dataset = CustomDataset(data_path=args.pretrain_data_path, csv_file='/all_train_source.csv',
									transform=transform, pre_train=True)
	mae_train_dataloader = DataLoader(mae_train_dataset, batch_size=16, shuffle=True, num_workers=4)

	optimizer = optim.Adam(mae_model.parameters(), lr=1e-4)
	scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
											lr_lambda=lambda epoch: 0.95 ** epoch,
											last_epoch=-1,
											verbose=False)

	mae_model.train()

	for epoch in range(args.epochs):
		running_loss = 0.0  # Renamed from epoch_loss for clarity
		
		for images in tqdm(mae_train_dataloader):
			images = images.float().to(args.device)
			loss, y, mask = mae_model(images, 0.4)

			optimizer.zero_grad()  # Clear gradients before backward pass
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item()
			
		scheduler.step()
		
		# Calculate average loss for the epoch
		epoch_loss = running_loss / len(mae_train_dataloader)
		
		print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
	return mae_model

def import_mae(args):
	if args.determine_pretrain == 'y':
		return pretrain_mae(args)
	else:
		ckpt_dir = ''
		return torch.load(ckpt_dir)

def train(model, device, trainloader, optimizer, criterion):
	for epoch in range(5):
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



def main(args):
	mae_model = import_mae(args)
	
	for p in mae_model.parameters():
		p.requires_grad=False
	unet_model = UNet(mae_model).to(args.device)

	transform = get_transform(224, 224)
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = optim.Adam(unet_model.parameters(), lr=1e-4, weight_decay=1e-5)
	scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
	lr_lambda=lambda epoch: 0.95 ** epoch,
	last_epoch=-1,
	verbose=False)
	
	unet_train_dataset = CustomDataset(data_path=args.train_data_path, csv_file='train_source.csv', transform=transform)
	unet_train_dataloader = DataLoader(unet_train_dataset, batch_size=16, shuffle=True, num_workers=4)
	
	unet_validation_dataset = CustomDataset(data_path=args.valid_data_path, csv_file='val_source.csv', transform=transform)
	unet_validation_dataloader = DataLoader(unet_validation_dataset, batch_size=16, shuffle=False, num_workers=4)

	train(unet_model, args.device, unet_train_dataloader, optimizer, criterion)
	#result = test(model, args.device, testloader)
        
	#submit = pd.read_csv('/shared/s2/lab01/dataset/sait_uda/data/sample_submission.csv')
	#submit['mask_rle'] = result
	#submit.to_csv('./baseline_submit.csv', index=False)


if __name__ == '__main__':
	args = get_args_parser()
	args = args.parse_args()
	main(args)
                
