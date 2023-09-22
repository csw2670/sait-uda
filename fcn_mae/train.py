import torch
import argparse
import sys
import os.path
import csv

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
from monai.losses.dice import DiceLoss, one_hot


def get_args_parser():
	parser = argparse.ArgumentParser('train the model', add_help=False)

	parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
	
	parser.add_argument('--pre_train_epochs', default=180, type=int)
	parser.add_argument('--train_epochs', default=600, type=int)
	
	parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
	
	parser.add_argument('--determine_pretrain', default='n', type=str,
                        help='Decide whether to train the mae model or not')
	
	parser.add_argument('--pretrain_data_path', default='/shared/s2/lab01/dataset/myfish', type=str,
                        help='pre-train dataset path')
	
	parser.add_argument('--train_data_path', default='../../data/', type=str,
                        help='train dataset path')
	
	parser.add_argument('--valid_data_path', default='../../data/', type=str,
                        help='validation dataset path')

	parser.add_argument('--save_pretrained_model', default='y', type=str,
                        help='validation dataset path')
	
	parser.add_argument('--pretrained_ckpt_dir', default='VitMae_fft_full_data_mask04_e50.pth', type=str,
                        help='validation dataset path')
	parser.add_argument('--pretraining_ckpt_dir', default='../mae_visualize_vit_large.pth', type=str,
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
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    return model

def pretrain_mae(args):
	#chkpt_dir = '../mae_visualize_vit_large.pth'
	#mae_model = prepare_model(chkpt_dir, 'mae_vit_large_patch16').to(args.device)
	ckpt_dir = args.pretraining_ckpt_dir
	mae_model = torch.load(ckpt_dir).to(args.device)
	transform = get_transform(224, 224)

	mae_train_dataset = CustomDataset(data_path=args.pretrain_data_path, csv_file='/all_train_source.csv',
									transform=transform, pre_train=True)
	mae_train_dataloader = DataLoader(mae_train_dataset, batch_size=16, shuffle=True, num_workers=4)

	optimizer = optim.Adam(mae_model.parameters(), lr=1e-4)
	scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
											lr_lambda=lambda epoch: 0.95 ** epoch,
											last_epoch=-1,
											verbose=False)

	mae_model.train()

	for epoch in range(args.pre_train_epochs):
		running_loss = 0.0  # Renamed from epoch_loss for clarity
		
		for images in tqdm(mae_train_dataloader):
			images = images.float().to(args.device)
			loss, y, mask = mae_model(images, 0.75)

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
		ckpt_dir = args.pretraining_ckpt_dir
		#return torch.load(ckpt_dir)
		return prepare_model(ckpt_dir).to(args.device)

def miou(result_pred, result_gt, device):
    category = 13 * result_gt + result_pred
    category = category.flatten()
    category = torch.cat([category, torch.tensor([168]).to(device)], dim=0) 
    
    con_matrix = torch.bincount(category)
    con_matrix = con_matrix.reshape(13,13)
    con_matrix[12][12] -= 1
    
    iou = []
    col_sum = torch.sum(con_matrix, dim=0)
    row_sum = torch.sum(con_matrix, dim=1)
    for i in range(len(con_matrix)):
        intersection = con_matrix[i][i]
        if intersection == 0 and row_sum[i] != 0:
            iou.append(0)
        else:
            union = col_sum[i] + row_sum[i] - intersection
            if union != 0:
                iou.append((intersection/union).cpu().numpy())
    return np.mean(iou)

def change_format_ito(x, mode="valid"):
    x = torch.softmax(x, dim=1)
    outputs = torch.argmax(x, dim=1)
    
    return outputs

def do_test(temp_mae_model, temp_decoder_model, fcnhead_dataloader, device, criterion):
    miou_result = 0
    running_loss = 0
    with torch.no_grad():
        miou_results = []
        for images, masks in tqdm(fcnhead_dataloader):
            images = images.float().to(device)
            masks = masks.long().to(device)
            encoder_output, _, _ = temp_mae_model.forward_encoder(images, mask_ratio=0.75)
            outputs = temp_decoder_model(encoder_output)
            outputs = change_format_ito(outputs)
            outputs_loss = temp_decoder_model(encoder_output)
            loss = criterion(outputs_loss, masks.squeeze(1))

            batch_results = []
            for idx in range(len(outputs)):
                batch_results.append(miou(outputs[idx], masks[idx], device))
            miou_results.append(np.mean(batch_results))
            running_loss += loss.item()
        running_loss = running_loss / len(fcnhead_dataloader)
        miou_result = np.mean(miou_results)
    return miou_result, running_loss

def train(mae_model, decoder_model, epochs, device, trainloader, validationloader, optimizer, criterion):
	five_avg_training_miou = []
	five_avg_training_loss = []
	five_avg_validation_miou = []
	five_avg_validation_loss = []
	loss_history = []
	for epoch in range(1, epochs+1):
		decoder_model.train()
		running_loss = 0
	
		for images, masks in tqdm(trainloader):
			images = images.float().to(device)
			masks = masks.long().to(device)
			outputs, _,  _ = mae_model.forward_encoder(images, mask_ratio=0.75)
			outputs = decoder_model(outputs)
			loss = criterion(outputs, masks.squeeze(1))
			
			optimizer.zero_grad()  # Clear gradients before backward pass
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			
		loss_history.append(running_loss / len(trainloader))

		if epoch % 5 == 0:
			temp_mae_model = mae_model
			temp_decoder_model = decoder_model
			
			miou_training, loss_training = do_test(temp_mae_model, temp_decoder_model, trainloader, device, criterion)
			five_avg_training_miou.append(miou_training)
			five_avg_training_loss.append(loss_training)
			
			miou_validation, loss_validation = do_test(temp_mae_model, temp_decoder_model, validationloader, device, criterion)
			five_avg_validation_miou.append(miou_validation)
			five_avg_validation_loss.append(loss_validation)
		
		if epoch % 100 == 0:
			save_file_name = './FCN_fft_e' + str(epoch) +'.pt'
			if os.path.isfile(save_file_name):
				os.remove(save_file_name)
			torch.save({'epoch': epoch,
						'model_state_dict': decoder_model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),
						'loss_history': loss_history
						}, save_file_name)
	
	save_file_name = 'fcn_mask075_e400_result.csv'
	if os.path.isfile(save_file_name):
		os.remove(save_file_name)

	f = open(save_file_name,'w', newline='')
	wr = csv.writer(f)
	wr.writerow(['training_miou', five_avg_training_miou])
	wr.writerow(['training_loss', five_avg_training_loss])
	wr.writerow(['validation_miou', five_avg_training_miou])
	wr.writerow(['validation_loss', five_avg_validation_loss])
	
	f.close()
		
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
	final_model = FCNDecoder().to(args.device)

	transform = get_transform(224, 224)
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = optim.Adam(final_model.parameters(), lr=1e-4, weight_decay=1e-5)
	scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
	lr_lambda=lambda epoch: 0.95 ** epoch,
	last_epoch=-1,
	verbose=False)
	
	train_dataset = CustomDataset(data_path=args.train_data_path, csv_file='train_source.csv', transform=transform)
	train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
	
	validation_dataset = CustomDataset(data_path=args.valid_data_path, csv_file='val_source.csv', transform=transform)
	validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=False, num_workers=4)

	train(mae_model, final_model, args.train_epochs, args.device, train_dataloader, validation_dataloader, optimizer, criterion)
	
	#validation(final_model, args.device, unet_validation_dataloader)
	#result = test(model, args.device, testloader)
        
	#submit = pd.read_csv('/shared/s2/lab01/dataset/sait_uda/data/sample_submission.csv')
	#submit['mask_rle'] = result
	#submit.to_csv('./baseline_submit.csv', index=False)


if __name__ == '__main__':
	sys.path.insert(0, '../mae/models/')
	args = get_args_parser()
	args = args.parse_args()
	main(args)
                
