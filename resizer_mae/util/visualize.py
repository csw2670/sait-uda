import torch
import cv2

import matplotlib.pyplot as plt
import numpy as np

def show_image(image, title='', accuracy=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def reshape(x):
    x = x.squeeze()
    x = torch.einsum('hwc->chw', x)
    return x

def mse_metric(pred, target):
    pred = reshape(pred)
    mse = torch.mean((target - pred) ** 2)
    return mse.item()

#to check how accurately mae_model was trained
def print_reconstruction_result(iteration, img_path, model, transform):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image=image)['image']
    
    image = image.float()
    x = torch.tensor(image).cpu()
    # make it a batch-like
    x = x.unsqueeze(dim=0)
    
    # run MAE
    model = model.cpu()
    loss, y, mask = model(x, mask_ratio=0.75)
    y, mask = y.cpu(), mask.cpu()

    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach()
    
    x = torch.einsum('nchw->nhwc', x)
    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
    
    mse = mse_metric(im_paste, image)
    if iteration >= 20:
        return mse
    print("mse =",mse)
    
    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]
    
    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()
    
    return mse

def print_ssegmentation_result(source_path, gt_path, model):
    image = cv2.imread(source_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    mask[mask == 255] = 12
    
    augmented = transform(image=image, mask=mask)
    image = augmented['image']
    mask = augmented['mask']
    
    image = image.float().to(device)
    x = torch.tensor(image)
    
    x = x.unsqueeze(dim=0)
    
    # run final model
    outputs = model(x)
    outputs = torch.softmax(outputs, dim=1).cpu()
    outputs = torch.argmax(outputs, dim=1).numpy()
    outputs = outputs.squeeze()
    
    plt.rcParams['figure.figsize'] = [24, 24]
    
    image = cv2.imread(source_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("original image", fontsize=16)
    
    plt.subplot(1, 3, 2)
    plt.imshow(outputs)
    plt.title("predicted image", fontsize=16)

    plt.subplot(1, 3, 3)
    plt.imshow(mask)
    plt.title("ground truth", fontsize=16)
    
    plt.show()
    
    