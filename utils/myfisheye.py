import os
from defisheye import Defisheye

def mydefisheye(idx):
    dtype = 'linear'
    format = 'fullframe'
    fov = 90
    pfov = 60

    server_url = "/shared/s2/lab01/dataset/sait_uda/data"
    local_url = "/Users/csw/Downloads/data"
    new_train_dir = "/defisheye_train"

    if not os.path.exists(local_url + new_train_dir):
        os.mkdir(local_url)
        
    img = local_url + f"/train_target_image/TRAIN_TARGET_{idx}.png"
    img_out = local_url + new_train_dir + f"/TRAIN_TARGET_{idx}_v2.png"
    print(img_out)
    obj = Defisheye(img, dtype=dtype, format=format, fov=fov, pfov=pfov)

    # To save image locally 
    obj.convert(outfile=img_out)

    # To use the converted image in memory

    new_image = obj.convert()

mydefisheye('0000')
mydefisheye('0001')