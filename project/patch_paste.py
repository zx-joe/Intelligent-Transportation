import torchvision.transforms as transforms_pytorch
import math
import numpy as np
from PIL import Image


def pasteImg(target_img_i, background,x,y):
    
    random_angle = np.random.randint(-20, 20)
    mask = Image.new('L', target_img_i.size, 255)
    target_img_i = target_img_i.rotate(random_angle,expand=True)
    mask = mask.rotate(random_angle, expand=True)
    background.paste(target_img_i, (x,y),mask)
    
    return background


def paste_img(background, target_img, image_id):
    transform_train = transforms_pytorch.Compose([
        transforms_pytorch.RandomHorizontalFlip(), ## Modify: You can remove
        transforms_pytorch.RandomAffine(math.pi/6, translate=None, shear=None, resample=False),# ## Modify + A fillcolor='white' can be added as argument
        transforms_pytorch.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5) ## Modify: different values
    ])

    img_w, img_h = target_img.size
    bg_w, bg_h = background.size
    scale = img_w / img_h
    img_size_w = bg_w
    img_size_h = bg_h
    min_object_size = bg_h//16 #16## Modify: different scale
    max_object_size = bg_h//2  ## Modify: different scale


    # paste target image
    image_annotations = []
    is_bbox = np.random.rand() > 0.5 ## Modify: different probabilities
    if (is_bbox == False):
        x,y,h,w,x2,y2,h2,w2=0, 0, 0, 0, 0, 0, 0, 0
    else:
        h = np.random.randint(min_object_size, max_object_size)
        w = int(h*scale)
        target_img_i = target_img.copy()
        target_img_i = transform_train(target_img_i)
        target_img_i = target_img_i.resize((w,h))
        
        #x = np.random.randint(0, (img_size_w - w) if (img_size_w - w)>0 else 0)
        #y = np.random.randint(0, (img_size_h - h) if (img_size_h - h)>0 else 0)
        
        if img_size_w - w >0:
            x = np.random.randint(0, (img_size_w - w)) # if (img_size_w - w)>0 else 0)
        else:
            x = 0
        if img_size_h - h > 0:
            y = np.random.randint(0, (img_size_h - h)) #if (img_size_h - h)>0 else 0)
        else:
            y = 0

        
        background = pasteImg(target_img_i, background,x,y)
        #background = background.resize((80,60), Image.ANTIALIAS)
        
    image_annotations.append({
                'image_id': image_id,
                'category_id': 0,
                'keypoints': [x+w//2, y+h//2,2 if is_bbox else 0],
                'num_keypoints' : 1 if is_bbox else 0,
                'bbox': [x, y, w, h],
                'iscrowd': 0,
                'segmentation': 0,
            })

    return background, image_annotations
