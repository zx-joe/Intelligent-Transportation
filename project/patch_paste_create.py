import torchvision.transforms as transforms_pytorch
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFile


def pasteImg(target_img_i, background,x,y):
    
    random_angle = np.random.randint(-30, 30)
    mask = Image.new('L', target_img_i.size, 255)
    target_img_i = target_img_i.rotate(random_angle,expand=True)
    mask = mask.rotate(random_angle, expand=True)
    background.paste(target_img_i, (x,y),mask)
    
    return background

def randomGaussian(image, mean=0.2, sigma=0.3):
    
    def gaussianNoisy(im, mean=0.2, sigma=0.3):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    #img.flags.writeable = True 
    width, height = img.shape[:2]
    #print(img.shape[:2])
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    return Image.fromarray(np.uint8(img))

def randomColor(image):
    # change intensities of pixels
    random_factor = np.random.randint(0, 31) / 10.  
    color_image = ImageEnhance.Color(image).enhance(random_factor)  
    random_factor = np.random.randint(10, 21) / 10.  
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor) 
    random_factor = np.random.randint(10, 21) / 10.  
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  
    random_factor = np.random.randint(0, 31) / 10.  
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)   



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
    is_bbox = np.random.rand() > 0 ## Modify: different probabilities
    if (is_bbox == False):
        x,y,h,w,x2,y2,h2,w2=0, 0, 0, 0, 0, 0, 0, 0
    else:
        h = np.random.randint(min_object_size, max_object_size)
        w = int(h*scale)
        target_img_i = target_img.copy()
        target_img_i = transform_train(target_img_i)
        target_img_i = target_img_i.resize((w,h))
        x = np.random.randint(0, (img_size_w - w) if (img_size_w - w)>0 else 0)
        y = np.random.randint(0, (img_size_h - h) if (img_size_h - h)>0 else 0)

        
        background = pasteImg(target_img_i, background,x,y)
        background = randomColor(background)
        #background = randomGaussian(background, mean=0.4, sigma=0.5)
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
    background.save("test/test_pasted_iamge.jpg")

    return background
