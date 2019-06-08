from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
import logging

def randomGaussian(image, mean=0.2, sigma=0.3):
    
    def gaussianNoisy(im, mean=0.2, sigma=0.3):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    img.flags.writeable = True 
    width, height = img.shape[:2]
    #print(img.shape[:2])
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    return Image.fromarray(np.uint8(img))
   
    
    
    
def randomRotation(image, mode=Image.BICUBIC):#mode=Image.BICUBIC

    random_angle = np.random.randint(-30, 30)
    image = image.convert('RGBA')
    rot = image.rotate(random_angle, mode)
    fill_background = Image.new('RGBA', rot.size, (255,255,255,255))
    
    out = Image.composite(rot, fill_background, mask=rot)
    
    return out
 
    
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
    

def pasteImg(source, background):
    # source is the target image we want to paste onto the background image
    #background should be type of PIL.Image.Image
    sourceImg = source
    
    height_b, width_b, c_b = np.asarray(background).shape
    # rescale the source image

    # note that resize configuration (width,height)
    sourceImg = sourceImg.resize((width_b,height_b), Image.ANTIALIAS)
    height_s,width_s,c_s = np.asarray(sourceImg).shape
    
    #rotation
    sourceImg = randomRotation(sourceImg)
    #Gaussian noise
    sourceImg = randomGaussian(sourceImg)
    #Change intensities
    sourceImg = randomColor(sourceImg)
    
    
    
    # random resize size
    resized_height_s = random.randint(int(height_s/4),height_s)
    resized_width_s = random.randint(int(width_s/4),width_s)
    
    #resized_height_s = int(height_s/2)
    #resized_width_s = int(width_s/2)
    
    # note that resize configuration (width,height)
    resizedImg = sourceImg.resize((resized_width_s,resized_height_s), Image.ANTIALIAS)
    
    
    #print(int(resized_height_s/2))
    #print(int(height_b - resized_height_s/2))
    #print(height_b)
    
    center_img_height = random.randint(int(resized_height_s/2), int(height_b - resized_height_s/2))
    center_img_width = random.randint(int(resized_width_s/2), int(width_b - resized_width_s/2))
    #center_img_height = random.randint(1,height_b)
    #center_img_width = random.randint(1,width_b)
    #background_copy = Image.fromarray(np.asarray(background))
    background_copy = background#.convert('RGB')
    #print(type(background_copy))
    #print(type(resizedImg))
    background_copy.paste(resizedImg,(center_img_width-int(resized_width_s/2),center_img_height-int(resized_height_s/2)))
    #c = PhotoImage(background_copy)
    keypoint_x = center_img_height#-int(resized_height_s/2)
    keypoint_y = center_img_width#-int(resized_width_s/2)
    
    return background_copy, keypoint_x, keypoint_y # for keypoints






