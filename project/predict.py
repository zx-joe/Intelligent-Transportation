"""Predict poses for given images."""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import cv2

import numpy as np
import torch

from openpifpaf.network import nets
from openpifpaf import decoder, transforms

import openpifpaf.datasets as datasets
import show
from data import COCO_LABELS

import torchvision
def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    nets.cli(parser)
    decoder.cli(parser, force_complete_pose=False, instance_threshold=0.05)
    parser.add_argument('images', nargs='*',
                        help='input images')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('-o', '--output-directory',
                        help=('Output directory. When using this option, make '
                              'sure input images have distinct file names.'))
    parser.add_argument('--show', default=False, action='store_true',
                        help='show image of output overlay')
    parser.add_argument('--output-types', nargs='+', default=['skeleton', 'json'],
                        help='what to output: skeleton, keypoints, json')
    parser.add_argument('--loader-workers', default=2, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--figure-width', default=10.0, type=float,
                        help='figure width')
    parser.add_argument('--dpi-factor', default=1.0, type=float,
                        help='increase dpi of output image by this factor')
    args = parser.parse_args()

    # glob
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    return args

def plotCenter(img,keypoints):
    point_color = (0, 0, 255)
    img = np.array(img)*255
    #print(img)
    #cv2.imwrite('Mytest2.jpg',img1)
    for keypoint in keypoints:
        #print(keypoint[0][0])
        #print(keypoint[0][1])
        img = cv2.circle(img, (int(keypoint[0][0]), int(keypoint[0][1])), 5, point_color, -1)
        
    return img
    



def main():
    args = cli()

    # load model
    model, _ = nets.factory_from_args(args)
    model = model.to(args.device)
    processor = decoder.factory_from_args(args, model)

    # data
    data = datasets.ImageList(args.images)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers)

    # visualizers
    skeleton_painter = show.InstancePainter(show_box=True, color_connections=True,
                                            markersize=1, linewidth=6)

    for image_i, (image_paths, image_tensors, processed_images_cpu) in enumerate(data_loader):
        images = image_tensors.permute(0, 2, 3, 1)
        img_show = np.squeeze(images)
        print(image_paths)
        #print(img_show)
        #plt.imshow(img_show)
        processed_images = processed_images_cpu.to(args.device, non_blocking=True)
        fields_batch = processor.fields(processed_images)
        # unbatch
        for image_path, image, processed_image_cpu, fields in zip(
                image_paths,
                images,
                processed_images_cpu,
                fields_batch):

            if args.output_directory is None:
                output_path = image_path
            else:
                file_name = os.path.basename(image_path)
                output_path = os.path.join(args.output_directory, file_name)
            print('image', image_i, image_path, output_path)

            processor.set_cpu_image(image, processed_image_cpu)
            keypoint_sets, scores = processor.keypoint_sets(fields)
            
            img = plotCenter(img_show,keypoint_sets)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_path + '.center.png',img)

            if 'json' in args.output_types:
                with open(output_path + '.pifpaf.json', 'w') as f:
                    json.dump([
                        {'keypoints': np.around(kps, 1).reshape(-1).tolist(),
                         'bbox': [np.min(kps[:, 0]), np.min(kps[:, 1]),
                                  np.max(kps[:, 0]), np.max(kps[:, 1])]}
                        for kps in keypoint_sets
                    ], f)
                    
            

            texts = [COCO_LABELS[np.argmax(kps[:,2])+1] for kps in keypoint_sets]

            #print(texts)

            if 'skeleton' not in args.output_types:
                with show.image_canvas(image,
                                       output_path + '.skeleton.png',
                                       show=args.show,
                                       fig_width=args.figure_width,
                                       dpi_factor=args.dpi_factor) as ax:
                    skeleton_painter.keypoints(ax, keypoint_sets, scores=scores,texts=texts)


if __name__ == '__main__':
    main()
