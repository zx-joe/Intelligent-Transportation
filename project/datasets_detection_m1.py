import copy
import logging
import os
import torch.utils.data
import torchvision
from PIL import Image

from openpifpaf import transforms
from openpifpaf import utils
from openpifpaf.datasets import collate_images_targets_meta

import numpy as np
from data import COCO_LABELS #added by cmb

ANNOTATIONS_TRAIN = '/data/data-mscoco/annotations/instances_train2017.json'
ANNOTATIONS_VAL = '/data/data-mscoco/annotations/instances_val2017.json'
IMAGE_DIR_TRAIN = '/data/data-mscoco/images/train2017'
IMAGE_DIR_VAL = '/data/data-mscoco/images/val2017'

################################################################################
# TODO:                                                                        #
# - Create dataset class modeled after CocoKeypoints in the official           #
#   OpenPifPaf repo                                                            #
# - Modify to take all categories of COCO (CocoKeypoints uses only the human   #
#   category)                                                                  #
# - Using the bounding box and class labels, create a new ground-truth         #
#   annotation that can be used for detection                                  #
#   (using a single keypoint per class, being the center of the bounding box)  #
#                                                                              #
# Hint: Use the OpenPifPaf repo for reference                                  #
#                                                                              #
################################################################################
class CocoKeypoints(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Based on `torchvision.dataset.CocoDetection`.
    Caches preprocessing.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, image_transform=None, target_transforms=None,
                 n_images=None, preprocess=None, all_images=False, all_persons=False):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        #print(self.root)
        # got all 80 category names and corresponding cat_ids
        catNms = list(COCO_LABELS.values())[1:] 
        self.cat_ids = self.coco.getCatIds(catNms=catNms)
        
        # self.dict is used to map original cat_id to our cat_id
        new_cat_id = [i+1 for i in range(len(catNms))]
        self.dict = dict(zip(self.cat_ids, new_cat_id))
        
        self.ids = self.coco.getImgIds()

        #self.ids = []
        #for i in range(len(self.cat_ids)):
            #self.ids.append(self.coco.getImgIds(catIds = self.cat_ids[i]))
        
#         if all_images:
#             self.ids = self.coco.getImgIds(catIds =self.cat_ids)
#             #print(self.ids)

#         elif all_persons:
#             self.ids = self.coco.getImgIds(catIds=self.cat_ids)
#         else:
#             self.ids = self.coco.getImgIds(catIds=self.cat_ids)
#             #self.filter_for_keypoint_annotations()
            
        if n_images:
            self.ids = self.ids[:n_images]
            
        print('Images: {}'.format(len(self.ids)))

        self.preprocess = preprocess or transforms.Normalize()
        self.image_transform = image_transform or transforms.image_transform
        self.target_transforms = target_transforms

        self.log = logging.getLogger(self.__class__.__name__)

    def filter_for_keypoint_annotations(self):
        print('filter for keypoint annotations ...')
        def has_keypoint_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_keypoint_annotation(image_id)]
        print('... done.')    
        


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        #print("image_id"+len(image_id))
        #print("ann_ids"+len(ann_ids))
        #ann_idss = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
        #print(len(ann_idss))
        
        anns = self.coco.loadAnns(ann_ids)
                
        for i in range(len(anns)):
            index_new = self.dict[anns[i]['category_id']]
            anns[i]['keypoints'] = [0]*240
            keypoint_x, keypoint_y, keypoint_v = self.keypoints(anns[i])
            index_for_keypoint_x = 3*(index_new-1)
            index_for_keypoint_y = index_for_keypoint_x + 1
            index_for_keypoint_v = index_for_keypoint_y + 1
            anns[i]['keypoints'][index_for_keypoint_x] = keypoint_x
            anns[i]['keypoints'][index_for_keypoint_y] = keypoint_y
            anns[i]['keypoints'][index_for_keypoint_v] = keypoint_v            
            
            
            
            # to overwrite the keypoints of 'person' category
#             if anns[i]['category_id'] == 1 and ('keypoints' in anns[i]):
#                 if len(anns[i]['keypoints']) != 240:
#                     anns[i]['keypoints'] = [0]*240
                
#             # to map the original category_id to our id, e.g. 90->80
#             index_new = self.dict[anns[i]['category_id']]
            
#             # if there is no 'keypoints' in the dict, initialize one and write
#             if 'keypoints' not in anns[i]:
                
#                 # initialize a list with length of 240, e.g. 3x80
#                 anns[i]['keypoints'] = [0]*240
                
#                 # calculate the keypoint from the bounding box
#                 keypoint_x, keypoint_y, keypoint_v = self.keypoints(anns[i])
                
#                 # corresponding index in the keypoints list
#                 index_for_keypoint_x = 3*(index_new-1)
#                 index_for_keypoint_y = index_for_keypoint_x + 1
#                 index_for_keypoint_v = index_for_keypoint_y + 1
                
#                 # write center point of bounding box and visibility to keypoints
#                 anns[i]['keypoints'][index_for_keypoint_x] = keypoint_x
#                 anns[i]['keypoints'][index_for_keypoint_y] = keypoint_y
#                 anns[i]['keypoints'][index_for_keypoint_v] = keypoint_v
                                
#             # if there is a 'keypoints' in the dict, just write keypoint information to the list
#             else:
#                 keypoint_x, keypoint_y, keypoint_v = self.keypoints(anns[i])
#                 index_for_keypoint_x = 3*(index_new-1)
#                 index_for_keypoint_y = index_for_keypoint_x + 1
#                 index_for_keypoint_v = index_for_keypoint_y + 1
#                 anns[i]['keypoints'][index_for_keypoint_x] = keypoint_x
#                 anns[i]['keypoints'][index_for_keypoint_y] = keypoint_y
#                 anns[i]['keypoints'][index_for_keypoint_v] = keypoint_v
            
        anns = copy.deepcopy(anns)
        image_info = self.coco.loadImgs(image_id)[0]
        
        #print(image_info)
        
        self.log.debug(image_info)
        with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
            image = Image.open(f).convert('RGB')

        meta = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_info['file_name'],
        }

        if 'flickr_url' in image_info:
            _, flickr_file_name = image_info['flickr_url'].rsplit('/', maxsplit=1)
            flickr_id, _ = flickr_file_name.split('_', maxsplit=1)
            meta['flickr_full_page'] = 'http://flickr.com/photo.gne?id={}'.format(flickr_id)

        # preprocess image and annotations
        image, anns, preprocess_meta = self.preprocess(image, anns)
        meta.update(preprocess_meta)

        # transform image
        original_size = image.size
        image = self.image_transform(image)
        assert image.size(2) == original_size[0]
        assert image.size(1) == original_size[1]

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_image(image, valid_area)

        # if there are not target transforms, done here
        self.log.debug(meta)
        if self.target_transforms is None:
            return image, anns, meta

        # transform targets
        targets = [t(anns, original_size) for t in self.target_transforms]
        return image, targets, meta
    
    def keypoints(self, ann):
        
        x = ann['bbox'][0]
        y = ann['bbox'][1]
        width = ann['bbox'][2]
        height = ann['bbox'][3]
        keypoint_x = x + width / 2
        keypoint_y = y + height / 2
        keypoint_v = 2
        return keypoint_x, keypoint_y, keypoint_v

    def __len__(self):
        return len(self.ids)
    

################################################################################
#                              END OF YOUR CODE                                #
################################################################################


def train_cli(parser):
    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--train-annotations', default=ANNOTATIONS_TRAIN)
    group.add_argument('--train-image-dir', default=IMAGE_DIR_TRAIN)
    group.add_argument('--val-annotations', default=ANNOTATIONS_VAL)
    group.add_argument('--val-image-dir', default=IMAGE_DIR_VAL)
    group.add_argument('--pre-n-images', default=8000, type=int,
                       help='number of images to sampe for pretraining')
    group.add_argument('--n-images', default=None, type=int,
                       help='number of images to sampe')
    group.add_argument('--loader-workers', default=2, type=int,
                       help='number of workers for data loading')
    group.add_argument('--batch-size', default=8, type=int,
                       help='batch size')


def train_factory(args, preprocess, target_transforms):
       
    train_data = CocoKeypoints(
        root=args.train_image_dir,
        annFile=args.train_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.n_images,
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=not args.debug,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    
    
    np.random.seed(100)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_data, np.random.choice(len(train_data), 20000)), 
                                               batch_size=args.batch_size, shuffle=not args.debug,pin_memory=args.pin_memory, num_workers=args.loader_workers,
                                               drop_last=True, collate_fn=collate_images_targets_meta)
            
    val_data = CocoKeypoints(
        root=args.val_image_dir,
        annFile=args.val_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.n_images,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    pre_train_data = CocoKeypoints(
        root=args.train_image_dir,
        annFile=args.train_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.pre_n_images,
    )
    pre_train_loader = torch.utils.data.DataLoader(
        pre_train_data, batch_size=args.batch_size, shuffle=True,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    return train_loader, val_loader, pre_train_loader
