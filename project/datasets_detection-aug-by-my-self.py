import copy
import logging
import os
import torch.utils.data
import torchvision
from PIL import Image
from random import sample

from openpifpaf import transforms
from openpifpaf import utils
from openpifpaf.datasets import collate_images_targets_meta


from image_augmentation import pasteImg #added by cuimingbo

import numpy as np
from data import COCO_LABELS # added by cmb

ANNOTATIONS_TRAIN = '/data/data-mscoco/annotations/instances_train2017.json'
ANNOTATIONS_VAL = '/data/data-mscoco/annotations/instances_val2017.json'
IMAGE_DIR_TRAIN = '/data/data-mscoco/images/train2017'
IMAGE_DIR_VAL = '/data/data-mscoco/images/val2017'


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
        
        # got category names and corresponding cat_ids
        catNms = ['car']
        self.cat_ids = self.coco.getCatIds(catNms=catNms)
        #print('category length'+str(self.cat_ids))
        
        #reverse cocolabels to (name:id)
        self.reverse_dict = {value:key for key, value in COCO_LABELS.items()}
        
        # new_cat_id is the id corresponding to the choosen category name
        self.new_cat_id = self.reverse_dict[catNms[0]]
        #print('new category length'+str(self.new_cat_id))
        
        # self.dict is used to map original cat_id to our cat_id
        #self.dict = dict(zip(self.cat_ids, self.new_cat_id))
        self.dict = {self.cat_ids[0]:self.new_cat_id}
        #print(type(self.dict))

        self.ids = self.coco.getImgIds(catIds =self.cat_ids)
        
        
        # set of all image_ids of coco dataset
        self.all_ids_set = set(self.coco.getImgIds())
        # set of all choosen image ids
        self.ids_set = set(self.ids)
        # image_ids which does not contain the object which I choose
        self.other_obejct_id = list(self.all_ids_set - self.ids_set)
        
        # choose some images which do not contain the object
        num_images = int(0.8 * len(self.ids))
        self.other_id = sample(self.other_obejct_id, num_images)
        #merge two lists of ids
        self.ids = self.ids + self.other_id
        
        

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
        #get all annotation_id exsisted in this image
        ann_ids = self.coco.getAnnIds(imgIds=image_id)

        # get all annaotation which fulfil the filter conditions
        anns = self.coco.loadAnns(ann_ids)
        
        image_info = self.coco.loadImgs(image_id)[0]
        source = Image.open("test/car.jpg")
        self.log.debug(image_info)
        with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
            image = Image.open(f).convert('RGB')
            image, keypoint_x, keypoint_y = pasteImg(source, image)
        

        for i in range(len(anns)):
                
            # self.cat_ids now stores only one category_id, which is corresponding to the category we chose before,
            #if the category_id in this annotations is not the category we choose, we assign it to background
            anns[i]['keypoints'] = [0]*3
            
#             if self.cat_ids[0] != anns[i]['category_id']:
#                 index_for_keypoints = 0
#             if self.cat_ids[0] == anns[i]['category_id']:
#                 index_for_keypoints = 1
            
#             #if index_for_keypoints = 0, means that it is not the object we want to detect.
#             if index_for_keypoints == 0:
#                 #overwrite the keypoints
#                 anns[i]['keypoints'] = [0]*3
#             else:
#                 anns[i]['keypoints'] = [0]*3
#                 keypoint_x, keypoint_y, keypoint_v = self.keypoints(anns[i])
#                 index_for_keypoint_x = 0
#                 index_for_keypoint_y = 1
#                 index_for_keypoint_v = 2
#                 anns[i]['keypoints'][index_for_keypoint_x] = keypoint_x
#                 anns[i]['keypoints'][index_for_keypoint_y] = keypoint_y
#                 anns[i]['keypoints'][index_for_keypoint_v] = keypoint_v
        ann_additioanl = anns[len(anns)-1]
        ann_additional['keypoints'] = [keypint_x,keypoint_y,2]
        anns = anns + ann_additional
                                     
        anns = copy.deepcopy(anns)
        #image_info = self.coco.loadImgs(image_id)[0]
        
        # the figure I will use in the final detection competition
        #source = Image.open("test/car.jpg")
        #self.log.debug(image_info)
        #with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
            #image = Image.open(f).convert('RGB')
            #image, keypoint_x, keypoint_y = pasteImg(source, image)


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
        keypoint_x = x + height / 2
        keypoint_y = y + width / 2
        keypoint_v = 2
        #print('bounding box:' + str(ann['bbox']))
        #print('keypoint:' + str(keypoint_x)+' and '+str(keypoint_y))
        
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
    print(len(train_loader))
    
# np.random.seed(100)
# train_loader = torch.utils.data.DataLoader(
#         torch.utils.data.Subset(train_data, np.random.choice(len(train_data),
# 20000)), batch_size=args.batch_size, shuffle=not args.debug,
#         pin_memory=args.pin_memory, num_workers=args.loader_workers,
# drop_last=True,
#         collate_fn=collate_images_targets_meta)

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
