
import os
import math
import torch
import torchvision
import torchvision.transforms as transforms

import torchvision.datasets.voc as VOC


classes = ['person' , 'bird', 'cat', 'cow',
           'dog', 'horse', 'sheep', 'aeroplane',
           'bicycle', 'boat', 'bus', 'car',
           'motorbike', 'train', 'bottle', 'chair',
           'diningtable', 'pottedplant', 'sofa', 'tvmonitor']


def preprocess_target(target: dict):
    """

    This applies several transformations to the target. None of the fields are removed
    we only make conversions.

    occluded, difficult, truncated:   str -> bool
    bndbox['xmax'], bnbox['xmin'], bnbbox['..'] :  str -> int

    object : this is always converted to a list

    """
    output_target = target.copy()

    output_target['annotation']['segmented'] = bool(output_target['annotation']['segmented'])
    for k, v in output_target['annotation']['size'].items():
        output_target['annotation']['size'][k] = int(v)


    if type(output_target['annotation']['object']) is not list:
        # If this is a not a list, it contains a single object
        # that we put in a list
        output_target['annotation']['object'] = [output_target['annotation']['object']]

    objects = output_target['annotation']['object']

    for o in objects:
        for k, v in o['bndbox'].items():
            o['bndbox'][k] = int(v)
        for k in ['occluded', 'difficult', 'truncated']:
            o[k] = bool(int(o[k]))

    return output_target


def extract_class_and_bndbox(target: dict,
                             image_transform_params: dict):

    """
    target['annotation'] = {'filename': '2008_000008.jpg',
                            'folder': 'VOC2012',
                            'object': [{'bndbox': {'xmax': 471,
                                                   'xmin': 53,
                                                   'ymax': 420,
                                                   'ymin': 87},
                                        'difficult': False,
                                        'name': 'horse',
                                        'occluded': True,

                                        'pose': 'Left',
                                        'truncated': False},
                                       {'bndbox': {'xmax': 289,
                                                   'xmin': 158,
                                                   'ymax': 167,
                                                   'ymin': 44},
                                        'difficult': False,
                                        'name': 'person',
                                        'occluded': False,
                                        'pose': 'Unspecified',
                                        'truncated': True}],
                            'segmented': False,
                            'size': {'depth': 3, 'height': 442, 'width': 500},
                            'source': {'annotation': 'PASCAL VOC2008',
                                       'database': 'The VOC2008 Database',
                                       'image': 'flickr'}}

    example output :
        [{'bndbox': {'xmax': , 'xmin': , 'ymax':, 'ymin': }, 'class':5}, {'bndbox': {..}, 'class': ...}]
    """

    return [{'bndbox': transform_bbox(o['bndbox'],
                                      {'width' : target['annotation']['size']['width'],
                                       'height': target['annotation']['size']['height']},
                                      image_transform_params),
             'class': classes.index(o['name'])}
             for o in target['annotation']['object']]

def transform_bbox(bbox: dict, input_image_size: dict, image_transform_params:dict):
    """
        bbox : {'xmin': int, 'xmax': int, 'ymax': int, 'ymin': int}
        input_image_size : {'width': int, 'height': int}
        resize_image : one of ['none', 'shrink', 'crop']
        output_image_size : {'width': int, 'height': int}
    """

    # The encoding is the center of the bounding box
    #  and its width/height.
    # All these coordinates are relative to the output_image_size
    out_bbox = {"cx": 0.0, "cy": 0.0, "width": 0.0, "height": 0.0}

    image_mode = image_transform_params['image_mode']
    if image_mode == 'none':
        out_bbox["cx"] = 0.5 * (bbox['xmin'] + bbox['xmax']) / input_image_size['width']
        out_bbox["cy"] = 0.5 * (bbox['ymin'] + bbox['ymax']) / input_image_size['height']
        out_bbox["width"] = float(bbox["xmax"] - bbox["xmin"]) / input_image_size["width"]
        out_bbox["height"] = float(bbox["ymax"] - bbox["ymin"]) / input_image_size["height"]

    elif(image_mode == 'shrink'):
        output_image_size = image_transform_params['output_image_size']
        scale_width  = float(output_image_size['width']) / input_image_size['width']
        scale_height = float(output_image_size['height']) / input_image_size['height']
        out_bbox["cx"]     = scale_width * 0.5 * (bbox['xmin'] + bbox['xmax']) / output_image_size['width']
        out_bbox["cy"]     = scale_height * 0.5 * (bbox['ymin'] + bbox['ymax']) / output_image_size['height']
        out_bbox["width"]  = scale_width * float(bbox["xmax"] - bbox["xmin"]) / output_image_size["width"]
        out_bbox["height"] = scale_height * float(bbox["ymax"] - bbox["ymin"]) / output_image_size["height"]

    elif(image_mode == 'crop'):
        output_image_size = image_transform_params['output_image_size']
        offset_width  = int(round((input_image_size['width'] - output_image_size['width']) / 2.))
        offset_height = int(round((input_image_size['height'] - output_image_size['height']) / 2.))

        cropped_bbox = {"xmin": 0.0, "xmax": 0.0, "ymin": 0.0, "ymax": 0.0}
        for sfx in ['min', 'max']:
            cropped_bbox['x%s'%sfx] = min(max(bbox['x%s'%sfx] - offset_width, 0), output_image_size['width'])
            cropped_bbox['y%s'%sfx] = min(max(bbox['y%s'%sfx] - offset_height, 0), output_image_size['height'])
        out_bbox["cx"] = 0.5 * (cropped_bbox['xmin'] + cropped_bbox['xmax']) / output_image_size['width']
        out_bbox["cy"] = 0.5 * (cropped_bbox['ymin'] + cropped_bbox['ymax']) / output_image_size['height']
        out_bbox["width"] = float(cropped_bbox["xmax"] -  cropped_bbox["xmin"]) / output_image_size["width"]
        out_bbox["height"] = float(cropped_bbox["ymax"] - cropped_bbox["ymin"]) / output_image_size["height"]

    else:
        raise ValueError('invalid image_mode for transform_bbox, got "{}"'.format(image_mode))
    return out_bbox


def filter_largest(objects: list):
    """
    This builds and return a function which acts the way depicted below.
    output_image_size and mode is used to adapt the bounding box coordinates

    example input :

    objects : [{'bndbox': {'cx': ..., 'cy': ....,
                           width': ..., 'height': ...},
                'class': 5},
               {'bndbox': {'cx': ..., 'cy': ....,
                          'width': ..., 'height': ...},
                'class': 4},
               ...]


    example output (a single object) :

    {'bndbox': {}, 'class': 5}

    """

    #####    #####
    # TO BE DONE #
    #vvvvvvvvvvvv#

    # return ...

    #^^^^^^^^^^^^#
    #####    #####

def target_to_tensor(obj: dict):
    """
    Input :
        obj :{'bndbox': {}, 'class': 5}

    Output : two tensors,
                -  the first with [cx, cy, width, height]
                -  the second with [class]

    """
    #####    #####
    # TO BE DONE #
    #vvvvvvvvvvvv#

    #return {'bboxes': ...,
    #        'labels': ...

    #^^^^^^^^^^^^#
    #####    #####


def cell_idx_of_center(coordinates, num_cells: int):
    return math.floor(coordinates[0] * num_cells), math.floor(coordinates[1] * num_cells)

def targets_to_grid_cell_tensor(objects: list, num_cells: int):
    """
    This builds and return a function which acts the way depicted below.
    output_image_size and mode is used to adapt the bounding box coordinates

    Example:
    Input
    objects : [{'bndbox': {}, 'class': 5}, {'bndbox': {}, 'class': 4}, ...]
    num_cells : 6

    Output :  three tensors
        'bboxes' : (6, 6, 4)   with (cx, cy, width ,height)
        'has_obj': (6, 6)   whether the cell (i,j) contains the center of an object
        'labels' : (6, 6)   labels

    Every cell is affected at most one object; If multiple objects share the same cell
    only one is preserved
    """
    bboxes  = torch.zeros((4, num_cells, num_cells),dtype=torch.float)
    has_obj = torch.zeros((num_cells, num_cells), dtype=torch.int)
    labels  = torch.zeros((num_cells, num_cells),dtype=torch.int)
    for ko, o in enumerate(objects):
        bndbox = o['bndbox']
        cx, cy, width, height = bndbox['cx'], bndbox['cy'], bndbox['width'], bndbox['height']
        cj, ci = cell_idx_of_center((cx, cy), num_cells)
        #####    #####
        # TO BE DONE #
        #vvvvvvvvvvvv#

        #bboxes[:, ci, cj] =
        #has_obj[ci, cj] =
        #labels[ci, cj] =

        #^^^^^^^^^^^^#
        #####    #####
    return {'bboxes': bboxes, 'has_obj': has_obj, 'labels': labels}

def check_key(d, key, valid_values):
    if not key in d:
        raise KeyError('Missing key {} in dictionnary {}'.format(key, d))
    if not d[key] in valid_values:
        raise ValueError("Key {}: got \"{}\" , expected one of {}".format(key, d[key], valid_values))

def validate_image_transform_params(image_transform_params: dict):
    """
    {'image_mode'='none'}
    {'image_mode'='shrink', output_image_size={'width':.., 'height': ..}}
    {'image_mode'='crop'  , output_image_size={'width':.., 'height': ..}}
    """
    check_key(image_transform_params, 'image_mode', ['none', 'shrink', 'crop'])

    if(image_transform_params['image_mode'] == 'none'):
        return
    else:
        assert('output_image_size' in image_transform_params)
        assert(type(image_transform_params['output_image_size']) is dict)
        assert('width' in image_transform_params['output_image_size'])
        assert('height' in image_transform_params['output_image_size'])

def make_image_transform(image_transform_params: dict,
                         transform: object):
    """
    image_transform_params :
        {'image_mode'='none'}
        {'image_mode'='shrink', output_image_size={'width':.., 'height': ..}}
        {'image_mode'='crop'  , output_image_size={'width':.., 'height': ..}}
    transform : a torchvision.transforms type of object
    """
    validate_image_transform_params(image_transform_params)

    resize_image = image_transform_params['image_mode']
    if resize_image == 'none':
        preprocess_image = None
    elif resize_image == 'shrink':
        preprocess_image = transforms.Resize((image_transform_params['output_image_size']['width'],
                                              image_transform_params['output_image_size']['height']))
    elif resize_image == 'crop':
        preprocess_image = transforms.CenterCrop((image_transform_params['output_image_size']['width'],
                                                  image_transform_params['output_image_size']['height']))

    if preprocess_image is not None:
        if transform is not None:
            image_transform = transforms.Compose([preprocess_image, transform])
        else:
            image_transform = preprocess_image
    else:
        image_transform = transform

    return image_transform


def validate_target_transforms_params(target_transform_params: dict):
    """
    {'target_mode'='orig'}
    {'target_mode'='preprocessed'}
    {'target_mode'='largest_bbox', 'image_transform_params': dict}
    {'target_mode'='all_bbox'    , 'image_transform_params': dict, 'num_cells': int}
    """
    check_key(target_transform_params, 'target_mode', ['orig', 'preprocessed', 'largest_bbox', 'all_bbox'])

    if(target_transform_params['target_mode'] in ['orig', 'preprocessed']):
        return
    else:
        assert('image_transform_params' in target_transform_params)
        assert(type(target_transform_params['image_transform_params']) is dict)
        validate_image_transform_params(target_transform_params['image_transform_params'])
        if(target_transform_params['target_mode'] == 'all_bbox'):
            assert('num_cells' in target_transform_params)


def make_target_transform(target_transform_params: dict):
    """
        target_mode :
            orig          : keeps the original unaltered targets
            preprocessed  : perform some preprocessing on the targets, see data.preprocess_target
            all_bbox      : keeps all the bounding boxes and convert them into "grid cell" tensors
            largest_bbox  : outputs a tensor with the largest bbox (4 numbers)
        see also help(validate_target_transforms_params)
    """
    validate_target_transforms_params(target_transform_params)

    target_mode = target_transform_params['target_mode']
    if target_mode == 'orig':
        return None
    elif target_mode == 'preprocessed':
        t_transform = lambda target: preprocess_target(target)
    else:
        image_transform_params = target_transform_params['image_transform_params']
        get_bbox = lambda target: extract_class_and_bndbox(preprocess_target(target), image_transform_params)
        if target_mode == 'largest_bbox':
            t_transform = lambda target: target_to_tensor(filter_largest(get_bbox(target)))
        else:
            t_transform = lambda target: targets_to_grid_cell_tensor(get_bbox(target), target_transform_params['num_cells'])
    return t_transform



def make_trainval_dataset(dataset_dir: str,
                          image_transform_params: dict,
                          transform: object,
                          target_transform_params: dict,
                          download: bool):
    if not dataset_dir:
        dataset_dir = os.path.join(os.path.expanduser("~"), 'Datasets', 'PascalVOC')

    image_transform  = make_image_transform(image_transform_params, transform)
    target_transform = make_target_transform(target_transform_params)

    dataset_train = VOC.VOCDetection(root=dataset_dir, image_set='train',
                                     transform = image_transform,
                                     target_transform = target_transform,
                                     download=download)
    dataset_val   = VOC.VOCDetection(root=dataset_dir, image_set='val'  ,
                                     transform = image_transform,
                                     target_transform = target_transform,
                                     download=download)


    return dataset_train, dataset_val




def make_test_dataset(dataset_dir: str,
        resize_image: str,
        transform: object,
        target_mode: str,
        download: bool,
        output_image_size=None):
    pass

#def objects_collate(batch):
#    labels = []
#    bboxes = []
#    imgs = []
#    for sample in batch:
#        imgs.append(sample[0])
#        bboxes.append(sample[1]['bboxes'])
#        labels.append(sample[1]['classes'])
#
#    return torch.stack(imgs, 0), bboxes, labels
#
#if __name__ == '__main__':
#    train, val = make_trainval_dataset(dataset_dir=None,
#                          resize_image='shrink',
#                          transform = None,
#                          target_mode='bbox',
#                          download = False, output_image_size={'width':250, 'height':250})
#
#    img, target = train[0]
#    print(target)
#
