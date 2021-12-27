# Imports
import os
import random
import json
from tqdm import tqdm
import matplotlib.image

# Set path and filename lists to global variables
FILE_NAME_PATH = '../train_validate_split'

ANN_PATH = '../Annotations_train_sep'

IMG_PATH = '../coco/images/train'

TEST_PATH = '../coco/images/test'


TRAIN_FILES = ['ALB_train.txt', 'BET_train.txt', 'DOL_train.txt',
               'LAG_train.txt', 'SHARK_train.txt', 'YFT_train.txt',
               'NoF_train.txt', 'OTHER_train.txt']

VALIDATE_FILES = ['ALB_validate.txt', 'BET_validate.txt', 'DOL_validate.txt',
                  'LAG_validate.txt', 'SHARK_validate.txt', 'YFT_validate.txt',
                  'NoF_validate.txt', 'OTHER_validate.txt']

START_BOUNDING_BOX_ID = 1

# If necessary, pre-define category and its id
PRE_DEFINE_CATEGORIES = {"Fish": 1}


# Get image_id by striping the prefix 'img_' and turn to int
def get_img_id(text, prefix):
    text = text.strip('.jpg')
    return int(text[text.startswith(prefix) and len(prefix):])


# Get images filename from the txt file.
def get_img_filename(filename_list):
    # make a dict with all img_names in the filename list.
    imgfile_dict = {}

    for filename in filename_list:
        img_filenames = []
        filepath = os.path.join(FILE_NAME_PATH, filename)
        file = open(filepath, 'r')
        # Get all img_names in a file.
        for line in file.readlines():
            img_filename = line.strip()
            img_filenames.append(img_filename)
        imgfile_dict[filename] = img_filenames

    return imgfile_dict


# Get image info list of one image
def get_one_img_info(root_dir, img_name):
    try:
        image_id = get_img_id(img_name, 'img_')
    except:
        image_id = get_img_id(img_name, 'image_')

    img_file = img_name.strip('.jpg') + '.jpg'
    path = os.path.join(root_dir, img_file)
    img = matplotlib.image.imread(path)
    height, width = img.shape[0], img.shape[1]
    image = [{
            "file_name": img_file,
            "height": height,
            "width": width,
            "id": image_id,
            }]

    return image


# Get annotation list of one image in annotations file
def get_one_ann(img_name):
    global START_BOUNDING_BOX_ID
    filename = img_name + '.json'
    filepath = os.path.join(ANN_PATH, filename)
    anns = []
    # generate annotation if the file exist
    # (when the class is NoF, the annotation is empty)
    if not os.path.exists(filepath):
        return anns

    ann_file = open(filepath)
    ann_from_file = json.load(ann_file)

    categories = PRE_DEFINE_CATEGORIES

    for ann in list(ann_from_file.values())[0]:
        rect = ann['rect']
        x, y, width, height = rect['x'], rect['y'], rect['w'], rect['h']
        label = ann["class"]
        # If the lable is BAIT, give no annotations (seen as no fish)
        if label == 'BAIT':
            continue

        image_id = get_img_id(img_name, 'img_')
        # Only one classes for detector : fish
        ann_output = {
                        "area": width * height,
                        "iscrowd": 0,
                        "image_id": image_id,
                        "bbox": [x, y, width, height],
                        "category_id": 1,
                        "id": START_BOUNDING_BOX_ID,
                        "ignore": 0,
                        "segmentation": [],
                     }
        START_BOUNDING_BOX_ID += 1
        anns.append(ann_output)
    ann_file.close()

    return anns


def make_json_coco(imgfile_info, json_file_path, is_all_train=True):
    json_dict = {"images": [], "annotations": [], "categories": []}
    img_files = []

    if not is_all_train:
        img_files_list = list(imgfile_info.values())
        for img_file in img_files_list:
            img_files += img_file
    else:
        img_files = imgfile_info

    categories = PRE_DEFINE_CATEGORIES

    # make images and annotations of coco format json file
    for img_file in tqdm(img_files):
        # get one image "image" and write to json_dict
        if not img_file.endswith('.jpg'):
            continue
        image = get_one_img_info(IMG_PATH, img_file)
        json_dict["images"] += image

        # get one image "annotations" and write to json_dict
        # the ann will be empty if the class is "NoF" or "BAIT"
        ann = get_one_ann(img_file.strip('.jpg'))
        json_dict["annotations"] += ann

    # make categories of coco format json file
    for cate, cid in categories.items():
        cat = {"supercategory": "fish", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    # generate .json file.
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    json_fp = open(json_file_path, "w")
    json_str = json.dumps(json_dict, indent=4)
    json_fp.write(json_str)
    json_fp.close()
    print('Successfully generate json file.')


def make_json_coco_test(json_file_path):
    json_dict = {"images": [], "annotations": [], "categories": []}

    img_files = os.listdir(TEST_PATH)

    categories = PRE_DEFINE_CATEGORIES

    # make images and annotations of coco format json file
    for img_file in tqdm(img_files):
        # get one image "image" and write to json_dict
        image = get_one_img_info(TEST_PATH, img_file)
        json_dict["images"] += image

    # make categories of coco format json file
    for cate, cid in categories.items():
        cat = {"supercategory": "fish", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    # generate .json file.
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    json_fp = open(json_file_path, "w")
    json_str = json.dumps(json_dict, indent=4)
    json_fp.write(json_str)
    json_fp.close()
    print('Successfully generate json file.')

# Seperate for train val set
# train_imgfile_dict = get_img_filename(TRAIN_FILES)
# val_imgfile_dict = get_img_filename(VALIDATE_FILES)

# Use all images to train
train_imgfile_dict = os.listdir(IMG_PATH)
val_imgfile_dict = train_imgfile_dict[:500]

# Make train.json, val.json and test.json
make_json_coco(train_imgfile_dict, "../coco/annotations/train.json")
make_json_coco(val_imgfile_dict, "../coco/annotations/val.json")
make_json_coco_test("../coco/annotations/test.json")
