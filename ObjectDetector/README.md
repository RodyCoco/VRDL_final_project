# VRDL_final_project

### Description

Object detector is used to generate bounding box for testing images in the testing phase.

The ObjectDetector folder include one folder:
+ ```src```  
  + ```make_json_2classes.py``` : make train.json and test.json
  + ```my-faster_rcnn-rxt101-configs_checkpoints_2classes.py``` : config file of faster rcnn used in the project 

## Requirements

In training and testing object detector, mmdetection was used. Please refer to [mmdetection](https://github.com/open-mmlab/mmdetection.git) for installation.

To install other requirements:

```setup
pip install -r requirements.txt
```
## Dataset Preparation
#### Prepare annotations
+ Run ```src/make_json_2classes.py``` to make train.json and test.json

#### Project structure
```
mmdetection 
└─── coco
│    │
│    └─── annotations
│    │    │  train.json
│    │    |  test.json
│    │
│    └─── images
│         │  1.png
│         |  2.png
│         |  ...
│
└─── my_configs
      │
      └─── my-faster_rcnn-rxt101-configs_checkpoints_2classes.py
└─── configs
└─── ...
```


## Training

To train the model, run this command:

```train
cd mmdetection
python tools/train.py my-configs/my-faster_rcnn-rxt101-configs_checkpoints_2classes.py
```

## Testing

To test the trained model, run:

```test
python tools/test.py my-configs/my-faster_rcnn-rxt101-configs_checkpoints_2classes.py  ${CHECKPOINT_FILE} --format-only --options jsonfile_prefix=${JSONFILE_PREFIX}
```

## Model Weight Link

https://drive.google.com/drive/folders/15G6Dwk4HJVyCQWI4RThBOrx71EkfuQ8V
