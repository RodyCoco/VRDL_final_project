# VRDL_final: The Nature Conservancy Fisheries Monitoring

## Requirements

To install requirements:

```setup
pip install -r ObjectDetector/requirements.txt
pip install -r ViT/requirements.txt
pip install -r classifier/requirements.txt
```

## Hardware

RTX A5000 *4

## Training

To train the model, see following:

```
detector(faster-RCNN): see ObjectDetector/README.md
classifier(Vit_large_patch16_384): run 「python ViT/main.py」
```

## Reproduce performance
To reproduce performance:

```
1. get Model Weight from link and download 「test_stg1」 and 「test_stg2」from Kaggle
2. put *.pkl, annotation and images as following
${ROOT}
  +- ObjectDetector
  +- classifier
  +- ViT
  |  +- Vit_large_patch16_384_bbox_1.pkl
  |  +- Vit_large_patch16_384.pkl
  +- test_stg1
  +- test_stg2
  +- faster_rcnn
3. run 「cd ViT」
4. run 「python test.py」
5. run 「python test_bbox.py」
6. run 「ensemble.py」
7. get the final result 「ensemble.csv」 under the folder ViT
```
## Model Weight and annotation Link
https://drive.google.com/drive/folders/15G6Dwk4HJVyCQWI4RThBOrx71EkfuQ8V?usp=sharing
