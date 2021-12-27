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

```train
detector(faster-RCNN): see ObjectDetector/README.md
classifier: run 「python ViT/main.py」
```

## Reproduce performance
To evaluate my model, run:

```eval
python inference.py
```
## Model Weight Link
https://drive.google.com/drive/folders/1qMi1xaPWkpJr7vNadyh4fJ_niqN22O92?usp=sharing
