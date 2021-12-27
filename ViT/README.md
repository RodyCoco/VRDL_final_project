# VRDL_final_project

## Training
Before training, put 「train」 images folder like the following:
```
${ROOT}
  +- ObjectDetector
  +- classifier
  +- ViT
  |  +- Vit_large_patch16_384_bbox_1.pkl
  |  +- Vit_large_patch16_384.pkl
  +- train
```
To train the model, run this command:

```train
cd ViT
python main.py
```
