# Surgical-DeSAM

## Usage
* First clone the repository locally: `git clone https://github.com/YuyangSheng/Surgical-DeSAM.git`
* Then download EndoVis 17 dataset first and put it under `DATASET_NAME` folder. (If you want to further change the dataset class, please check `datasets/coco.py`.)
* Download well-trained [DETR checkpoint](https://drive.google.com/file/d/1RuqI5cjOgLdKhzQxPOJmlCP0PxsXtxde/view?usp=sharing) and put it under `weights` folder (the pre-trained weights for object detection).
* Please create a `outputs` folder. The training log, evaluation results and visualization will be generated to this folder.

## Training
We trained our model on single GPU with batch_size=1 and leverage pre-trained DETR weights as preliminaries. Please note that in our experiment, batch_size can only be 1.

`python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --endovis_path DATASET_PATH --detr_weights ./weights/swin_detr_1024.pth`

The specific training, evaluation and visualization processes can be seen in `engine_instance_seg.py`.

## Evaluation
Please download our pre-trained [Surgical-DeSAM checkpoint](https://drive.google.com/file/d/1qKFfHgJFO9E35ARsnoUWAmMSG8EP52Cn/view?usp=sharing) first and put it under `weights` folder.

`python main.py --no_aux_loss --eval --resume './weights/surgical_desam_1024.pth' --endovis_path DATASET_PATH`


