# Surgical-DeSAM
This is the official repository of Surgical-DeSAM, which is proposed with an innovative architecture based on transformer, integrating a detection baseline (DETR) and part of a segmentation foundation model (SAM). By decoupling and reengineering these two baselines, our proposed model inherits both capabilities and demonstrates commendable performance on our surgical instrument dataset.

## Usage
* First clone the repository locally: `git clone https://github.com/YuyangSheng/Surgical-DeSAM.git`
* Then download the EndoVis 17 dataset first and put it under `DATASET_NAME` folder. (If you want to further change the dataset class, please check `datasets/coco.py`.)
* Download well-trained [DETR checkpoint](https://drive.google.com/file/d/1RuqI5cjOgLdKhzQxPOJmlCP0PxsXtxde/view?usp=sharing) and put it under `weights` folder (the pre-trained weights for object detection).
* Please create a `outputs` folder. The training log, evaluation results and visualization will be generated to this folder.

## Training detection baseline
If you want to train the detection model from scratch, please follow the command:

`python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --endovis_path DATASET_PATH --sam False`

Then, replace `--detr_weights` parameter with the path of the best checkpoint from the output if you want to train Surgical-DeSAM. 

## Training Surgical-DeSAM
We trained our model on a single GPU with batch_size=1 and leveraged pre-trained DETR weights as preliminaries. Please note that in our experiment, batch_size can only be 1.

`python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --endovis_path DATASET_PATH --detr_weights ./weights/swin_detr_1024.pth`

The specific training, evaluation, and visualization processes can be seen in `engine_instance_seg.py`.

## Evaluation
Please download our pre-trained [Surgical-DeSAM checkpoint](https://drive.google.com/file/d/1qKFfHgJFO9E35ARsnoUWAmMSG8EP52Cn/view?usp=sharing) first and put it under `weights` folder.

`python main.py --no_aux_loss --eval --resume './weights/surgical_desam_1024.pth' --endovis_path DATASET_PATH`

## Results
The instance segmentation results are shown as below:
<div align='center'>
<img src='https://github.com/YuyangSheng/Surgical-DeSAM/blob/main/assets/instance_seg_res.jpg' width=550>
</div>
