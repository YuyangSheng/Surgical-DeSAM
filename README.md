# Surgical-DeSAM
This is the official repository of Surgical-DeSAM, which is proposed with an innovative architecture based on transformer, integrating a detection baseline (DETR) and part of a segmentation foundation model (SAM). By decoupling and reengineering these two baselines, our proposed model inherits both capabilities and demonstrates commendable performance on our surgical instrument dataset.

## For endovis 2017 dataset
### Usage
* First clone the repository locally: `git clone https://github.com/YuyangSheng/Surgical-DeSAM.git`
* Then download the EndoVis 17 dataset first and put it under `DATASET_NAME` folder. (If you want to further change the dataset class, please check `datasets/coco.py`.)
* Download well-trained [DETR checkpoint](https://drive.google.com/file/d/1RuqI5cjOgLdKhzQxPOJmlCP0PxsXtxde/view?usp=sharing) and put it under `weights` folder (the pre-trained weights for object detection).
* Please create a `outputs` folder. The training log, evaluation results and visualization will be generated to this folder.


### Training detection baseline
If you want to train the detection model from scratch, please follow the command:

`python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --endovis_path DATASET_PATH --sam False`

Then, replace `--detr_weights` parameter with the path of the best checkpoint from the output if you want to train Surgical-DeSAM. 

### Training Surgical-DeSAM
We trained our model on a single GPU with batch_size=1 and leveraged pre-trained DETR weights as preliminaries. Please note that in our experiment, batch_size can only be 1.

`python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --endovis_path DATASET_PATH --detr_weights ./weights/swin_detr_1024.pth`

The specific training, evaluation, and visualization processes can be seen in `engine_instance_seg.py`.

### Evaluation
Please download our pre-trained [Surgical-DeSAM checkpoint](https://drive.google.com/file/d/1ffoeEA8rJGPVUOgMTvVr0k1x7esXm3rn/view?usp=sharing) first and put it under `weights` folder.

`python main.py --no_aux_loss --eval --resume './weights/surgical_desam_1024.pth' --endovis_path DATASET_PATH`

### Results
The instance segmentation results are shown as below:
<div align='center'>
<img src='https://github.com/YuyangSheng/Surgical-DeSAM/blob/main/assets/instance_seg_res.jpg' width=550>
</div>

## For endovis 2018 dataset
### Instructions for training
* First clone the repository locally: `git clone https://github.com/YuyangSheng/Surgical-DeSAM.git`
* Then download the [EndoVis2018 dataset](https://drive.google.com/drive/folders/12kvir0wm1JyzIplOtiM9JszZNJzV65Vw?usp=sharing).
* To train the detection model, please follow the command:

  `python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --endovis_path ./endovis18 --sam False --plot False` (`plot` parameter here is used to determine whether to visualize the results and the visualization function can be found in `'plot_results_gt'` function in `engine_instance_seg.py` file.) Then save the best checkpoint to `PATH_TO_DETR_WEIGHTS`.
* Next, using well-trained detection baseline to further train the Surgical-DeSAM model. **Please note that in our experiment, batch_size can only be 1.**

  Copy the command: `python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --endovis_path ./endovis18 --detr_weights PATH_TO_DETR_WEIGHTS --sam True --plot False`
  (The visualization function can be found in the `'plot_instance_seg'` function in the `engine_instance_seg.py` file.)
  
### Instructions for evaluation
To evaluate the detection model only, please copy the command:

`python main.py --no_aux_loss --eval --resume PATH_TO_DETR_WEIGHTS --sam False`

To evaluate Surgical-DeSAM using pre-trained weights, follow the command:

`python main.py --no_aux_loss --eval --resume PATH_TO_SurgicalDESAM_WEIGHTS --sam True`

(To get the IoU results for each instrument class, please change the second `target_labels` parameter to the specific label number (e.g., 0) in `engine_instance_seg.py` in line 321.)


