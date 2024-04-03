# Action Anticipation with Goal Consistency
Code for the paper 'Action Anticipation with Goal Consistency'.

## Experiments
We conducted experiments on two large-scale datasets: Assembly101 and COIN. For both datasets, we report Mean Top-5 Action /Recall as our main metric. For Assembly101, we additionally report Mean Top-1 Verb and Noun Recall.

| Dataset | Model | Action | Noun | Verb | 
|---------|-------|--------|------|------|
|Assembly101| Ours (1 goal) | 10.39 | 27.50 | 54.59 | 
|Assembly101| Ours (2 goals) | 10.64 | 27.63 | 55.82| 
|COIN| Ours (1 goal) | 13.93 | - | - | 

## Installation
Clone repository and install conda environment:

```
git clone https://github.com/olga-zats/goal_consistency.git
cd goal_consistency
conda env create -f env.yml
conda activate goal
```
## Assembly101

### Data
Download TSM features provided by Assembly101 from [here](https://assembly-101.github.io/). 

Create a directory ```assembly/data```. Download annotations from [here](https://drive.google.com/drive/folders/1_HoWvfF3XFYjEFqOG7yDrIGqLzhGjL2K?usp=sharing) and place them into the ```assembly/data``` folder.

### Training
Before running the training, update the paths to match your system.

Ours (1 goal):
```
python main.py --mode train --epochs 15 \
--path_to_data /home/user/db_TSM_features \
--path_to_models /home/user/models_anticipation \
--path_to_anno data/CSVs \
--modality fixed+ego \
--views all --past_attention \
--batch_size 64 --num_workers 16 \
--predict_latent True \
--single_latent True \
--gt_fc_cons_loss True \
--gt_fc_cons_loss_weight 5.0 \
```

Ours (2 goals):
```
python main.py --mode train --epochs 15 \
--path_to_data /home/user/db_TSM_features \
--path_to_models /home/user/models_anticipation \
--path_to_anno data/CSVs \
--modality fixed+ego \
--views all --past_attention \
--batch_size 64 --num_workers 16 \
--predict_latent True \
--predict_ts_latent True \
--single_latent True \
--gt_fc_cons_loss True \
--gt_fc_cons_loss_weight 2.5 \
--gt_fts_cons_loss True \
--gt_fts_cons_loss_weight 2.5 \
```


### Testing
Before running testing, update the paths to match your system and create the directory ```assembly/json```.

Ours (1 goal)
```
python main.py --mode validate --epochs 15 \
--path_to_data /home/user/db_TSM_features \
--path_to_models  /home/user/models_anticipation \
--path_to_anno data/CSVs \
--modality fixed+ego \
--views all --past_attention \
--batch_size 64 --num_workers 16 \
--predict_latent True \
--single_latent True \
--gt_fc_cons_loss True \
--gt_fc_cons_loss_weight 5.0 \
--save_json json/single_latent_gt_fc_cons_loss_5.0 \
```

Ours (2 goals)
```
python main.py --mode validate --epochs 15 \
--path_to_data /home/user/db_TSM_features \
--path_to_models  /home/user/models_anticipation \
--path_to_anno data/CSVs \
--modality fixed+ego \
--views all --past_attention \
--batch_size 64 --num_workers 16 \
--predict_latent True \
--predict_ts_latent True \
--single_latent True \
--gt_fc_cons_loss True \
--gt_fc_cons_loss_weight 2.5 \
--gt_fts_cons_loss True \
--gt_fts_cons_loss_weight 2.5 \
--save_json json/single_latent_ts_latent_gt_fc_cons_loss_2.5_gt_fts_cons_loss_2.5 \
```

To additionally evaluate the models on the unseen, seen and tail splits, run the following:

Ours (1 goal):

```
python evaluate.py data json/single_latent_ts_latent_gt_fc_cons_loss_2.5_gt_fts_cons_loss_2.5
```

Ours (2 goals):
```
python evaluate.py data json/single_latent_gt_fc_cons_loss_5.0
```



## COIN

### Data

### Features

### Training

### Testing
