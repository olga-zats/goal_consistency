# Action Anticipation with Goal Consistency
Code for the paper 'Action Anticipation with Goal Consistency'.

## Experiments
We conducted experiments on two large-scale datasets: Assembly101 and COIN. For both datasets, we report Mean Top-5 Action /Recall as our main metric. For Assembly101, we additionally report Mean Top-1 Verb and Noun Recall.

| Dataset | Model | Action | Noun | Verb | 
|---------|-------|--------|------|------|------------|
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

### Training

### Testing

### Models

## COIN

### Data

### Features

### Training

### Testing
