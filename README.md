# Quasi Zigzag Persistence: A Topological Framework for Analyzing Time-Varying Data

This codebase contains implementation and experiments with ZZ-GRIL. 


## Instructions
First clone this repo to say $ZZGRIL. Then create a conda environment by

    conda create -n zzgril python=3.10.4
    
    conda activate zzgril

    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia    
    

**Additional Dependencies:**

1. aeon
2. fzz (https://github.com/TDA-Jyamiti/fzz)

## Running Experiments

### ISRUC Experiments
#### Downloading the Dataset
```python
get_ISRUC_S3.sh
```
This will download 10 files in <code>rawdata/RawData/</code> and 10 files in <code>rawdata/ExtractedChannels/</code>.

#### Running the Experiment
```python
python preprocess_ISRUC_for_zzgril.py
```
This will preprocess ISRUC_S3 data and convert into a sequence of graphs and store the result in <code>data/ISRUC_S3/</code>.
```python
python compute_zz_graph_ISRUC.py --num_center_pts 36 --fold 0
```
This will compute ZZGRIL for the sequence of graphs and store the result in saved_zz_gril/zz_graphs/ISRUC_S3/. This needs to be done from fold 0 to fold 9.

<code>cd STDP-GCN/</code>
```python
python train.py --num_center_pts 36 --use_only_lambda_0 False
```
This will augment ZZ-GRIL to STDP-GCN framework and train the model. <code>--use_only_lambda_0</code> flag is used to denote if you do not want to use the information in $$H_1$$.

### UEA Experiments
To convert the multivariate time series into sequence of graphs and compute ZZGRIL, run
```python
python compute_zz_graph_UEA.py --dataset NATOPS --num_center_pts 36 --num_vert 24
```
Note that <code>--num_vert</code> flag denotes the number of time series in the dataset.

To convert the multivariate time series into sequence of point clouds and compute ZZGRIL, run
```python
python compute_zz_pcd_UEA.py --dataset NATOPS --num_center_pts 36
```

<code>cd TodyNet/ </code>
```python
python train.py --dataset NATOPS --num_center_pts 36 --use_only_lambda_0 False --gril_graphs True
```
This will augment ZZGRIL to TodyNet and train the model. <code>--use_only_lambda_0</code> flag is used to denote if you do not want to use the information in $$H_1$$, <code>--gril_graphs</code> flag is used to denote if you are using ZZGRIL processed as a sequence of graphs or as a sequence of point clouds.
