# Training EMMA on Messenger

## Setup
The conda environment we use is provided in `env.yml`. You can clone it using:
```
conda env create --name msgr-emma -f env.yml
```
After doing this, you will still need to install the `messenger` package by following the instructions [here](../README.md)

## Running
We provide scripts to submit jobs using SLURM, you may need to modify / adapt these for your own systems or if you want to run these interactively. For example to submit jobs on stage 1:
```
cd training/stage_1
conda activate msgr-emma
bash stage_1.sh
```
Note: We run EMMA on stage 1 for 12 hours, but this is to make comparisons with baselines that take this long to converge. In general EMMA should converge much faster than this.

## Results
You can see the training curves generated using the provided scripts at [this wandb project](https://wandb.ai/ahjwang/msgr-emma?workspace=user-ahjwang)