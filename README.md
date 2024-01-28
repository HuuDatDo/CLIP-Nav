# CLIP-NAV: Pretraining Image-Language for Drone Navigation

This work extends from [Learning to Map Natural Language Instructions to Physical Quadcopter Control using Simulated Flight](https://arxiv.org/abs/1910.09664) by [Valts Blukis](https://www.cs.cornell.edu/~valts/). This repo introduces a new method to achieve a comparable success rate for drone navigation without the need for auxiliary losses. Specifically, CLIP-NAV leverages the zero-shot capabilities of CLIP and combines it with the LingUNet via patch-wise product and regularization. **A full report will be uploaded soon!**

##System Setup

Please follow the instructions to download data and simulation in the original [drif repo](https://github.com/lil-lab/drif?tab=readme-ov-file#data-and-simulator-download)

## Running Experiments
### 1. Create anaconda environment:

`conda create -n <env_name>`

`pip install -r requirements.txt`

### 2. Training

Pre-train Stage 1 of the model for 25 epochs on real and sim oracle rollouts:
`python mains/train/train_supervised_bidomain.py corl_2019/cliplingunet_stage1_bidomain_aug1-2`

Run SuReAL to jointly train CLIP-Nav Stage 1 with supervised learning and Stage 2 with RL:
`python mains/train/train_sureal.py corl_2019/sureal_train_pvn2_bidomain_aug1-2`

### 3. Evaluation:

`python mains/eval/evaluate.py corl_2019/eval/tables/dev_small/eval_clipunet_sureal_dev_small_sim`
