# README

1. Create conda environment:

`conda create -n <env_name>`

`pip install -r requirements.txt`

2. Download checkpoints:

For Lseg-based model:

https://drive.google.com/file/d/1lVW8MLe6SnCug2lDSJFGSbK9pOB8WnTY/view?usp=sharing

For CLIP-based model:

https://drive.google.com/file/d/1WU2Sqs0OhO18ObWQ_1SMTMSsToA7obDE/view?usp=sharing

3. Run evaluation:

On Lseg-based model:

`python mains/eval/evaluate.py corl_2019/eval/tables/dev_small/eval_lseg_sureal_dev_small_sim`

On CLIP-based model:

`python mains/eval/evaluate.py corl_2019/eval/tables/dev_small/eval_cliplingunet_sureal_dev_small_sim`