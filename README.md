# README

1. Create conda environment:

`conda create -n <env_name>`
`pip install -r requirements.txt`

2. Run evaluation:

On Lseg-based model:
`python mains/eval/evaluate.py corl_2019/eval/tables/dev_small/eval_lseg_sureal_dev_small_sim`

On CLIP-based model:
`python mains/eval/evaluate.py corl_2019/eval/tables/dev_small/eval_cliplingunet_sureal_dev_small_sim`