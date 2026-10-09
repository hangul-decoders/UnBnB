#!/bin/bash
# 실험 5: 학습 원문 확장·정리 + 동적 난독화 8에폭, 빔 8 평가 (2026-10-09 dh 승인)
set -e
cd ~/UnBnB
export HF_HOME=~/UnBnB/.hf_cache TMPDIR=~/UnBnB/tmp
PY=.venv/bin/python
$PY part2/expand_clean.py > logs/expand_clean.log 2>&1
$PY part3/train.py --dynamic --orig_file data/clean_v2/train.txt --dyn_repeat 4 --epochs 8 --n_val_gen 2000 --name dyn_v2_e8 > logs/dyn_v2_e8.log 2>&1
$PY part3/evaluate.py --models models/dyn_e8 models/dyn_v2_e8 --test_a data/pairs/val.tsv --num_beams 8 --tag v2_val > logs/eval_v2_val.log 2>&1
$PY part3/evaluate.py --models models/dyn_e8 models/dyn_v2_e8 --num_beams 8 --tag v2_A > logs/eval_v2_A.log 2>&1
$PY part3/evaluate.py --models models/dyn_e8 models/dyn_v2_e8 --test_a data/pairs/test_a2.tsv --num_beams 8 --tag v2_A2 > logs/eval_v2_A2.log 2>&1
echo ALL_DONE > logs/exp5.done
