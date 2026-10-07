#!/bin/bash
# 실험 2: light 데이터 생성, normal/light 6에폭 학습, 평가 (2026-10-07 dh 승인)
set -e
cd ~/UnBnB
export HF_HOME=~/UnBnB/.hf_cache TMPDIR=~/UnBnB/tmp
PY=.venv/bin/python
$PY part2/build_dataset.py --level light --out data/pairs_light > logs/build_light.log 2>&1
$PY part3/train.py --epochs 6 --name normal_e6 > logs/normal_e6.log 2>&1
$PY part3/train.py --epochs 6 --name light_e6 --data data/pairs_light > logs/light_e6.log 2>&1
$PY part3/evaluate.py --models gec_kr models/full models/normal_e6 models/light_e6 --tag normalA > logs/eval_normalA.log 2>&1
$PY part3/evaluate.py --models gec_kr models/full models/normal_e6 models/light_e6 --test_a data/pairs_light/test_a.tsv --tag lightA > logs/eval_lightA.log 2>&1
echo ALL_DONE > logs/exp2.done
