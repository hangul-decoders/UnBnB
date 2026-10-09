#!/bin/bash
# 실험 4: 동적 난독화(에폭마다 새로 비틀기, 외우기 방지) + 새 평가 세트 A2 (2026-10-09 dh 승인)
set -e
cd ~/UnBnB
export HF_HOME=~/UnBnB/.hf_cache TMPDIR=~/UnBnB/tmp
PY=.venv/bin/python
$PY part2/make_test_a2.py > logs/make_test_a2.log 2>&1
# 동작 확인용 짧은 실행 (원문 약 2천 개, 1에폭)
$PY part3/train.py --dynamic --limit 2000 --epochs 1 --n_val_gen 200 --name dyn_smoke > logs/dyn_smoke.log 2>&1
rm -rf models/dyn_smoke
$PY part3/train.py --dynamic --dyn_repeat 4 --epochs 8 --n_val_gen 2000 --name dyn_e8 > logs/dyn_e8.log 2>&1
$PY part3/evaluate.py --models models/normal_v4_e6 models/dyn_e8 --tag dyn_A > logs/eval_dyn_A.log 2>&1
$PY part3/evaluate.py --models gec_kr models/normal_v4_e6 models/dyn_e8 --test_a data/pairs/test_a2.tsv --tag dyn_A2 > logs/eval_dyn_A2.log 2>&1
echo ALL_DONE > logs/exp4.done
