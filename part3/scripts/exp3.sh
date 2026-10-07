#!/bin/bash
# 실험 3: 원문당 난독문 4개 (과적합 완화), normal 6에폭 (2026-10-07 dh 승인)
# 검증·평가 A는 기존 data/pairs와 같은 파일을 써서 normal_e6과 바로 비교
set -e
cd ~/UnBnB
export HF_HOME=~/UnBnB/.hf_cache TMPDIR=~/UnBnB/tmp
PY=.venv/bin/python
$PY part2/build_dataset.py --n_variants 4 --out data/pairs_v4 > logs/build_v4.log 2>&1
cp data/pairs/val.tsv data/pairs/test_a.tsv data/pairs_v4/
$PY part3/train.py --epochs 6 --name normal_v4_e6 --data data/pairs_v4 > logs/normal_v4_e6.log 2>&1
$PY part3/evaluate.py --models models/normal_e6 models/normal_v4_e6 --tag v4_normalA > logs/eval_v4_normalA.log 2>&1
$PY part3/evaluate.py --models models/normal_e6 models/normal_v4_e6 --test_a data/pairs_light/test_a.tsv --tag v4_lightA > logs/eval_v4_lightA.log 2>&1
echo ALL_DONE > logs/exp3.done
