"""최종 보고용 새 평가 세트 A2를 만든다 (2026-10-09 dh: "테스트는 항상 새로운 문장으로").

- 평가 A는 여러 실험에서 반복해서 봤으므로, 학습·검증·평가 A 어디에도 없는 원문으로 1,000문장을 새로 뽑는다.
- 난독화는 학습과 같은 normal 규칙.
- A2는 모델 선택·튜닝에 쓰지 않고 최종 결과 보고에만 쓴다.

실행 (프로젝트 루트에서): python part2/make_test_a2.py
출력: data/pairs/test_a2.tsv
"""

import csv
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "part2"))
from obfuscate import check_pair, obfuscate  # noqa: E402


def targets(path):
    with open(path, encoding="utf-8") as f:
        return {r["target"] for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}


def main():
    used = set()
    for name in ["train.tsv", "val.tsv", "test_a.tsv"]:
        used |= targets(ROOT / "data/pairs" / name)
    long_run = re.compile(r"[가-힣]{11,}")  # build_dataset.py --max_run 10 과 같은 기준
    src = [s for s in (ROOT / "data/clean/test.txt").read_text(encoding="utf-8").splitlines()
           if s.strip() and not long_run.search(s) and s not in used]
    rng = random.Random(2026)
    picked = rng.sample(src, 1000)
    rows = []
    for s in picked:
        obf, applied = obfuscate(s, rng)
        if not check_pair(obf, s):
            rows.append((obf, s, ",".join(applied)))
    with open(ROOT / "data/pairs/test_a2.tsv", "w", encoding="utf-8") as f:
        f.write("input\ttarget\trules\n")
        for x, y, r in rows:
            f.write(f"{x}\t{y}\t{r}\n")
    print(f"후보 원문 {len(src):,}개 (학습·검증·평가A와 겹치지 않음) 중 {len(rows):,}문장 -> data/pairs/test_a2.tsv")


if __name__ == "__main__":
    main()
