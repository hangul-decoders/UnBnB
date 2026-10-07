"""깨끗한 원문(data/clean)에 난독화 규칙을 적용해 학습·검증·평가A 데이터를 만든다.

- 원문 기준 분할: ratings_train 출처 → 학습/검증, ratings_test 출처 → 평가 A (같은 원문이 양쪽에 섞이지 않음)
- 학습 원문 하나당 난독문 n_variants개 생성
- 망각 대응(대안 A): 학습 데이터의 identity_ratio 비율만큼 (원문, 원문) 쌍을 섞어 "멀쩡한 문장은 그대로 두기"를 유지

실행 (프로젝트 루트에서): python part2/build_dataset.py
출력: data/pairs/{train,val,test_a}.tsv  (열: 입력\t정답\t적용규칙)
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "part2"))
from obfuscate import check_pair, obfuscate  # noqa: E402


def read(path):
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("input\ttarget\trules\n")
        for x, y, r in rows:
            f.write(f"{x}\t{y}\t{r}\n")


def make_pairs(origs, rng, n_variants):
    rows = []
    for s in origs:
        seen = set()
        for _ in range(n_variants):
            obf, applied = obfuscate(s, rng)
            if obf in seen or check_pair(obf, s):
                continue
            seen.add(obf)
            rows.append((obf, s, ",".join(applied)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_variants", type=int, default=2)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--n_test_a", type=int, default=1000)
    ap.add_argument("--identity_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    train_src = read(ROOT / "data/clean/train.txt")
    test_src = read(ROOT / "data/clean/test.txt")
    test_set = set(test_src)
    train_src = [s for s in train_src if s not in test_set]  # 출처가 달라도 같은 문장이면 제외
    rng.shuffle(train_src)
    val_src, train_src = train_src[:args.n_val], train_src[args.n_val:]
    test_a_src = rng.sample(test_src, min(args.n_test_a, len(test_src)))

    train = make_pairs(train_src, rng, args.n_variants)
    n_id = int(len(train) * args.identity_ratio)
    train += [(s, s, "원문유지") for s in rng.sample(train_src, min(n_id, len(train_src)))]
    rng.shuffle(train)
    val = make_pairs(val_src, rng, 1)
    test_a = make_pairs(test_a_src, rng, 1)

    out = ROOT / "data/pairs"
    out.mkdir(parents=True, exist_ok=True)
    write(out / "train.tsv", train)
    write(out / "val.tsv", val)
    write(out / "test_a.tsv", test_a)
    print(f"원문: 학습 {len(train_src):,} / 검증 {len(val_src):,} / 평가A {len(test_a_src):,}")
    print(f"쌍: 학습 {len(train):,} (원문유지 {n_id:,} 포함) / 검증 {len(val):,} / 평가A {len(test_a):,}")


if __name__ == "__main__":
    main()
