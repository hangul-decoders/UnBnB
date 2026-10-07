"""기준 모델(gec_kr)과 학습한 모델을 평가 데이터 A/B/C로 비교한다.

지표
  - 완전 일치율(EM): 출력 == 정답
  - CER: 글자 단위 편집 거리 / 정답 글자 수
  - 의미 보존: 정답의 부정 표현(안·못·않·없·아니)과 숫자가 출력에 그대로 있는 비율
비교 전 NFKC 정규화와 공백 정리를 한다 (토크나이저가 ㅋ을 ᄏ로 바꾸는 문제 대응).

실행 (프로젝트 루트에서): python part3/evaluate.py --models gec_kr models/full
평가 파일: data/pairs/test_a.tsv, data/eval/test_b.tsv, data/eval/test_c.tsv (있는 것만)
출력: part3/평가_결과.md, part3/평가_오류사례.tsv
"""

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "part1"))
from gec_utils import correct, load  # noqa: E402

TEST_SETS = {"A (같은 규칙)": ROOT / "data/pairs/test_a.tsv",
             "B (airbnbfy)": ROOT / "data/eval/test_b.tsv",
             "C (사람이 작성)": ROOT / "data/eval/test_c.tsv"}
NUM = re.compile(r"\d+")


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def negations(s):
    return sorted(m.group().strip() for m in re.finditer(r"못|않|없|아니|(?<![가-힣])안(?![가-힣])", s))


def score(preds, golds):
    n = len(golds)
    em = sum(norm(p) == norm(g) for p, g in zip(preds, golds)) / n
    cer = sum(edit_distance(norm(p), norm(g)) for p, g in zip(preds, golds)) / sum(len(norm(g)) for g in golds)
    neg_idx = [i for i, g in enumerate(golds) if negations(norm(g))]
    num_idx = [i for i, g in enumerate(golds) if NUM.search(g)]
    neg = (sum(negations(norm(preds[i])) == negations(norm(golds[i])) for i in neg_idx) / len(neg_idx)
           if neg_idx else None)
    num = (sum(NUM.findall(preds[i]) == NUM.findall(golds[i]) for i in num_idx) / len(num_idx)
           if num_idx else None)
    return em, cer, neg, len(neg_idx), num, len(num_idx)


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        return [(r["input"], r["target"]) for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="gec_kr(기준 모델) 또는 models/<이름> 경로")
    ap.add_argument("--test_a", default="data/pairs/test_a.tsv", help="평가 A 파일 (강도별 데이터면 바꿔서 지정)")
    ap.add_argument("--tag", default="", help="결과 파일 이름 뒤에 붙일 이름 (예: light → 평가_결과_light.md)")
    args = ap.parse_args()
    TEST_SETS["A (같은 규칙)"] = ROOT / args.test_a
    suffix = f"_{args.tag}" if args.tag else ""

    sets = {k: read_tsv(p) for k, p in TEST_SETS.items() if p.exists()}
    pct = lambda v: "-" if v is None else f"{v:.1%}"
    out = ["# 평가 결과", "", "| 모델 | 평가 데이터 | 문장 수 | 완전 일치율 | CER | 부정 표현 유지 | 숫자 유지 |",
           "|---|---|---|---|---|---|---|"]
    errors = open(ROOT / f"part3/평가_오류사례{suffix}.tsv", "w", encoding="utf-8")
    errors.write("model\tset\tinput\toutput\ttarget\n")
    for name in args.models:
        baseline = name == "gec_kr"
        tok, model, device = load("Soyoung97/gec_kr" if baseline else str(ROOT / name))
        for set_name, rows in sets.items():
            preds = correct([x for x, _ in rows], tok, model, device, batch_size=128,
                            repetition_penalty=2.0 if baseline else 1.0)
            em, cer, neg, n_neg, num, n_num = score(preds, [y for _, y in rows])
            out.append(f"| {name} | {set_name} | {len(rows):,} | {em:.1%} | {cer:.1%} | "
                       f"{pct(neg)} ({n_neg}) | {pct(num)} ({n_num}) |")
            print(out[-1], flush=True)
            for (x, y), p in zip(rows, preds):
                if norm(p) != norm(y):
                    errors.write(f"{name}\t{set_name}\t{x}\t{p}\t{y}\n")
        del model
    errors.close()
    out += ["", "- CER은 낮을수록 좋음. 괄호 안 숫자는 해당 항목이 있는 정답 문장 수",
            "- 기준 모델은 README 설정(repetition_penalty 2.0), 학습한 모델은 1.0으로 생성",
            f"- 평가 A 파일: `{args.test_a}`", f"- 틀린 사례 전체: `part3/평가_오류사례{suffix}.tsv`"]
    (ROOT / f"part3/평가_결과{suffix}.md").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
