"""학습 원문 확장·정리 (2026-10-09 dh 승인: 원문 수 늘리기 + 정답 원문 정리).

NSMC 학습 파일 출처만 쓴다 (평가 파일 출처는 평가 A/A2용으로 남겨 둠).
1. 기존 깨끗한 원문 (data/clean/train.txt)
2. gec_kr이 띄어쓰기만 고친 문장 → gec_kr 교정본을 원문(정답)으로 사용 (filter_log.tsv에서)
3. "같은 글자 3회 반복"으로만 빠졌던 문장 → 기호 반복(..., !!!, ~~~)을 2개로 줄인 뒤 1차 규칙 재검사
   → gec_kr 재검사: 그대로면 사용, 띄어쓰기만 고쳤으면 교정본 사용
4. 정리: 과거형 받침 오타 패턴(잇어, 엇다, 봣, 햇, 겟 등) 문장, 띄어쓰기 없이 한글 11자 이상 붙은 문장 제외
5. 검증·평가 A·평가 A2 원문과 같은 문장 제외

실행 (프로젝트 루트에서): python part2/expand_clean.py
출력: data/clean_v2/train.txt, part2/원문확장_결과.md
"""

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "part1"))
sys.path.insert(0, str(ROOT / "part2"))
from filter_clean import rule_reason  # noqa: E402

TYPO = re.compile(r"햇|겟|봣|됫|겻|셧|렷|켯|쳣|졋|잇(?=[어다고는음지])|(?<=[가-힣])엇(?=[다어음는고])")
LONG_RUN = re.compile(r"[가-힣]{11,}")
SYMBOL_RUN = re.compile(r"([^가-힣0-9])\1{2,}")


def spacing_only(a, b):
    return a != b and a.replace(" ", "") == b.replace(" ", "")


def eval_targets():
    out = set()
    for name in ["val.tsv", "test_a.tsv", "test_a2.tsv"]:
        with open(ROOT / "data/pairs" / name, encoding="utf-8") as f:
            out |= {r["target"] for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}
    return out


def main():
    from gec_utils import correct, load

    with open(ROOT / "data/clean/filter_log.tsv", encoding="utf-8") as f:
        log = [r for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if r["split"] == "train"]

    base = [s for s in (ROOT / "data/clean/train.txt").read_text(encoding="utf-8").splitlines() if s]
    spaced = [r["gec_output"] for r in log
              if r["result"] == "2차:교정됨" and spacing_only(r["sentence"], r["gec_output"])]

    repeat_src = [r["sentence"] for r in log if r["result"] == "1차:3회 반복"]
    normed = [SYMBOL_RUN.sub(r"\1\1", s) for s in repeat_src]
    normed = list(dict.fromkeys(s for s in normed if rule_reason(s) is None))
    tok, model, device = load()
    preds = correct(normed, tok, model, device, batch_size=128)
    repeat_ok = [s for s, p in zip(normed, preds) if p == s]
    repeat_spaced = [p for s, p in zip(normed, preds) if spacing_only(s, p)]

    held_out = eval_targets()
    groups = [("기존 깨끗한 원문", base), ("띄어쓰기만 교정", spaced),
              ("기호 반복 정리 후 통과", repeat_ok), ("기호 반복 정리 + 띄어쓰기 교정", repeat_spaced)]
    final, seen, rows = [], set(), []
    for name, sents in groups:
        n_typo = n_long = n_dup = n_held = 0
        for s in sents:
            if TYPO.search(s):
                n_typo += 1
            elif LONG_RUN.search(s):
                n_long += 1
            elif s in held_out:
                n_held += 1
            elif s in seen:
                n_dup += 1
            else:
                seen.add(s)
                final.append(s)
        added = len(sents) - n_typo - n_long - n_dup - n_held
        rows.append(f"| {name} | {len(sents):,} | {n_typo:,} | {n_long:,} | {n_held:,} | {n_dup:,} | {added:,} |")

    out = ROOT / "data/clean_v2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(final) + "\n", encoding="utf-8")

    report = ["# 학습 원문 확장·정리 결과", "",
              f"- 기호 반복 문장: {len(repeat_src):,}개 → 기호 정리 후 1차 통과 {len(normed):,}개 → gec_kr 재검사",
              f"- 최종 학습 원문: **{len(final):,}개** (기존 {len(base):,}개)", "",
              "| 출처 | 후보 | 오타 패턴 제외 | 붙여쓰기 제외 | 검증·평가 원문과 같음 | 중복 | 추가 |",
              "|---|---|---|---|---|---|---|"] + rows
    report += ["", "## 예시", "", "| 출처 | 원문(정답) |", "|---|---|"]
    report += [f"| 띄어쓰기만 교정 | {s} |" for s in spaced[:5]]
    report += [f"| 기호 반복 정리 | {s} |" for s in repeat_ok[:5]]
    (ROOT / "part2/원문확장_결과.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report[:12]))


if __name__ == "__main__":
    main()
