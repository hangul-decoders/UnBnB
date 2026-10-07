"""NSMC 문장에 난독화 규칙을 적용해 검토용 샘플 50쌍과 자동 검사 통계를 만든다.

실행: python3 part2/make_samples.py   (프로젝트 루트에서)
출력: part2/샘플50_검토.md
"""

import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from obfuscate import RULES, check_pair, obfuscate

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


def load_nsmc(path):
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    sents = [ln.split("\t")[1].strip() for ln in lines if ln.count("\t") == 2]
    sents = [re.sub(r"\s+", " ", s) for s in sents]
    # 빈 문장 제거, 한글 음절 5자 이상, 길이 60자 이하, 중복 제거
    sents = [s for s in sents if len(re.findall(r"[가-힣]", s)) >= 5 and len(s) <= 60]
    return list(dict.fromkeys(sents))


def main():
    sents = load_nsmc(ROOT / "data/raw/ratings_train.txt")
    rng = random.Random(SEED)

    # 자동 검사: 2만 문장
    pool = rng.sample(sents, 20000)
    problems, rule_cnt, changed = Counter(), Counter(), []
    for s in pool:
        obf, applied = obfuscate(s, rng)
        rule_cnt.update(applied)
        for p in check_pair(obf, s):
            problems[p] += 1
        n_syl = sum("가" <= ch <= "힣" for ch in s)
        changed.append(sum(a != b for a, b in zip(obf, s)) / n_syl)

    # 검토용 50쌍
    sample_rng = random.Random(SEED + 1)
    samples = []
    for s in sample_rng.sample(sents, 50):
        obf, applied = obfuscate(s, sample_rng)
        samples.append((obf, s, applied))

    out = ["# 난독화 샘플 50쌍 (검토용)", "",
           f"- 원문: NSMC ratings_train.txt, 전처리 후 {len(sents):,}문장 중 무작위 (seed {SEED + 1})",
           "- 규칙 코드: `part2/obfuscate.py`, 이 파일 생성: `part2/make_samples.py`",
           "- 검토 방법: 소리 내어 읽었을 때 원문을 알아볼 수 있는지, 에어비앤비체처럼 보이는지 판단.",
           "  이상한 줄은 번호와 함께 '못 읽음 / 너무 약함 / 어색함' 등으로 알려 주기", "",
           "## 자동 검사 (2만 문장)", "",
           f"- 바뀐 글자 비율 평균: {sum(changed) / len(changed):.1%} (한글 음절 기준)",
           "- 문제 발생: " + (", ".join(f"{k} {v}건" for k, v in problems.items()) or "없음"),
           "- 규칙별 적용 횟수: " + ", ".join(f"{r[0]} {rule_cnt[r[0]]:,}" for r in RULES), "",
           "## 샘플", "",
           "| # | 난독문 | 원문 | 적용 규칙 |", "|---|---|---|---|"]
    esc = lambda t: t.replace("|", "\\|")
    for i, (obf, s, applied) in enumerate(samples, 1):
        out.append(f"| {i} | {esc(obf)} | {esc(s)} | {', '.join(applied)} |")
    (ROOT / "part2/샘플50_검토.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:12]))


if __name__ == "__main__":
    main()
