"""NSMC에서 맞춤법·띄어쓰기가 맞는 '깨끗한' 문장만 골라 원문으로 쓴다.

1차 (규칙): 자모(ㅋㅋ, ㅠㅠ, 초성체), 허용 외 문자, 같은 글자 3번 이상 반복, 한글 비율 낮은 문장 제거
2차 (gec_kr): gec_kr에 넣어 출력이 입력과 똑같은(고칠 것 없는) 문장만 남김

실행 (프로젝트 루트에서): python part2/filter_clean.py [--limit N]
출력: data/clean/{train,test}.txt  (한 줄에 한 문장), data/clean/filter_log.tsv, part2/필터_결과.md
"""

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "part1"))

ALLOWED = re.compile(r"^[가-힣0-9 .,!?~'\"()%\-]+$")
REPEAT = re.compile(r"(.)\1\1")
JAMO = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]")

# dh가 샘플 검토에서 지적한 문장 (걸러져야 정상)
FLAGGED = [
    "주연배우들이 허접해서 무슨 독립영화만치도 몰입도가 없슴.재미도없고.",
    "진짜 연기도 잘하고 너무 생생함재미잏는 권투까지 10점안주면..",
    "빂이어서 그런지 참 재밌구나",
]


def load_nsmc(path):
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    sents = [ln.split("\t")[1].strip() for ln in lines if ln.count("\t") == 2]
    sents = [re.sub(r"\s+", " ", s) for s in sents]
    sents = [s for s in sents if len(re.findall(r"[가-힣]", s)) >= 5 and len(s) <= 60]
    return list(dict.fromkeys(sents))


def rule_reason(s):
    if JAMO.search(s):
        return "자모"
    if not ALLOWED.match(s):
        return "허용 외 문자"
    if REPEAT.search(s):
        return "3회 반복"
    if len(re.findall(r"[가-힣]", s)) / len(s.replace(" ", "")) < 0.7:
        return "한글 비율 낮음"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="파일별 앞 N문장만 (시험용)")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    from gec_utils import correct, load
    tok, model, device = load()

    out_dir = ROOT / "data/clean"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = ["# NSMC 깨끗한 문장 필터 결과", "",
              "| 파일 | 전처리 후 | 1차 통과 | 2차 통과 (최종) | 최종 비율 |", "|---|---|---|---|---|"]
    reasons_all = {}
    log = open(out_dir / "filter_log.tsv", "w", encoding="utf-8")
    log.write("split\tsentence\tresult\tgec_output\n")
    examples = []

    for split, fname in [("train", "ratings_train.txt"), ("test", "ratings_test.txt")]:
        sents = load_nsmc(ROOT / "data/raw" / fname)
        if args.limit:
            sents = sents[:args.limit]
        stage1 = []
        for s in sents:
            r = rule_reason(s)
            reasons_all[r] = reasons_all.get(r, 0) + 1
            if r:
                log.write(f"{split}\t{s}\t1차:{r}\t\n")
            else:
                stage1.append(s)
        t0 = time.time()
        preds = correct(stage1, tok, model, device, batch_size=args.batch_size)
        print(f"[{split}] gec_kr {len(stage1):,}문장 {time.time() - t0:.0f}초", flush=True)
        clean = []
        for s, p in zip(stage1, preds):
            ok = p == s
            log.write(f"{split}\t{s}\t{'통과' if ok else '2차:교정됨'}\t{p}\n")
            if ok:
                clean.append(s)
            elif len(examples) < 30:
                examples.append((s, p))
        (out_dir / f"{split}.txt").write_text("\n".join(clean) + "\n", encoding="utf-8")
        report.append(f"| {fname} | {len(sents):,} | {len(stage1):,} | {len(clean):,} | "
                      f"{len(clean) / len(sents):.1%} |")
    log.close()

    flagged_pred = correct(FLAGGED, tok, model, device)
    report += ["", "1차 제거 사유: " + ", ".join(f"{k} {v:,}" for k, v in reasons_all.items() if k), "",
               "## dh가 지적한 문장 확인 (걸러져야 정상)", "",
               "| 문장 | 1차 | gec_kr 출력 | 최종 |", "|---|---|---|---|"]
    for s, p in zip(FLAGGED, flagged_pred):
        r = rule_reason(s)
        final = "제거" if r or p != s else "**통과 (필터가 못 잡음)**"
        report.append(f"| {s} | {r or '통과'} | {p} | {final} |")
    report += ["", "## 2차에서 걸러진 예 (gec_kr이 고친 문장)", "", "| 원래 문장 | gec_kr 출력 |", "|---|---|"]
    report += [f"| {s} | {p} |" for s, p in examples]
    (ROOT / "part2/필터_결과.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report[:12]))


if __name__ == "__main__":
    main()
