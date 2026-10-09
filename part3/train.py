"""(난독문, 원문) 쌍으로 어휘 확장 gec_kr(models/gec_kr_ext)을 추가 학습한다.

실행 예 (프로젝트 루트에서):
  시험 학습 (1만 쌍): python part3/train.py --limit 10000 --epochs 1 --name trial_10k
  전체 학습:          python part3/train.py --epochs 3 --name full
  동적 난독화 학습:   python part3/train.py --dynamic --dyn_repeat 4 --epochs 8 --name dyn_e8
    (원문만 읽고, 원문이 나올 때마다 새로 비틀어서 같은 난독문을 두 번 보지 않게 함 = 외우기 방지)
출력: models/<name>/ (검증 완전 일치율이 가장 좋은 체크포인트), logs/train_<name>.tsv
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "part1"))
sys.path.insert(0, str(ROOT / "part2"))
from gec_utils import correct  # noqa: E402
from obfuscate import obfuscate  # noqa: E402

_RNG = random.Random()  # 동적 난독화용. DataLoader 워커마다 다른 시드로 초기화


def _seed_worker(_):
    _RNG.seed(torch.initial_seed())  # 에폭·워커마다 시드가 달라짐


def dynamic_pairs(origs, identity_ratio):
    """원문 리스트 -> 그때그때 새로 비튼 (난독문, 원문) 쌍"""
    rows = []
    for s in origs:
        if _RNG.random() < identity_ratio:
            rows.append((s, s))
        else:
            rows.append((obfuscate(s, _RNG)[0], s))
    return rows


def read_tsv(path, limit=None):
    with open(path, encoding="utf-8") as f:
        rows = [(r["input"], r["target"]) for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]
    return rows[:limit] if limit else rows


def collate(batch, tok, max_len):
    pad = tok.pad_token_id
    xs = [tok.encode(x)[:max_len] for x, _ in batch]
    ys = [tok.encode(y)[:max_len] for _, y in batch]
    wx, wy = max(map(len, xs)), max(map(len, ys))
    input_ids = torch.tensor([x + [pad] * (wx - len(x)) for x in xs])
    labels = torch.tensor([y + [-100] * (wy - len(y)) for y in ys])
    return input_ids, (input_ids != pad).long(), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--init", default=str(ROOT / "models/gec_kr_ext"))
    ap.add_argument("--limit", type=int, default=None, help="학습 쌍 앞 N개만")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--n_val_gen", type=int, default=500, help="완전 일치율을 잴 검증 문장 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default="data/pairs", help="train.tsv, val.tsv가 있는 폴더")
    ap.add_argument("--dynamic", action="store_true", help="에폭마다 난독문을 새로 만듦 (외우기 방지)")
    ap.add_argument("--dyn_repeat", type=int, default=4, help="동적 모드: 한 에폭에 원문당 몇 번 보여줄지")
    ap.add_argument("--identity_ratio", type=float, default=0.1, help="동적 모드: (원문, 원문) 쌍 비율")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    tok = PreTrainedTokenizerFast.from_pretrained(args.init)
    model = BartForConditionalGeneration.from_pretrained(args.init).to(device)

    train = read_tsv(ROOT / args.data / "train.tsv", args.limit)
    val = read_tsv(ROOT / args.data / "val.tsv")
    val_gen = val[:args.n_val_gen]
    if args.dynamic:
        origs = sorted({y for _, y in train})  # 학습 원문만 (검증·평가 원문은 train.tsv에 없음)
        train = origs * args.dyn_repeat
        print(f"동적 난독화: 학습 원문 {len(origs):,}개 x {args.dyn_repeat}회/에폭", flush=True)
        dl = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4,
                        worker_init_fn=_seed_worker, persistent_workers=False,
                        collate_fn=lambda b: collate(dynamic_pairs(b, args.identity_ratio), tok, args.max_len))
    else:
        dl = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tok, args.max_len))
    val_dl = DataLoader(val, batch_size=args.batch_size, collate_fn=lambda b: collate(b, tok, args.max_len))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * 0.05), total)

    out_dir = ROOT / "models" / args.name
    log_path = ROOT / "logs" / f"train_{args.name}.tsv"
    log_path.parent.mkdir(exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")
    log.write("epoch\tstep\ttrain_loss\tval_loss\tval_em\tminutes\n")
    print(f"학습 {len(train):,}{'문장(동적)' if args.dynamic else '쌍'}, 검증 {len(val):,}쌍, 스텝 {total:,}", flush=True)

    best_em, t0, step = -1.0, time.time(), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for i, (x, m, y) in enumerate(dl, 1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x.to(device), attention_mask=m.to(device), labels=y.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            run_loss += loss.item()
            if i % 200 == 0:
                print(f"epoch {epoch} step {i}/{len(dl)} loss {run_loss / i:.4f} "
                      f"{(time.time() - t0) / 60:.1f}분", flush=True)

        model.eval()
        with torch.no_grad():
            vl = [model(input_ids=x.to(device), attention_mask=m.to(device), labels=y.to(device)).loss.item()
                  for x, m, y in val_dl]
        preds = correct([x for x, _ in val_gen], tok, model, device, batch_size=128, repetition_penalty=1.0)
        em = sum(p == y for p, (_, y) in zip(preds, val_gen)) / len(val_gen)
        val_loss = sum(vl) / len(vl)
        minutes = (time.time() - t0) / 60
        print(f"== epoch {epoch}: train_loss {run_loss / len(dl):.4f} val_loss {val_loss:.4f} "
              f"val_EM {em:.1%} ({minutes:.1f}분)", flush=True)
        for (x, y), p in list(zip(val_gen, preds))[:5]:
            print(f"   {x} -> {p} (정답: {y})", flush=True)
        log.write(f"{epoch}\t{step}\t{run_loss / len(dl):.4f}\t{val_loss:.4f}\t{em:.4f}\t{minutes:.1f}\n")
        log.flush()
        if em > best_em:
            best_em = em
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            print(f"   저장: {out_dir} (최고 val_EM)", flush=True)
    log.close()


if __name__ == "__main__":
    main()
