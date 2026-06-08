#!/usr/bin/env python3
"""
diag_compare_logs.py — why does dpv score lower than off?
Checks: (1) do the two logs share the same (query,option) keys, or are we
comparing different images?  (2) dpv collection health: rank_used, all_collected,
gate rejects.  (3) overlap-only score sanity (re-reads the judge_r5 output).

Usage:
  python diag_compare_logs.py \
    --logs dpv:data/images_dpv/collection_log.jsonl,off:data/images_siglip_top1/collection_log.jsonl \
    --judge data/eval/judge_r5.jsonl
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

def load(p):
    out=[]
    if not Path(p).exists(): return out
    for l in open(p):
        l=l.strip()
        if l: out.append(json.loads(l))
    return out

def parse_logs(s):
    out=[]
    for c in s.split(","):
        c=c.strip()
        if not c: continue
        if ":" in c and not c.split(":",1)[1].startswith("//"):
            n,p=c.split(":",1)
        else:
            p=c; n=Path(p).parent.name
        out.append((n.strip(),p.strip()))
    return out

def keyset(recs):
    ks=set()
    for r in recs:
        q=r.get("query_id")
        for k in "ABCD":
            o=(r.get("options") or {}).get(k) or {}
            if o.get("image_path"): ks.add(f"{q}|{k}")
    return ks

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--logs",required=True)
    ap.add_argument("--judge",default=None)
    a=ap.parse_args()
    logs=parse_logs(a.logs)
    data={n:load(p) for n,p in logs}

    print("="*70)
    print("  (1) LOG COVERAGE")
    print("="*70)
    sets={}
    for n,recs in data.items():
        ks=keyset(recs)
        sets[n]=ks
        nq=len({k.split('|')[0] for k in ks})
        print(f"  {n:6s}: {len(recs)} records, {nq} queries, {len(ks)} filled options")
    if len(sets)==2:
        a_,b_=list(sets)
        inter=sets[a_]&sets[b_]
        print(f"\n  shared (query,option) between {a_} & {b_}: {len(inter)}")
        print(f"  only in {a_}: {len(sets[a_]-sets[b_])}   only in {b_}: {len(sets[b_]-sets[a_])}")
        if len(inter) < 0.5*min(len(sets[a_]),len(sets[b_])):
            print("  *** WARNING: small overlap -> you are NOT comparing the same images. ***")

    print("\n"+"="*70)
    print("  (2) DPV COLLECTION HEALTH (per method)")
    print("="*70)
    for n,recs in data.items():
        rank_hi=0; notfull=0; tot=0; gender_incons=0
        rank_dist=defaultdict(int)
        for r in recs:
            if r.get("all_collected") is False: notfull+=1
            if r.get("gender_consistent") is False: gender_incons+=1
            for k in "ABCD":
                o=(r.get("options") or {}).get(k) or {}
                if not o.get("image_path"): continue
                tot+=1
                ru=o.get("rank_used")
                if isinstance(ru,int):
                    rank_dist[ru]+=1
                    if ru>=3: rank_hi+=1
        print(f"  {n:6s}: options={tot}  rank>=3:{rank_hi}  "
              f"all_collected=False:{notfull}  gender_inconsistent:{gender_incons}")
        if rank_dist:
            top=sorted(rank_dist.items())[:6]
            print(f"          rank_used dist: {dict(top)}")

    if a.judge and Path(a.judge).exists():
        print("\n"+"="*70)
        print("  (3) OVERLAP-ONLY SCORE (same (query,option) judged in BOTH methods)")
        print("="*70)
        rows=load(a.judge)
        by=defaultdict(dict)  # key -> {method: overall}
        meta={}
        for r in rows:
            j=r.get("judge") or {}
            if "garment" not in j: continue
            try:
                ov=(int(j["garment"])+int(j["color"])+int(j["pattern"]))/3.0
            except Exception: continue
            key=f"{r['query_id']}|{r['option']}"
            by[key][r["method"]]=ov
            meta[key]=r.get("active_axis")
        methods=sorted({m for d in by.values() for m in d})
        if len(methods)==2:
            m1,m2=methods
            paired=[(k,v[m1],v[m2]) for k,v in by.items() if m1 in v and m2 in v]
            print(f"  paired (query,option) judged in both: {len(paired)}")
            if paired:
                a1=sum(x[1] for x in paired)/len(paired)
                a2=sum(x[2] for x in paired)/len(paired)
                print(f"  mean overall on SHARED set:  {m1}={a1:.2f}  {m2}={a2:.2f}")
                better=sum(1 for _,x,y in paired if y>x)
                worse =sum(1 for _,x,y in paired if y<x)
                print(f"  {m2} better than {m1}: {better}   worse: {worse}   tie: {len(paired)-better-worse}")
        else:
            print("  judge file has only one method's rows -> cannot pair. "
                  "Re-run the judge over BOTH logs with the SAME (query,option) set.")

if __name__=="__main__":
    main()
