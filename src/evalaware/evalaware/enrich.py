from __future__ import annotations

def enrich_items(items, cfg):
    out=[]
    for i,row in enumerate(items):
        r=dict(row)
        r["eval_like_text"] = bool(i%3==0)
        r["safety_label"] = int(row.get("label",0))
        r["verbalized_awareness"] = int(i%4==0)
        r["construct_valid"] = int(i%5!=0)
        out.append(r)
    return out

