#!/usr/bin/env python3
"""대화 검수용 HTML 뷰어 생성.

    POD_VARIANT=wacv_scenario_v5 python -m dialog.make_viewer
    → data_<variant>/dialogs/viewer.html  (브라우저로 열면 됨)

목적은 감상이 아니라 **검수**다. 한 화면에서 세 가지를 대조할 수 있어야 한다:
발화가 무엇을 주장하는가 / 이미지가 실제로 무엇인가 / 프로필의 정답은 무엇인가.
그래서 evidence 턴마다 이미지 아래에 속성을 해독해 붙이고, evidence 축은
강조하고, 우측에 18개 취향 원장을 두어 노출 상태를 추적한다.

이미지는 상대 경로(기본 ../images)로 참조한다. 파일이 없으면 속성 타일로
대체되므로 이미지 없는 환경에서도 구조 검수가 가능하다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from configs.config import DATA_DIR                                  # noqa: E402
from configs.dialog_templates import AXIS_WORD, EPISODES             # noqa: E402

HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>POD-Bench · dialogue inspector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --ink:#101318; --ink-2:#39414d; --ink-3:#79838f;
  --paper:#f2f3f5; --card:#ffffff; --line:#dfe3e8;
  --keep:#0f766e;        /* like */
  --keep-bg:#e6f2f0;
  --drop:#a21456;        /* dislike */
  --drop-bg:#fbe9f1;
  --signal:#1d4ed8;      /* evidence axis */
  --signal-bg:#e7edfd;
  --sans:'Space Grotesk',system-ui,-apple-system,sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
     font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
a{color:inherit}
.wrap{display:grid;grid-template-columns:220px minmax(0,1fr) 300px;
      gap:0;min-height:100vh}

/* ── 좌: 대화 목록 ─────────────────────────────────────────────── */
.rail{border-right:1px solid var(--line);background:var(--card);
      position:sticky;top:0;height:100vh;overflow-y:auto}
.rail h1{font-size:13px;letter-spacing:.14em;text-transform:uppercase;
         margin:0;padding:18px 16px 6px;color:var(--ink-3);font-weight:500}
.rail .meta{padding:0 16px 14px;font-family:var(--mono);font-size:11px;
            color:var(--ink-3);border-bottom:1px solid var(--line)}
.uitem{display:flex;justify-content:space-between;align-items:center;
       padding:9px 16px;cursor:pointer;border-left:3px solid transparent;
       font-family:var(--mono);font-size:13px}
.uitem:hover{background:var(--paper)}
.uitem.on{background:var(--signal-bg);border-left-color:var(--signal);
          color:var(--signal);font-weight:600}
.uitem span{font-size:11px;color:var(--ink-3)}
.uitem.on span{color:var(--signal)}
.flag{display:inline-block;width:6px;height:6px;border-radius:50%;
      background:#d97706;margin-left:6px;vertical-align:middle}

/* ── 중: 대화 ──────────────────────────────────────────────────── */
main{padding:26px 34px 90px;max-width:900px}
.head{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;
      margin-bottom:6px}
.head h2{font-size:27px;margin:0;letter-spacing:-.02em}
.head .sub{font-family:var(--mono);font-size:12px;color:var(--ink-3)}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0 26px;
         padding-bottom:16px;border-bottom:1px solid var(--line)}
.tog{font-family:var(--mono);font-size:11px;letter-spacing:.04em;
     padding:5px 11px;border:1px solid var(--line);background:var(--card);
     border-radius:2px;cursor:pointer;color:var(--ink-2)}
.tog.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.tog:focus-visible{outline:2px solid var(--signal);outline-offset:2px}

.ep{margin:30px 0 12px;display:flex;align-items:center;gap:12px}
.ep b{font-family:var(--mono);font-size:11px;letter-spacing:.12em;
      text-transform:uppercase;color:var(--ink-3);font-weight:500}
.ep i{flex:1;height:1px;background:var(--line)}
.ep time{font-family:var(--mono);font-size:11px;color:var(--ink-3)}

.turn{display:flex;gap:10px;margin:8px 0;align-items:flex-start}
.turn.assistant{justify-content:flex-end}
.bub{max-width:78%;padding:10px 14px;border-radius:14px;background:var(--card);
     border:1px solid var(--line)}
.turn.assistant .bub{background:#eceef1;border-color:#eceef1;color:var(--ink-2)}
.who{font-family:var(--mono);font-size:10px;color:var(--ink-3);
     letter-spacing:.1em;padding-top:11px;min-width:30px}
.turn.assistant .who{order:2;text-align:right}

/* evidence 턴 */
.turn.ev .bub{border-left:3px solid var(--ink-3)}
.turn.ev.like .bub{border-left-color:var(--keep)}
.turn.ev.dislike .bub{border-left-color:var(--drop)}
.tag{display:inline-flex;gap:6px;align-items:center;font-family:var(--mono);
     font-size:10px;letter-spacing:.06em;margin-bottom:6px}
.tag em{font-style:normal;padding:1px 6px;border-radius:2px;
        background:var(--paper);color:var(--ink-2)}
.tag em.pol-like{background:var(--keep-bg);color:var(--keep)}
.tag em.pol-dislike{background:var(--drop-bg);color:var(--drop)}
.tag em.dg{background:#fef3c7;color:#92400e}

.shots{display:flex;gap:10px;margin-top:10px;flex-wrap:wrap}
figure{margin:0;width:112px}
figure img{width:112px;height:112px;object-fit:cover;border-radius:6px;
           border:1px solid var(--line);background:#fff;display:block}
figure.miss img{display:none}
figure .alt{display:none;width:112px;height:112px;border-radius:6px;
            border:1px dashed var(--line);background:repeating-linear-gradient(
            45deg,#fafbfc,#fafbfc 6px,#f1f3f5 6px,#f1f3f5 12px);
            align-items:center;justify-content:center;font-family:var(--mono);
            font-size:9px;color:var(--ink-3);text-align:center;padding:8px}
figure.miss .alt{display:flex}
figcaption{margin-top:5px;font-family:var(--mono);font-size:10px;
           line-height:1.45;color:var(--ink-3)}
figcaption b{display:block;font-weight:600;color:var(--ink-2)}
figcaption b.hit{color:var(--signal);background:var(--signal-bg);
                 padding:0 3px;border-radius:2px;display:inline-block}
figure.carrier img{border-color:var(--signal);border-width:2px}
.ord{position:relative}
.ord::after{content:attr(data-ord);position:absolute;top:5px;left:5px;
            font-family:var(--mono);font-size:9px;background:rgba(16,19,24,.82);
            color:#fff;padding:1px 5px;border-radius:2px;letter-spacing:.06em}
.placeholder{font-family:var(--mono);font-size:11px;color:var(--ink-3);
             margin-top:8px;padding:7px 9px;background:var(--paper);
             border-radius:4px;border:1px solid var(--line)}

/* ── 우: 정답 원장 ─────────────────────────────────────────────── */
.side{border-left:1px solid var(--line);background:var(--card);
      position:sticky;top:0;height:100vh;overflow-y:auto;padding:18px 16px 40px}
.side h3{font-size:11px;letter-spacing:.14em;text-transform:uppercase;
         color:var(--ink-3);margin:0 0 4px;font-weight:500}
.side p.note{font-size:11px;color:var(--ink-3);margin:0 0 16px;line-height:1.5}
.axis{margin-bottom:18px}
.axis h4{font-family:var(--mono);font-size:11px;margin:0 0 7px;
         color:var(--ink-2);font-weight:500;letter-spacing:.04em}
.chips{display:flex;flex-wrap:wrap;gap:5px}
.chip{font-family:var(--mono);font-size:10.5px;padding:3px 7px;border-radius:3px;
      border:1px solid var(--line);color:var(--ink-3);background:var(--paper);
      cursor:default}
.chip.like{border-color:var(--keep);color:var(--keep);background:var(--keep-bg)}
.chip.dislike{border-color:var(--drop);color:var(--drop);background:var(--drop-bg)}
.chip u{text-decoration:none;opacity:.62;margin-left:4px;font-size:9px}
.chip.neutral{opacity:.5}
.bal{font-family:var(--mono);font-size:10.5px;color:var(--ink-3);
     border-top:1px solid var(--line);padding-top:12px;margin-top:4px}
.bal div{display:flex;justify-content:space-between;padding:1.5px 0}
.warn{margin-top:12px;padding:8px 10px;background:#fef3c7;border-radius:4px;
      font-family:var(--mono);font-size:10px;color:#92400e;line-height:1.5}

.hidden{display:none !important}
@media (max-width:1180px){
  .wrap{grid-template-columns:1fr}
  .rail,.side{position:static;height:auto;border:0;border-bottom:1px solid var(--line)}
  .rail{display:flex;flex-wrap:wrap;align-items:center}
  .rail h1,.rail .meta{width:100%}
  main{padding:20px}
}
@media (prefers-reduced-motion:no-preference){
  .turn{animation:in .18s ease both}
  @keyframes in{from{opacity:0;transform:translateY(3px)}to{opacity:1}}
}
</style>
</head>
<body>
<div class="wrap">
  <nav class="rail">
    <h1>Dialogues</h1>
    <div class="meta" id="railmeta"></div>
    <div id="ulist"></div>
  </nav>

  <main>
    <div class="head">
      <h2 id="title"></h2>
      <span class="sub" id="subtitle"></span>
    </div>
    <div class="toolbar">
      <button class="tog on" id="tVariant">variant: image</button>
      <button class="tog" id="tEv">evidence turns only</button>
      <button class="tog on" id="tDecode">decode images</button>
      <button class="tog on" id="tTruth">ground truth</button>
    </div>
    <div id="thread"></div>
  </main>

  <aside class="side">
    <h3>Preference ledger</h3>
    <p class="note">프로필의 취향 18개. 대화가 노출한 것에는 표현 방식이
      붙는다. 비어 있으면 커버리지 결함이다.</p>
    <div id="ledger"></div>
    <div class="bal" id="bal"></div>
    <div id="warn"></div>
  </aside>
</div>

<script>
const DATA = __DATA__;
const IMROOT = __IMROOT__;
const AXES = ["garment_category","color","pattern"];
const AXIS_WORD = __AXIS_WORD__;
const EP_ORDER = __EP_ORDER__;

let uid = DATA.users[0].user_id, variant = "image";
let evOnly = false, decode = true, truth = true;

const $ = s => document.querySelector(s);
const el = (t, c, txt) => { const n = document.createElement(t);
  if (c) n.className = c; if (txt !== undefined) n.textContent = txt; return n; };
const cur = () => DATA.dialogs[uid + "|" + variant];
const prof = () => DATA.users.find(u => u.user_id === uid);

function railRender(){
  $("#railmeta").textContent =
    DATA.users.length + " users · " + DATA.meta.template_version;
  const box = $("#ulist"); box.innerHTML = "";
  DATA.users.forEach(u => {
    const d = DATA.dialogs[u.user_id + "|image"];
    const n = el("div", "uitem" + (u.user_id === uid ? " on" : ""));
    n.append(u.user_id);
    const s = el("span", null, d.evidence.length + " ev");
    n.append(s);
    if (d.downgrades.length) { const f = el("i", "flag"); f.title =
      d.downgrades.length + " downgrade"; n.append(f); }
    n.onclick = () => { uid = u.user_id; render(); };
    box.append(n);
  });
}

function attrCaption(a, axis){
  const cap = el("figcaption");
  AXES.forEach(k => {
    const b = el("b", k === axis ? "hit" : null, a[k].replace(/_/g," "));
    cap.append(b);
  });
  return cap;
}

function turnNode(t, ev){
  const row = el("div", "turn " + t.speaker + (ev ? " ev " + ev.polarity : ""));
  row.append(el("div", "who", t.speaker === "user" ? "USER" : "AI"));
  const bub = el("div", "bub");

  if (ev && truth){
    const tag = el("div", "tag");
    tag.append(el("em", "pol-" + ev.polarity, ev.polarity));
    tag.append(el("em", null, ev.expression_type));
    tag.append(el("em", null, AXIS_WORD[ev.axis] + " = " + ev.value.replace(/_/g," ")));
    if (ev.requested_type !== ev.expression_type)
      tag.append(el("em", "dg", ev.requested_type + "→" + ev.expression_type));
    bub.append(tag);
  }
  bub.append(el("div", null, t.text));

  if (t.images.length){
    const shots = el("div", "shots");
    t.images.forEach((src, i) => {
      const a = ev ? ev.image_attributes[i] : null;
      const carrier = ev && (ev.expression_type.startsWith("share")
                             || ev.evidence_index === i);
      const fig = el("figure", carrier && truth ? "carrier" : null);
      if (ev && ev.expression_type === "differ"){
        fig.classList.add("ord");
        fig.dataset.ord = i === 0 ? "first" : "second";
      }
      const img = el("img");
      img.src = IMROOT + "/" + src; img.alt = src; img.loading = "lazy";
      img.onerror = () => fig.classList.add("miss");
      fig.append(img);
      fig.append(el("div", "alt", src));
      if (decode && a) fig.append(attrCaption(a, truth ? ev.axis : null));
      shots.append(fig);
    });
    bub.append(shots);
  } else if (ev && variant === "text"){
    const m = t.text.match(/\\{[^}]+\\}/g);
    if (m) bub.append(el("div", "placeholder", "placeholders: " + m.join("  ")));
  }
  row.append(bub);
  return row;
}

function render(){
  railRender();
  const d = cur(), p = prof();
  $("#title").textContent = d.dialog_id;
  $("#subtitle").textContent =
    d.turns.length + " turns · " + d.evidence.length + " evidence · " +
    d.turns.reduce((s,t)=>s+t.images.length,0) + " images · seed " + d.seed;
  $("#tVariant").textContent = "variant: " + variant;

  const evByTurn = {}; d.evidence.forEach(e => evByTurn[e.revealed_in_turn] = e);
  const th = $("#thread"); th.innerHTML = "";
  let ep = null;
  d.turns.forEach(t => {
    const e = evByTurn[t.turn_id];
    const owner = e ? e.episode_id : null;
    if (!e && !evOnly && t.turn_id % 2 === 0){
      // 에피소드 오프너: 다음 evidence 의 episode_id 로 구간을 연다
      const nxt = d.evidence.find(x => x.revealed_in_turn > t.turn_id);
      if (nxt && nxt.episode_id !== ep){
        ep = nxt.episode_id;
        const h = el("div", "ep");
        h.append(el("b", null, ep.replace(/_/g," ")));
        h.append(el("i")); h.append(el("time", null, t.time));
        th.append(h);
      }
    }
    if (evOnly && !e && !(evByTurn[t.turn_id - 1])) return;
    th.append(turnNode(t, e));
  });

  // 원장
  const lg = $("#ledger"); lg.innerHTML = "";
  const seen = {}; d.evidence.forEach(e => seen[e.axis + "|" + e.value] = e);
  AXES.forEach(ax => {
    const box = el("div", "axis");
    box.append(el("h4", null, ax.replace(/_/g," ")));
    const chips = el("div", "chips");
    ["likes","dislikes"].forEach(kind => {
      p.attrs[ax][kind].forEach(v => {
        const e = seen[ax + "|" + v];
        const c = el("span", "chip " + (e ? kind.slice(0,-1) : "neutral"));
        c.append(v.replace(/_/g," "));
        const u = el("u", null, e ? e.expression_type : "MISSING");
        c.append(u);
        c.title = e ? "turn " + e.revealed_in_turn : "대화에 노출되지 않음";
        chips.append(c);
      });
    });
    box.append(chips); lg.append(box);
  });

  const cnt = {};
  d.evidence.forEach(e => cnt[e.expression_type] = (cnt[e.expression_type]||0)+1);
  const bal = $("#bal"); bal.innerHTML = "";
  ["single","differ","share2","share3"].forEach(k => {
    const r = el("div"); r.append(el("span", null, k));
    r.append(el("span", null, String(cnt[k] || 0))); bal.append(r);
  });
  const w = $("#warn"); w.innerHTML = "";
  if (d.downgrades.length){
    const b = el("div", "warn");
    b.append(el("div", null, d.downgrades.length + " downgrade — 셀 부족으로 방식 완화"));
    d.downgrades.forEach(x => b.append(el("div", null, "· " + x)));
    w.append(b);
  }
}

$("#tVariant").onclick = e => { variant = variant === "image" ? "text" : "image";
  e.target.classList.toggle("on", variant === "image"); render(); };
$("#tEv").onclick = e => { evOnly = !evOnly; e.target.classList.toggle("on", evOnly); render(); };
$("#tDecode").onclick = e => { decode = !decode; e.target.classList.toggle("on", decode); render(); };
$("#tTruth").onclick = e => { truth = !truth; e.target.classList.toggle("on", truth); render(); };
document.addEventListener("keydown", ev => {
  if (ev.target.tagName === "BUTTON") return;
  const i = DATA.users.findIndex(u => u.user_id === uid);
  if (ev.key === "j" && i < DATA.users.length - 1) { uid = DATA.users[i+1].user_id; render(); }
  if (ev.key === "k" && i > 0) { uid = DATA.users[i-1].user_id; render(); }
  if (ev.key === "v") $("#tVariant").click();
});
render();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogs", type=Path, default=DATA_DIR / "dialogs")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--images-root", default="../images",
                    help="HTML 기준 이미지 상대 경로")
    args = ap.parse_args()
    out = args.out or (args.dialogs / "viewer.html")

    profiles = {}
    with open(DATA_DIR / "profiles" / "profiles.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            profiles[r["user_id"]] = r

    dialogs, users, version = {}, [], "?"
    for p in sorted(args.dialogs.glob("dlg__*.json")):
        d = json.loads(p.read_text())
        if "expression_type" not in (d.get("evidence") or [{}])[0]:
            continue
        dialogs[f'{d["user_id"]}|{d["variant"]}'] = {
            "dialog_id": d["dialog_id"], "seed": d["seed"],
            "downgrades": d.get("downgrades", []),
            "evidence": d["evidence"], "turns": d["turns"]}
        version = d.get("template_version", version)
    for uid in sorted({k.split("|")[0] for k in dialogs}):
        users.append({"user_id": uid,
                      "attrs": profiles[uid]["structured_attributes"]})
    if not users:
        raise SystemExit(f"no dialogs in {args.dialogs}")

    payload = {"meta": {"template_version": version}, "users": users,
               "dialogs": dialogs}
    html = (HTML
            .replace("__DATA__", json.dumps(payload, ensure_ascii=False))
            .replace("__IMROOT__", json.dumps(args.images_root))
            .replace("__AXIS_WORD__", json.dumps(AXIS_WORD))
            .replace("__EP_ORDER__", json.dumps([e[0] for e in EPISODES])))
    out.write_text(html, encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"{len(users)} users, {len(dialogs)} dialogs -> {out} ({kb:.0f} KB)")
    print(f"images root: {args.images_root}  (없으면 속성 타일로 대체)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())