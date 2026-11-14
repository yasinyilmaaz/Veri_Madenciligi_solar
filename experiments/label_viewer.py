"""Etiket görüntüleyici: eğitim görüntülerini anotatör poligonları, omurgalar ve model tahminiyle tarayıcıda gösterir.

Yalnızca standart kütüphane (http.server) + projenin kendi okuyucuları. Yerel makinede çalışır, dışarı açılmaz.
Çalıştırma: PYTHONPATH=src python experiments/label_viewer.py  ->  http://127.0.0.1:8765
Model tahmini katmanı, OOF olasılıklarından (outputs/oof/<ens>_tta2) prob > 0,4 konturlarıdır; yani her görüntü
için tahmin, o görüntüyü eğitimde hiç görmemiş fold modelinden gelir.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np

from data import ANN_FILE, OUT, TRAIN_IMG, get_folds

CATS = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}
PRED_DIRS = ["ens1536_tta2", "ens4_tta2"]  # ilk bulunan kullanılır


def load_index():
    d = json.loads(ANN_FILE.read_text())
    folds = get_folds().set_index("stem")["fold"].to_dict()
    recs, meta = {}, {}
    for im in d["images"]:
        stem = Path(im["file_name"]).stem
        recs.setdefault(stem, {})[im["id"]] = []
        meta[stem] = im["date_captured"]
    for a in d["annotations"]:
        stem = a["image_id"].split("-", 1)[1]
        recs[stem][a["image_id"]].append({"seg": a["segmentation"], "spine": a.get("spine"),
                                          "cat": CATS.get(a["category_id"], str(a["category_id"])),
                                          "area": a.get("area")})
    rows = [dict(stem=s, date=meta[s], station=s[-2], year=int(s[:4]), fold=int(folds.get(s, -1)),
                 n_annot=len(r), n_fil=sum(len(v) for v in r.values()),
                 per_annot=[len(v) for v in r.values()]) for s, r in sorted(recs.items())]
    return recs, rows


def pred_polygons(stem, fold, thr=0.4, min_area=100):
    for name in PRED_DIRS:
        p = OUT / "oof" / name / f"fold{fold}" / "val_probs" / f"{stem}.npy"
        if p.exists():
            prob = np.load(p)
            m = (prob > thr * 255).astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys = [c.reshape(-1).tolist() for c in cs if cv2.contourArea(c) >= min_area]
            return {"source": name, "polys": polys}
    return {"source": None, "polys": []}


class Handler(BaseHTTPRequestHandler):
    recs, rows = None, None

    def _send(self, body, ctype="application/json", code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(self.path.split("?")[0])
        if path == "/":
            return self._send(PAGE, "text/html; charset=utf-8")
        if path == "/api/list":
            return self._send(self.rows)
        stem = path.rsplit("/", 1)[-1]
        if stem not in self.recs:  # yalnızca bilinen dosya adları; yol gezintisi yok
            return self._send({"error": "bulunamadı"}, code=404)
        if path.startswith("/api/ann/"):
            return self._send(self.recs[stem])
        if path.startswith("/api/pred/"):
            fold = next(r["fold"] for r in self.rows if r["stem"] == stem)
            return self._send(pred_polygons(stem, fold))
        if path.startswith("/img/"):
            return self._send((TRAIN_IMG / f"{stem}.jpeg").read_bytes(), "image/jpeg")
        return self._send({"error": "bulunamadı"}, code=404)

    def log_message(self, *a):
        pass


PAGE = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><title>Filament Etiketleri</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#111418;--panel:#1a1f26;--line:#2a313b;--text:#e6e9ee;--muted:#9aa4b2;--accent:#5aa9ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,sans-serif;display:flex;height:100vh;overflow:hidden}
#side{width:300px;min-width:300px;background:var(--panel);border-right:1px solid var(--line);display:flex;flex-direction:column}
#filters{padding:10px;border-bottom:1px solid var(--line);display:grid;gap:6px}
#filters input,#filters select{width:100%;background:#0d1015;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px}
#list{overflow:auto;flex:1}.item{padding:6px 10px;border-bottom:1px solid var(--line);cursor:pointer;font-size:13px}
.item:hover{background:#222833}.item.sel{background:#24344a}.item small{color:var(--muted)}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#bar{padding:8px 12px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:12px;align-items:center;background:var(--panel)}
#bar label{display:flex;gap:4px;align-items:center;cursor:pointer}.sw{width:12px;height:12px;border-radius:3px;display:inline-block}
#info{color:var(--muted);font-size:13px}
#view{flex:1;overflow:hidden;position:relative;cursor:grab;background:#000}
#stage{position:absolute;transform-origin:0 0}#stage img,#stage svg{position:absolute;left:0;top:0;width:2048px;height:2048px}
#tip{position:fixed;pointer-events:none;background:#000c;padding:4px 8px;border-radius:6px;font-size:12px;display:none}
button{background:#0d1015;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
</style></head><body>
<div id="side"><div id="filters">
<input id="q" placeholder="Ara (tarih / dosya adı)">
<select id="fst"><option value="">Tüm istasyonlar</option></select>
<select id="fan"><option value="">Tüm anotatör sayıları</option><option>1</option><option>2</option><option>3</option></select>
<div id="cnt" style="color:var(--muted);font-size:12px"></div></div><div id="list"></div></div>
<div id="main"><div id="bar">
<button id="prev">◀</button><button id="next">▶</button><button id="fit">Sığdır</button>
<span id="annots"></span>
<label><input type="checkbox" id="spine" checked>Omurga</label>
<label><input type="checkbox" id="fill">Dolgu</label>
<label><input type="checkbox" id="pred"><span class="sw" style="background:#ffffff"></span>Model tahmini (OOF)</label>
<label>Parlaklık <input type="range" id="bri" min="50" max="250" value="100"></label>
<span id="info"></span></div>
<div id="view"><div id="stage"><img id="img"><svg id="svg" viewBox="0 0 2048 2048"></svg></div></div></div>
<div id="tip"></div>
<script>
const COLORS=["#ff5a5a","#4ade80","#5aa9ff","#fbbf24","#c084fc"];
let rows=[],shown=[],cur=-1,ann={},pred=null,hidden=new Set(),scale=1,tx=0,ty=0;
const $=id=>document.getElementById(id);
fetch('/api/list').then(r=>r.json()).then(d=>{rows=d;
 [...new Set(d.map(r=>r.station))].sort().forEach(s=>$('fst').add(new Option('İstasyon '+s,s)));
 filter();});
function filter(){const q=$('q').value.toLowerCase(),st=$('fst').value,an=$('fan').value;
 shown=rows.filter(r=>(!q||r.stem.toLowerCase().includes(q)||r.date.includes(q))&&(!st||r.station==st)&&(!an||r.n_annot==an));
 $('cnt').textContent=shown.length+' görüntü (toplam '+rows.length+')';
 $('list').innerHTML=shown.map((r,i)=>`<div class="item" data-i="${i}">${r.date}<br><small>${r.stem} · ${r.n_annot} anotatör · ${r.per_annot.join('/')} filament · fold ${r.fold}</small></div>`).join('');
 [...document.querySelectorAll('.item')].forEach(e=>e.onclick=()=>open(+e.dataset.i));
 if(shown.length)open(0);}
['q','fst','fan'].forEach(id=>$(id).oninput=filter);
async function open(i){if(i<0||i>=shown.length)return;cur=i;const r=shown[i];
 document.querySelectorAll('.item').forEach(e=>e.classList.toggle('sel',+e.dataset.i==i));
 document.querySelector('.item.sel')?.scrollIntoView({block:'nearest'});
 $('img').src='/img/'+r.stem;ann=await (await fetch('/api/ann/'+r.stem)).json();pred=null;hidden.clear();
 if($('pred').checked)pred=await (await fetch('/api/pred/'+r.stem)).json();
 $('annots').innerHTML=Object.keys(ann).map((a,k)=>`<label><input type="checkbox" checked data-a="${a}"><span class="sw" style="background:${COLORS[k%5]}"></span>${a.split('-')[0]} (${ann[a].length})</label>`).join(' ');
 document.querySelectorAll('#annots input').forEach(e=>e.onchange=()=>{e.checked?hidden.delete(e.dataset.a):hidden.add(e.dataset.a);draw();});
 $('info').textContent=r.date+' · istasyon '+r.station;draw();}
function pts(a){let s='';for(let i=0;i<a.length;i+=2)s+=a[i].toFixed(1)+','+a[i+1].toFixed(1)+' ';return s;}
function draw(){let h='';const fill=$('fill').checked;
 Object.keys(ann).forEach((a,k)=>{if(hidden.has(a))return;const c=COLORS[k%5];
  ann[a].forEach((f,j)=>{(f.seg||[]).forEach(p=>h+=`<polygon points="${pts(p)}" fill="${fill?c+'55':'none'}" stroke="${c}" stroke-width="${1.6/Math.sqrt(scale)}" data-t="${a.split('-')[0]} · #${j+1} · ${f.cat} · alan ${Math.round(f.area||0)} px"/>`);
   if($('spine').checked&&f.spine&&f.spine.length>3)h+=`<polyline points="${pts(f.spine)}" fill="none" stroke="${c}" stroke-dasharray="6 4" stroke-width="${1.2/Math.sqrt(scale)}" opacity=".8"/>`;});});
 if(pred&&$('pred').checked)pred.polys.forEach(p=>h+=`<polygon points="${pts(p)}" fill="none" stroke="#fff" stroke-width="${1.4/Math.sqrt(scale)}" stroke-dasharray="3 3" data-t="model tahmini (${pred.source})"/>`);
 $('svg').innerHTML=h;}
['spine','fill'].forEach(id=>$(id).onchange=draw);
$('pred').onchange=async()=>{if($('pred').checked&&!pred&&cur>=0)pred=await (await fetch('/api/pred/'+shown[cur].stem)).json();draw();};
$('bri').oninput=()=>$('img').style.filter=`brightness(${$('bri').value}%)`;
$('prev').onclick=()=>open(cur-1);$('next').onclick=()=>open(cur+1);
document.onkeydown=e=>{if(e.target.tagName=='INPUT'&&e.target.type!='checkbox'&&e.target.type!='range')return;
 if(e.key=='ArrowRight'||e.key=='d')open(cur+1);if(e.key=='ArrowLeft'||e.key=='a')open(cur-1);};
function apply(){$('stage').style.transform=`translate(${tx}px,${ty}px) scale(${scale})`;}
function fit(){const v=$('view').getBoundingClientRect();scale=Math.min(v.width,v.height)/2048;tx=(v.width-2048*scale)/2;ty=(v.height-2048*scale)/2;apply();draw();}
$('fit').onclick=fit;window.onresize=fit;setTimeout(fit,50);
$('view').onwheel=e=>{e.preventDefault();const v=$('view').getBoundingClientRect(),mx=e.clientX-v.left,my=e.clientY-v.top,k=e.deltaY<0?1.2:1/1.2;
 tx=mx-(mx-tx)*k;ty=my-(my-ty)*k;scale*=k;apply();draw();};
let drag=null;$('view').onmousedown=e=>{drag=[e.clientX-tx,e.clientY-ty];$('view').style.cursor='grabbing';};
window.onmouseup=()=>{drag=null;$('view').style.cursor='grab';};
window.onmousemove=e=>{if(drag){tx=e.clientX-drag[0];ty=e.clientY-drag[1];apply();}
 const t=e.target.dataset?.t;$('tip').style.display=t?'block':'none';if(t){$('tip').textContent=t;$('tip').style.left=e.clientX+12+'px';$('tip').style.top=e.clientY+12+'px';}};
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    Handler.recs, Handler.rows = load_index()
    print(f"{len(Handler.rows)} etiketli görüntü. Tarayıcıda açın: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
