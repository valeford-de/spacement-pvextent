# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Read-only, loopback training and annotation monitoring; never load pixels or weights."""
from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import parse_qs, urlsplit

MAX_LOG_BYTES = 8 * 1024 * 1024


def _json(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_LOG_BYTES:
        return None
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None


def _events(path: Path):
    if path.is_symlink() or not path.is_file():
        return []
    result = []
    try:
        with path.open('rb') as stream:
            if path.stat().st_size > MAX_LOG_BYTES:
                first = stream.readline(65536)
                stream.seek(-MAX_LOG_BYTES, os.SEEK_END)
                lines = [first, *stream.read(MAX_LOG_BYTES).splitlines()[1:]]
            else:
                lines = stream.read(MAX_LOG_BYTES).splitlines()
        for line in lines:
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    result.append(value)
            except ValueError:
                continue  # A final in-flight line is not a metric.
    except OSError:
        pass
    return result


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _review_count(directory: Path, schema: str):
    count, batches, latest = 0, 0, None
    if not directory.is_dir() or directory.is_symlink():
        return {"images": 0, "batches": 0, "latest": None}
    for path in sorted(directory.glob('batch-*.review.json'))[:80]:
        value = _json(path)
        if not isinstance(value, dict) or value.get('review_schema_version') != schema:
            continue
        samples = value.get('samples')
        if not isinstance(samples, list) or not 1 <= len(samples) <= 50:
            continue
        valid = [s for s in samples if isinstance(s, dict) and s.get('full_image_inspected') is True
                 and (schema != '1.2' or s.get('proposal_overlay_inspected') is True)]
        if len(valid) != len(samples):
            continue
        count += len(samples)
        batches += 1
        latest = max(latest or 0, path.stat().st_mtime)
    return {"images": count, "batches": batches, "latest": latest}


def snapshot(repo_dir: Path | str, selected_run: str = '') -> dict:
    """Read bounded JSON logs only; historical runs are never reported as active."""
    repo = Path(repo_dir).resolve(strict=True)
    artifacts = repo / 'artifacts'
    package = artifacts / 'nrw-training-bulk-mask-checks-v2'
    drafts = _review_count(artifacts / 'nrw-training-bulk-mask-reviews-v1/lane-a', '1.1')
    first, second = [_review_count(package / f'lane-{lane}', '1.2') for lane in ('a', 'b')]
    dataset = (_json(repo / 'data/pvextent-training-v4/quality-quarantine-report.json')
               or _json(repo / 'data/pvextent-training-v3/augmentation-report.json'))
    runs = []
    run_root = repo / 'runs'
    if run_root.is_dir() and not run_root.is_symlink():
        for directory in sorted(run_root.iterdir())[:128]:
            if not directory.is_dir() or directory.is_symlink():
                continue
            report = _json(directory / 'baseline-report.json')
            telemetry = _events(directory / 'telemetry.jsonl')
            history = _events(directory / 'history.jsonl')
            if not report and not history and not telemetry:
                continue
            config = _json(directory / 'config.json') or {}
            active = False
            last = telemetry[-1] if telemetry else {}
            pid = telemetry[0].get('pid') if telemetry else None
            if type(pid) is int and pid > 0 and last.get('event') not in ('run_completed', 'run_failed'):
                try:
                    os.kill(pid, 0)
                    active = time.time() - (directory / 'telemetry.jsonl').stat().st_mtime < 600
                except OSError:
                    pass
            state = 'completed' if isinstance(report, dict) and report.get('ok') is True else 'running' if active else 'inactive'
            epoch_rows = [e for e in history if e.get('event') == 'epoch']
            loss = [{'step': e.get('global_step'), 'value': _number(e.get('loss'))}
                    for e in telemetry if e.get('event') == 'train_batch' and _number(e.get('loss')) is not None]
            if not loss:
                loss = [{'step': e.get('global_step'), 'value': _number(e.get('train_loss'))}
                        for e in epoch_rows if _number(e.get('train_loss')) is not None]
            # Bound response size while retaining the newest actual point.
            stride = max(1, math.ceil(len(loss) / 1200))
            loss = loss[::stride] + (loss[-1:] if len(loss) > 1 and (len(loss)-1) % stride else [])
            validation = [{'epoch': e.get('epoch'),
                           'iou': _number((e.get('validation') or {}).get('iou')),
                           'dice': _number((e.get('validation') or {}).get('dice'))} for e in epoch_rows]
            runs.append({'id': directory.name, 'state': state, 'epochs': (report or {}).get('total_epochs', config.get('epochs')),
                         'epoch': last.get('epoch', epoch_rows[-1].get('epoch') if epoch_rows else 0),
                         'global_step': last.get('global_step', (report or {}).get('global_step', 0)),
                         'batch': last.get('batch'), 'batches_per_epoch': last.get('batches_per_epoch'),
                         'loss': loss, 'validation': validation,
                         'best_iou': _number(((report or {}).get('best_val') or {}).get('iou')),
                         'device': (report or {}).get('device', last.get('device', telemetry[0].get('device') if telemetry else None))})
    active_runs = [r for r in runs if r['state'] == 'running']
    chosen = next((r for r in runs if r['id'] == selected_run), None) if selected_run else (active_runs[-1] if active_runs else None)
    return {'updated_at': datetime.now(timezone.utc).isoformat(), 'new_training_running': bool(active_runs),
            'phase': 'training' if active_runs else 'dataset_ready' if dataset and dataset.get('ok') else 'mask_checks',
            'target_images': 1000, 'drafts': drafts, 'check_c': first, 'check_d': second,
            'dataset': {'ready': bool(dataset and dataset.get('ok')),
                        'train_samples': dataset.get('effective_supervised_train_records') if dataset else None,
                        'added_samples': dataset.get('qualified_records_added_to_train') if dataset else None},
            'runs': [{k: r[k] for k in ('id', 'state')} for r in runs], 'selected_run': chosen}


HTML = r'''<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PVExtent · Live</title>
<style>
:root{color-scheme:dark;--bg:#0b0d0c;--fg:#eaf0ed;--muted:#a3ada8;--line:#29352e;--pv:#00a67d;--blue:#86b8ef}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:30px 24px}header{display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap}h1{font-size:26px;font-weight:500;margin:0}h2{font-size:18px;font-weight:500}p{color:var(--muted)}.state{color:var(--pv)}.progress-grid,.charts{display:grid;grid-template-columns:repeat(3,1fr);gap:28px;margin:30px 0}.charts{grid-template-columns:1fr 1fr}.count{font-size:26px;font-variant-numeric:tabular-nums}.track{height:6px;background:#202823;margin-top:12px;overflow:hidden}.fill{height:100%;background:var(--pv);transition:width .4s}small{color:var(--muted)}section{margin-top:32px;border-top:1px solid var(--line);padding-top:18px}select{background:#18201c;color:var(--fg);padding:9px;border:1px solid #39463e;border-radius:5px;font:inherit;max-width:100%}.toolbar{display:flex;gap:18px;align-items:center;flex-wrap:wrap}svg{width:100%;height:250px;overflow:visible}.axis{stroke:var(--line);stroke-width:1}.loss{stroke:var(--pv);fill:none;stroke-width:2}.iou{stroke:var(--pv);fill:none;stroke-width:2}.dice{stroke:var(--blue);fill:none;stroke-width:2;stroke-dasharray:5 3}svg text{fill:var(--muted);font:12px system-ui}a{color:var(--pv)}.empty{min-height:200px;display:grid;place-content:center;color:var(--muted)}#error{color:#ffb3a5}.legend{display:flex;gap:18px;font-size:13px}.legend span:first-child{color:var(--pv)}.legend span:last-child{color:var(--blue)}@media(max-width:700px){.progress-grid,.charts{grid-template-columns:1fr}main{padding:20px}.count{font-size:22px}}
</style><main><header><div><h1>spacement PVExtent</h1><div class="state" id="phase">Livefortschritt</div></div><small id="updated">Verbindung wird aufgebaut</small></header><p id="summary">Aktuelle Verarbeitung wird geladen.</p>
<div class="progress-grid"><div><small>Maskenvorschläge</small><div class="count" id="draft-count">—</div><div class="track"><div class="fill" id="draft-fill"></div></div></div><div><small>Unabhängige Prüfung C</small><div class="count" id="c-count">—</div><div class="track"><div class="fill" id="c-fill"></div></div></div><div><small>Unabhängige Prüfung D</small><div class="count" id="d-count">—</div><div class="track"><div class="fill" id="d-fill"></div></div></div></div>
<section><div class="toolbar"><h2>Training</h2><select id="run" aria-label="Trainingslauf"><option value="">Live / nächster Lauf</option></select><span id="run-status"></span></div><p id="run-detail"></p><div class="charts"><div><h2>Training-Loss</h2><div id="loss-chart"></div></div><div><h2>Interne Validierung</h2><div class="legend"><span>IoU · durchgezogen</span><span>Dice · gestrichelt</span></div><div id="val-chart"></div></div></div><small>Messpunkte stammen aus den tatsächlichen Laufprotokollen. Validierungswerte beschreiben den internen Datensatz.</small></section><p id="error" role="status"></p></main>
<script>
const el=id=>document.getElementById(id);let selection='',runOptionsKey='';el('run').addEventListener('change',e=>{selection=e.target.value;update()});
const finite=v=>typeof v==='number'&&Number.isFinite(v);const fmt=v=>new Intl.NumberFormat('de-DE').format(v);const esc=s=>String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function chart(target,series,xkey,ykeys,labels,percent=false){const rows=series.filter(r=>finite(r[xkey])&&ykeys.some(k=>finite(r[k])));if(!rows.length){el(target).innerHTML='<div class="empty">Noch keine Messpunkte</div>';return}const w=Math.max(240,el(target).clientWidth),h=250,l=52,r=18,t=15,b=42;let xmin=Math.min(...rows.map(x=>x[xkey])),xmax=Math.max(...rows.map(x=>x[xkey]));if(xmax===xmin)xmax=xmin+1;let values=rows.flatMap(row=>ykeys.map(k=>row[k]).filter(finite));let ymin=percent?Math.max(0,Math.min(...values)-.01):Math.min(...values)*.95,ymax=percent?Math.min(1,Math.max(...values)+.01):Math.max(...values)*1.05;if(ymax===ymin)ymax=ymin+1;const X=v=>l+(v-xmin)/(xmax-xmin)*(w-l-r),Y=v=>h-b-(v-ymin)/(ymax-ymin)*(h-t-b);let content='';for(let i=0;i<4;i++){let v=ymin+(ymax-ymin)*i/3,y=Y(v);content+=`<line class="axis" x1="${l}" x2="${w-r}" y1="${y}" y2="${y}"/><text x="${l-8}" y="${y+4}" text-anchor="end">${percent?(100*v).toFixed(1)+'%':v.toFixed(3)}</text>`}for(let i=0;i<3;i++){let v=xmin+(xmax-xmin)*i/2;content+=`<text x="${X(v)}" y="${h-b+22}" text-anchor="middle">${Math.round(v)}</text>`}ykeys.forEach((key,index)=>{const points=rows.filter(v=>finite(v[key]));content+=`<polyline class="${labels[index]}" points="${points.map(v=>X(v[xkey])+','+Y(v[key])).join(' ')}"/>`;for(const p of points)content+=`<circle cx="${X(p[xkey])}" cy="${Y(p[key])}" r="2" fill="${index?'var(--blue)':'var(--pv)'}"><title>${xkey==='step'?'Schritt':'Epoche'} ${p[xkey]} · ${key}: ${p[key].toFixed(5)}</title></circle>`});content+=`<text x="${(l+w-r)/2}" y="${h-2}" text-anchor="middle">${xkey==='step'?'Optimierungsschritt':'Epoche'}</text>`;el(target).innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(target)}">${content}</svg>`}
async function update(){try{const res=await fetch('/status.json?run='+encodeURIComponent(selection),{cache:'no-store'});if(!res.ok)throw new Error('Status nicht verfügbar');const s=await res.json();el('phase').textContent={mask_checks:'Maskenprüfung läuft',dataset_ready:'Trainingsdatensatz vorbereitet',training:'Training läuft'}[s.phase];el('summary').textContent=s.new_training_running?'Der aktuelle Trainingslauf liefert echte Livewerte.':s.dataset.ready?`${fmt(s.dataset.train_samples)} Trainingsbilder vorbereitet · ${fmt(s.dataset.added_samples)} neue geprüfte Beispiele.`:'Der neue Trainingslauf hat noch nicht begonnen. Aktuell werden die Masken geprüft.';for(const [prefix,value]of[['draft',s.drafts],['c',s.check_c],['d',s.check_d]]){el(prefix+'-count').textContent=fmt(value.images)+' / 1.000';el(prefix+'-fill').style.width=Math.min(100,value.images/10)+'%'}const optionsKey=JSON.stringify(s.runs);if(optionsKey!==runOptionsKey){runOptionsKey=optionsKey;el('run').innerHTML='<option value="">Live / nächster Lauf</option>'+s.runs.map(r=>`<option value="${esc(r.id)}">${esc(r.id)} · ${r.state==='completed'?'abgeschlossen':r.state==='running'?'läuft':'inaktiv'}</option>`).join('');el('run').value=selection;}const run=s.selected_run;el('run-status').textContent=run?(run.state==='completed'?'Abgeschlossener Lauf':run.state==='running'?'Live':'Inaktiver Lauf'):'Kein aktiver Trainingslauf';el('run-detail').textContent=run?`Epoche ${run.epoch??'—'} / ${run.epochs??'—'} · Schritt ${fmt(run.global_step??0)}${run.device?' · '+run.device:''}`:'Die Kurven erscheinen beim nächsten Trainingslauf automatisch. Vorherige Läufe sind separat auswählbar.';chart('loss-chart',run?.loss??[],'step',['value'],['loss']);chart('val-chart',run?.validation??[],'epoch',['iou','dice'],['iou','dice'],true);el('updated').textContent='Aktualisiert '+new Date(s.updated_at).toLocaleTimeString('de-DE');el('error').textContent=''}catch(e){el('error').textContent=e.message+' · erneuter Versuch in 2 Sekunden'}}update();setInterval(update,2000);
</script></html>'''


def serve(repo_dir: Path | str, *, port: int = 0):
    """Serve only a fixed UI and read-only JSON on the loopback interface."""
    repo = Path(repo_dir).resolve(strict=True)
    if not 0 <= port <= 65535:
        raise ValueError('port must be in 0..65535')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == '/':
                payload, kind = HTML.encode(), 'text/html; charset=utf-8'
            elif parsed.path == '/status.json':
                try:
                    selected = parse_qs(parsed.query).get('run', [''])[0]
                    if len(selected) > 256:
                        raise ValueError('run identifier too long')
                    payload, kind = json.dumps(snapshot(repo, selected), allow_nan=False).encode(), 'application/json'
                except (OSError, ValueError, TypeError):
                    self.send_error(503, 'Status temporarily unavailable')
                    return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *_):
            pass
    with ThreadingHTTPServer(('127.0.0.1', port), Handler) as server:
        print(f'PVExtent live: http://127.0.0.1:{server.server_port}/', flush=True)
        server.serve_forever()
