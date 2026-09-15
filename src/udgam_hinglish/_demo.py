#!/usr/bin/env python3
"""Private localhost parse preview; no actions, request-time downloads, or input logs."""
from __future__ import annotations

import argparse
import http.server
import json
import os
from pathlib import Path
import resource
import secrets
import socket
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
from ._vendor import runtime
from ._vendor.prompts import derive_schema, make_prompt
from ._vendor.retrieval import Retriever
from ._vendor.top import inspect_top, parse_top, to_json_tree, walk_nodes

MAX_BODY_BYTES = 16384
MAX_TEXT_CHARS = 2000
LIMITATION = (
    "Experimental preview for alarm, event, messaging, music, navigation, reminder, "
    "timer and weather commands in Romanized Hinglish or English. Other requests "
    "may be misclassified; the model has no reliable out-of-domain detector. "
    "A valid tree and copied words do not prove the interpretation is correct. "
    "No message is sent, reminder created, or other action executed."
)


class RequestError(ValueError):
    def __init__(self, message, *, status=400, code="invalid_request"):
        super().__init__(message)
        self.status, self.code = status, code


def artifact_size(path):
    """Logical local file bytes; not a claim about runtime memory or downloads."""
    if path is None:
        return None
    path = Path(path)
    files = [path] if path.is_file() else path.rglob("*")
    seen, total = set(), 0
    for file in files:
        if file.is_file() and file.resolve() not in seen:
            seen.add(file.resolve())
            total += file.stat().st_size
    return total


def process_peak_rss_bytes():
    amount = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(amount if sys.platform == "darwin" else amount * 1024)


def model_tensor_bytes(model):
    return sum(value.numel() * value.element_size() for value in model.parameters()) + sum(
        value.numel() * value.element_size() for value in model.buffers())


class NearestEngine:
    id = "nearest"
    label = "Nearest-neighbour baseline · saved training example"
    kind = "retrieval_baseline"

    def __init__(self, retriever, train_path):
        self.retriever = retriever
        self.info = {"id": self.id, "label": self.label, "kind": self.kind,
                     "is_fine_tuned_model": False, "device": "cpu",
                     "artifact_bytes": artifact_size(train_path),
                     "artifact_scope": "retrieval training file only",
                     "tensor_bytes": None, "shots": None,
                     "description": "Returns the closest saved training tree unchanged. It does not generate or adapt slot values."}

    def predict(self, text):
        found = self.retriever.retrieve(text, 1)
        if not found:
            raise RequestError("No training example is available.", code="empty_retrieval")
        example = found[0]
        return {"prediction": example["target"], "truncated": False,
                "reference": {"id": example["id"], "text": example["text"],
                              "similarity": example["retrieval_score"]},
                "example_ids": [example["id"]]}


class ModelEngine:
    def __init__(self, *, model, tokenizer, schema, retriever, shots, engine_id,
                 label, source, artifact_bytes=None, artifact_scope=None,
                 max_new_tokens=768, max_input_tokens=4096):
        self.id, self.label = engine_id, label
        self.kind = "fine_tuned_model" if engine_id == "fine_tuned" else "original_model"
        self.model, self.tokenizer = model, tokenizer
        self.schema, self.retriever, self.shots = schema, retriever, shots
        self.max_new_tokens, self.max_input_tokens = max_new_tokens, max_input_tokens
        self.info = {"id": self.id, "label": label, "kind": self.kind,
                     "is_fine_tuned_model": self.kind == "fine_tuned_model",
                     "source": source, "base_model": runtime.BASE_MODEL,
                     "base_revision": runtime.BASE_REVISION, "device": str(model.device),
                     "dtype": str(model.dtype),
                     "shots": shots, "artifact_bytes": artifact_bytes,
                     "artifact_scope": artifact_scope,
                     "tensor_bytes": model_tensor_bytes(model),
                     "max_new_tokens": max_new_tokens, "max_input_tokens": max_input_tokens,
                     "description": "Greedy, non-thinking parsing using the experiment's shared prompt."}

    def predict(self, text):
        prompt, ids = make_prompt(self.tokenizer, {"text": text}, self.schema,
                                  self.retriever, self.shots)
        output = runtime.generate_batch(self.model, self.tokenizer, [prompt],
                                        self.max_new_tokens, self.max_input_tokens)[0]
        return {**output, "example_ids": ids}


class DemoApp:
    """Only preloaded engines; aggregate measurements never retain user text."""
    def __init__(self, engines, schema, examples=()):
        if not engines or len({engine.id for engine in engines}) != len(engines):
            raise ValueError("Demo needs one or more distinct, preloaded engines")
        self.engines = {engine.id: engine for engine in engines}
        self.schema, self.labels = schema, set(schema["labels"])
        self.examples = list(examples)
        self._generation_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._metrics = {key: {"requests": 0, "valid_previews": 0,
                               "last_latency_ms": None, "total_latency_ms": 0.0}
                         for key in self.engines}

    def status(self):
        return {"name": "Udgam Hinglish Commands", "experimental": True,
                "publication": "See the installed model package release terms",
                "executes_actions": False, "request_time_network_calls": False,
                "logs_user_input": False, "limitation": LIMITATION,
                "max_text_characters": MAX_TEXT_CHARS,
                "engines": [engine.info for engine in self.engines.values()],
                "examples": self.examples}

    def metrics(self):
        with self._metrics_lock:
            measures = {key: dict(value) for key, value in self._metrics.items()}
        for value in measures.values():
            value["mean_latency_ms"] = (value["total_latency_ms"] / value["requests"]
                                        if value["requests"] else None)
        return {"engines": measures, "process_peak_rss_bytes": process_peak_rss_bytes(),
                "measurement_scope": "This process since startup; not a benchmark or an accuracy estimate.",
                "artifacts": [{"id": engine.id, "artifact_bytes": engine.info.get("artifact_bytes"),
                               "artifact_scope": engine.info.get("artifact_scope"),
                               "tensor_bytes": engine.info.get("tensor_bytes")}
                              for engine in self.engines.values()]}

    def parse(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"text", "engine"}:
            raise RequestError("Send exactly text and engine fields.")
        text, engine_id = payload["text"], payload["engine"]
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
            raise RequestError(f"Enter a nonempty command of at most {MAX_TEXT_CHARS} characters.")
        if not isinstance(engine_id, str) or engine_id not in self.engines:
            raise RequestError("Choose a loaded engine; no replacement will be selected automatically.", code="unknown_engine")
        if not self._generation_lock.acquire(blocking=False):
            raise RequestError("A local prediction is already running. Try again when it finishes.", status=429, code="busy")
        started, preview_ok = time.perf_counter(), False
        try:
            engine = self.engines[engine_id]
            output = engine.predict(text)
            prediction = output.get("prediction")
            if not isinstance(prediction, str):
                raise RequestError("The loaded engine returned an invalid response.", status=500, code="engine_error")
            checks = inspect_top(prediction, text)
            labels = {node.label for node in walk_nodes(parse_top(prediction))} if checks["syntax_valid"] else set()
            unknown_labels = sorted(labels - self.labels)
            truncated = bool(output.get("truncated", False))
            if truncated:
                code, message = "truncated", "Generation stopped before its end marker. No structured preview is accepted."
            elif output.get("error"):
                code, message = "generation_error", "The engine reported a generation error. No structured preview is accepted."
            elif not checks["syntax_valid"]:
                code, message = "malformed", "The output is not one valid TOP tree. It has not been repaired."
            elif unknown_labels:
                code, message = "unknown_labels", "The output uses labels outside the training schema."
            elif not checks["source_valid"]:
                code, message = "source_mismatch", "Some output words are absent from this command. No structured preview is accepted."
            else:
                code, message = "preview", "Structure and source-text checks passed. Check the interpretation; no action was executed."
                preview_ok = True
            return {"ok": preview_ok, "status": code, "message": message,
                    "engine": engine.info, "raw_top": prediction,
                    "tree": to_json_tree(prediction) if preview_ok else None,
                    "checks": {**checks, "unknown_labels": unknown_labels, "truncated": truncated,
                               "generation_error": bool(output.get("error"))},
                    "reference": output.get("reference"), "example_ids": output.get("example_ids", []),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "input_tokens": output.get("input_tokens"),
                    "generated_tokens": output.get("generated_tokens"),
                    "executed_action": False, "limitation": LIMITATION}
        except RequestError:
            raise
        except ValueError:
            # The shared runtime rejects overlong prompts instead of truncating.
            raise RequestError("The local engine could not process this command within its configured limits.", code="generation_rejected") from None
        except Exception:
            # No raw exception text: it can include input text or local paths.
            raise RequestError("Local inference failed. No action was executed and no alternative engine was used.", status=500, code="engine_error") from None
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            with self._metrics_lock:
                metrics = self._metrics[engine_id]
                metrics["requests"] += 1
                metrics["valid_previews"] += int(preview_ok)
                metrics["last_latency_ms"] = round(elapsed, 3)
                metrics["total_latency_ms"] += elapsed
            self._generation_lock.release()


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Udgam · Hinglish Commands</title><style nonce="__NONCE__">
:root{color-scheme:light;--ink:#182724;--muted:#59675f;--line:#d9e3dc;--green:#0b6448;--paper:#fafcf9;--tint:#edf4ee}
*{box-sizing:border-box}body{margin:0;background:var(--paper);font:16px/1.55 system-ui,sans-serif;color:var(--ink)}
main{max-width:1120px;margin:0 auto;padding:36px 24px 60px}header{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:42px}.brand{font-size:14px;font-weight:750;letter-spacing:.1em}.badge{font-size:12px;padding:5px 10px;border:1px solid var(--line);border-radius:20px;color:var(--muted)}
h1{font-size:clamp(30px,5vw,48px);line-height:1.12;letter-spacing:-.04em;margin:0 0 16px;max-width:760px}h2{font-size:18px;margin:0 0 15px}.intro{max-width:750px;color:var(--muted);margin:0 0 30px}.layout{display:grid;grid-template-columns:1fr 1fr;gap:22px;align-items:start}.panel{background:white;border:1px solid var(--line);border-radius:18px;padding:24px}.label{display:block;font-size:13px;font-weight:650;margin:0 0 8px}select,textarea{width:100%;font:inherit;border:1px solid #b9c9be;border-radius:9px;background:white;color:var(--ink);padding:12px}textarea{resize:vertical;min-height:160px;line-height:1.55}select{font-size:14px}textarea:focus,select:focus,button:focus-visible{outline:3px solid #b5d7c6;outline-offset:2px}.field{margin-bottom:20px}.hint{font-size:12px;color:var(--muted);margin:7px 0 0}.actions{display:flex;flex-wrap:wrap;gap:10px;align-items:center}button{cursor:pointer;border:0;border-radius:9px;padding:11px 16px;font:600 14px system-ui;background:var(--green);color:white}button.secondary{background:var(--tint);color:var(--green)}button:disabled{opacity:.5;cursor:wait}.notice{font-size:13px;padding:12px 14px;border-radius:9px;background:var(--tint);margin-bottom:16px}.notice.warn{background:#fff2dd;color:#704a12}.quiet{color:var(--muted);font-size:13px}.empty{padding:35px 8px;color:var(--muted);text-align:center}.resultmeta{font-size:12px;color:var(--muted);margin-bottom:12px}.tree,.tree ul{list-style:none;padding-left:16px;border-left:1px solid #c6d7cb}.tree{padding-left:0;border:0;margin:0}.tree li{margin:10px 0}.node{display:inline-block;font-size:12px;font-weight:650;color:var(--green);background:var(--tint);border-radius:5px;padding:3px 7px}.leaf{font-size:14px;overflow-wrap:anywhere;margin:6px 0}pre{font:12px/1.6 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f3;border-radius:8px;padding:14px;max-height:400px;overflow:auto}details{margin-top:18px}summary{font-size:13px;cursor:pointer;color:var(--muted)}footer{margin-top:24px;font-size:12px;color:var(--muted)}.bottom{margin-top:22px}.sr{position:absolute;width:1px;height:1px;padding:0;overflow:hidden;clip:rect(0,0,0,0)}[hidden]{display:none!important}@media(max-width:760px){main{padding:24px 16px 40px}header{margin-bottom:28px}.layout{grid-template-columns:1fr}.panel{padding:20px}}
</style></head><body><main>
<header><span class="brand">UDGAM LABS</span><span class="badge">Private local experiment</span></header>
<h1>A small command.<br>A structured interpretation.</h1>
<p class="intro">Explore how a Hinglish or English command becomes a nested tree. This is a preview tool: it never sends a message, sets an alarm, or takes an action.</p>
<div class="layout"><section class="panel" aria-labelledby="input-title"><h2 id="input-title">Try a command</h2>
<form id="form"><div class="field"><label class="label" for="engine">Loaded engine</label><select id="engine" disabled></select><p class="hint" id="engine-note"></p></div>
<div class="field"><label class="label" for="text">Your command</label><textarea id="text" maxlength="2000" placeholder="Type a short Romanized Hinglish or English command…" required></textarea><p class="hint">Up to 2,000 characters. Input is processed locally and is not written to logs.</p></div>
<p id="example-note" class="notice" hidden>Training example — this demonstrates the interface, not unseen accuracy.</p>
<div class="actions"><button id="submit" type="submit" disabled>Preview interpretation</button><button id="example" type="button" class="secondary" disabled>Try a training example</button></div></form>
<details><summary>Scope and limitations</summary><p class="quiet" id="scope"></p></details></section>
<section class="panel" aria-labelledby="output-title"><h2 id="output-title">Interpretation</h2><div id="result" aria-live="polite"><p class="empty">Choose an engine and enter a command.<br>The result will appear here.</p></div></section></div>
<section class="panel bottom"><details><summary>Local measurements and model details</summary><p class="quiet">Measured on this computer. File size, tensor memory, and process peak memory are different quantities. Interface checks do not measure accuracy.</p><pre id="metrics">Loading…</pre></details></section>
<footer>Experimental • Eight task domains • No out-of-domain guarantee • No actions executed • Nothing is published by this demo</footer>
</main><script nonce="__NONCE__">
'use strict';
const $ = id => document.getElementById(id); let state = null, exampleIndex = 0;
const textNode = (tag, text, cls) => { const el=document.createElement(tag); el.textContent=text; if(cls)el.className=cls; return el; };
function selected(){return state.engines.find(x=>x.id===$('engine').value);}
function engineNote(){const item=selected();$('engine-note').textContent=item.description+(item.is_fine_tuned_model?' Loaded fine-tuned artifact.':' This is a baseline, not the Udgam fine-tuned result.');}
function treeNode(tree){const li=document.createElement('li');li.append(textNode('span',tree.label.replace(/^(IN|SL):/,'').replaceAll('_',' ').toLowerCase(),'node'));const children=document.createElement('ul');let words=[];const flush=()=>{if(words.length){children.append(textNode('li',words.join(' '),'leaf'));words=[];}};for(const child of tree.children){if(typeof child==='string'){words.push(child);}else{flush();children.append(treeNode(child));}}flush();li.append(children);return li;}
async function metrics(){const [m,s]=await Promise.all([fetch('/api/metrics',{cache:'no-store'}).then(r=>r.json()),fetch('/api/status',{cache:'no-store'}).then(r=>r.json())]);$('metrics').textContent=JSON.stringify({loaded_engines:s.engines,measurements:m},null,2);}
function render(data){const target=$('result');target.replaceChildren();target.append(textNode('div',data.message,'notice'+(data.ok?'':' warn')));if(data.engine)target.append(textNode('div',data.engine.label+' · '+data.latency_ms+' ms · preview only','resultmeta'));if(data.tree){const list=document.createElement('ul');list.className='tree';list.append(treeNode(data.tree));target.append(list);}if(data.reference){target.append(textNode('p','Retrieved training example: '+data.reference.text,'quiet'));}for(const [title,value] of [['Original TOP output',data.raw_top],['Ordered JSON tree',data.tree],['Automatic checks',data.checks]]){if(value!==undefined&&value!==null){const d=document.createElement('details');d.append(textNode('summary',title));d.append(textNode('pre',typeof value==='string'?value:JSON.stringify(value,null,2)));target.append(d);}}}
$('engine').addEventListener('change',engineNote);$('text').addEventListener('input',()=>{$('example-note').hidden=true;});
$('example').addEventListener('click',()=>{const item=state.examples[exampleIndex++%state.examples.length];$('text').value=item.text;$('example-note').hidden=false;$('text').focus();});
$('form').addEventListener('submit',async event=>{event.preventDefault();$('submit').disabled=true;$('result').replaceChildren(textNode('p','Processing on this computer…','empty'));try{const response=await fetch('/api/parse',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:$('text').value,engine:$('engine').value})});const data=await response.json();if(!response.ok)throw new Error(data.message||'Local request failed.');render(data);}catch(error){$('result').replaceChildren(textNode('div',error.message,'notice warn'));}finally{$('submit').disabled=false;await metrics().catch(()=>{});}});
(async()=>{try{const response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw new Error('Unable to read local status.');state=await response.json();for(const item of state.engines){const option=document.createElement('option');option.value=item.id;option.textContent=item.label;$('engine').append(option);}$('scope').textContent=state.limitation;$('engine').disabled=false;$('submit').disabled=false;$('example').disabled=!state.examples.length;engineNote();await metrics();}catch(error){$('result').replaceChildren(textNode('div',error.message,'notice warn'));}})();
</script></body></html>'''


class LocalServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, app, port):
        self.app = app
        super().__init__(("127.0.0.1", port), LocalHandler)
        actual_port = self.server_address[1]
        self.allowed_hosts = {f"127.0.0.1:{actual_port}", f"localhost:{actual_port}"}
        self.allowed_origins = {f"http://{host}" for host in self.allowed_hosts}


class LocalHandler(http.server.BaseHTTPRequestHandler):
    server_version = "UdgamLocal/1"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format, *args):
        # Access paths or exception strings can contain private command text.
        pass

    def respond(self, status, content, content_type="application/json; charset=utf-8", nonce=None):
        body = content if isinstance(content, bytes) else json.dumps(content, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'none'; connect-src 'self'; "
                         + (f"script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; " if nonce else "")
                         + "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def reject(self, message, status=400, code="invalid_request"):
        self.respond(status, {"ok": False, "status": code, "message": message, "executed_action": False})

    def check_surface(self, require_origin=False):
        hosts = self.headers.get_all("Host", [])
        origins = self.headers.get_all("Origin", [])
        if len(hosts) != 1 or hosts[0] not in self.server.allowed_hosts:
            raise RequestError("Only this localhost address is accepted.", status=403, code="host_rejected")
        if len(origins) > 1 or (origins and origins[0] not in self.server.allowed_origins):
            raise RequestError("Only the local demo origin is accepted.", status=403, code="origin_rejected")
        if require_origin and len(origins) != 1:
            raise RequestError("A matching local Origin header is required.", status=403, code="origin_required")
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise RequestError("Cross-site requests are not accepted.", status=403, code="origin_rejected")

    def do_GET(self):
        try:
            self.check_surface()
            if self.path == "/":
                nonce = secrets.token_urlsafe(24)
                self.respond(200, HTML.replace("__NONCE__", nonce).encode("utf-8"), "text/html; charset=utf-8", nonce)
            elif self.path == "/api/status":
                self.respond(200, self.server.app.status())
            elif self.path == "/api/metrics":
                self.respond(200, self.server.app.metrics())
            else:
                self.reject("Unknown local endpoint.", 404, "not_found")
        except RequestError as error:
            self.reject(str(error), error.status, error.code)

    def do_POST(self):
        try:
            self.check_surface(require_origin=True)
            if self.path != "/api/parse":
                raise RequestError("Unknown local endpoint.", status=404, code="not_found")
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get("Transfer-Encoding") is not None or len(lengths) != 1:
                raise RequestError("One Content-Length header is required.")
            try:
                length = int(lengths[0])
            except ValueError:
                raise RequestError("Invalid body length.") from None
            if length < 1 or length > MAX_BODY_BYTES:
                raise RequestError("Request body exceeds the local limit.", status=413, code="body_limit")
            if self.headers.get_content_type() != "application/json":
                raise RequestError("Use application/json.", status=415, code="content_type")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise RequestError("Incomplete request body.")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise RequestError("Body must be valid UTF-8 JSON.") from None
            self.respond(200, self.server.app.parse(payload))
        except RequestError as error:
            self.reject(str(error), error.status, error.code)
        except (TimeoutError, socket.timeout):
            self.reject("The local request body timed out.", 408, "timeout")

    def do_OPTIONS(self):
        self.reject("Cross-origin access is not enabled.", 405, "method_not_allowed")

