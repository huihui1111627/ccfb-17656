"""显存实验台的后端 API (标准库 http.server, 零第三方依赖)。

接口:
  GET /api/models              模型列表与描述
  GET /api/gpus                卡型目录
  GET /api/precisions          精度与优化器选项
  GET /api/timeline?...        单步显存时间线 (逐 tick 各类别字节)
  GET /api/checkpoint?...      checkpoint 成本/收益分析
  GET /api/solve/max-batch?... 最大批大小
  GET /api/solve/min-gpus?...  最少卡数
  GET /api/validation          批量验证结果与误差汇总
所有预估都走 spec 模式, 毫秒级返回, 不做数值计算。
"""
from __future__ import annotations

import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from memlab.checkpoint import analyze_checkpoint_options
from memlab.estimator import StepConfig, estimate_step
from memlab.hardware import GPUS
from memlab.models import REGISTRY
from memlab.parallel import ParallelConfig
from memlab.solver import max_batch_size, min_gpus_for_batch

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results",
    "validation.json",
)


def _build_cfg(q) -> StepConfig:
    model = REGISTRY[q["model"][0]]
    pc = ParallelConfig(
        mode=q.get("mode", ["single"])[0],
        world_size=int(q.get("world_size", ["1"])[0]),
        pipeline_microbatches=int(
            q.get("microbatches", ["2"])[0]
        ),
    )
    ckpt = q.get("checkpoint", [""])[0]
    if ckpt:
        pc.checkpoint_layers = set(
            c for c in ckpt.split(",") if c
        )
    seq = q.get("seq_len", [""])[0]
    return StepConfig(
        model=model,
        batch_size=int(q.get("batch", ["8"])[0]),
        precision=q.get("precision", ["fp32"])[0],
        optimizer=q.get("optimizer", ["adam"])[0],
        parallel=pc,
        seq_len_override=int(seq) if seq else None,
    )


def _capacity(q) -> int:
    gpu = q.get("gpu", ["A100-40G"])[0]
    return GPUS[gpu].capacity_bytes


def _timeline_payload(cfg, capacity):
    r = estimate_step(cfg, capacity)
    ticks = []
    for t in r.ticks:
        ticks.append({
            "tick": t.tick, "phase": t.phase,
            "allocated": t.allocated, "reserved": t.reserved,
            "fragment": t.fragmentation,
            "by_category": t.by_category,
        })
    return {
        "ok": r.ok, "oom": r.oom, "error": r.error,
        "ticks": ticks,
        "peak_allocated": r.peak_allocated,
        "peak_reserved": r.peak_reserved,
        "peak_fragment": r.peak_fragment,
        "peak_internal_fragment": r.peak_internal_fragment,
        "peak_external_fragment": r.peak_external_fragment,
        "peak_tick": r.peak_tick, "peak_phase": r.peak_phase,
        "peak_top_tensors": r.peak_top_tensors,
        "static_bytes": r.static_bytes,
        "persistent_by_category": r.persistent_by_category,
        "per_gpu_peak": r.per_gpu_peak_reserved,
        "fragmentation_ratio": r.fragmentation_ratio,
        "flops": r.flops,
        "config": r.config,
        "total_events": r.total_events,
        "reuse_hits": r.reuse_hits,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, ctype):
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        q = urllib.parse.parse_qs(parsed.query)
        try:
            if path == "/api/models":
                return self._send([
                    {"name": m.name, "kind": m.kind,
                     "description": m.description,
                     "seq_len": m.seq_len, "vocab": m.vocab}
                    for m in REGISTRY.values()
                    if not m.name.endswith("_real")
                ])
            if path == "/api/gpus":
                return self._send([
                    {"name": g.name, "memory_gib": g.memory_gib,
                     "capacity_bytes": g.capacity_bytes,
                     "fp16_tflops": g.fp16_tflops,
                     "memory_bw_gbs": g.memory_bw_gbs}
                    for g in GPUS.values()
                ])
            if path == "/api/precisions":
                from memlab.dtype import PRESETS
                return self._send({
                    "precisions": list(PRESETS.keys()),
                    "optimizers": ["adam", "sgd"],
                    "modes": ["single", "dp", "zero", "pipeline"],
                })
            if path == "/api/timeline":
                cfg = _build_cfg(q)
                return self._send(_timeline_payload(cfg, _capacity(q)))
            if path == "/api/checkpoint":
                cfg = _build_cfg(q)
                res = analyze_checkpoint_options(cfg, _capacity(q))
                return self._send({
                    "baseline_peak": res["baseline_peak"],
                    "baseline_flops": res["baseline_flops"],
                    "options": [o.__dict__ for o in res["options"]],
                    "greedy": res["greedy_recommendation"],
                })
            if path == "/api/solve/max-batch":
                cfg = _build_cfg(q)
                ans = max_batch_size(
                    cfg, _capacity(q),
                    high=int(q.get("high", ["1024"])[0]),
                )
                return self._send({
                    "feasible": ans.feasible, "max_batch": ans.value,
                    "peak_bytes": ans.peak_bytes,
                    "headroom_bytes": ans.headroom_bytes,
                    "reason": ans.reason,
                    "search_log": ans.search_log,
                })
            if path == "/api/solve/min-gpus":
                cfg = _build_cfg(q)
                ans = min_gpus_for_batch(
                    cfg, batch=int(q["batch"][0]),
                    capacity_bytes=_capacity(q),
                    mode=q.get("solve_mode", ["zero"])[0],
                )
                return self._send({
                    "feasible": ans.feasible, "min_gpus": ans.value,
                    "peak_bytes": ans.peak_bytes,
                    "reason": ans.reason, "search_log": ans.search_log,
                })
            if path == "/api/validation":
                if not os.path.exists(RESULTS_PATH):
                    return self._send({"available": False})
                with open(RESULTS_PATH, encoding="utf-8") as f:
                    return self._send(json.load(f))
            if path in ("/", "/index.html"):
                return self._send_file(
                    os.path.join(STATIC_DIR, "index.html"),
                    "text/html; charset=utf-8",
                )
            if path == "/app.js":
                return self._send_file(
                    os.path.join(STATIC_DIR, "app.js"),
                    "application/javascript; charset=utf-8",
                )
            if path == "/styles.css":
                return self._send_file(
                    os.path.join(STATIC_DIR, "styles.css"),
                    "text/css; charset=utf-8",
                )
            self._send({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._send({"error": f"{type(e).__name__}: {e}"}, 500)


def serve(host="127.0.0.1", port=8765):
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"显存实验台: http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    serve(port=args.port)
