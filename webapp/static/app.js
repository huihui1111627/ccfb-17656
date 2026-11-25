// memlab 显存实验台前端 (原生 JS, 无依赖)
"use strict";

const CAT_COLORS = {
  param: "#5b9dff",
  master: "#8b7bff",
  grad: "#3ecf8e",
  optim: "#f5b942",
  activation: "#ff8f6b",
  workspace: "#b07cd9",
  input: "#7fd4d4",
  communication: "#ffd166",
};
const CAT_LABELS = {
  param: "参数", master: "FP32主权重", grad: "梯度", optim: "优化器状态",
  activation: "保存的激活", workspace: "临时工作区",
  input: "输入数据", communication: "通信缓冲",
};
const CAT_ORDER = ["param", "master", "grad", "optim", "activation",
  "workspace", "input", "communication"];
const PHASE_LABELS = {
  build: "构建", forward: "前向", backward: "反向",
  optimizer_step: "优化器", communication: "通信",
};
const MiB = (b) => (b / (1024 * 1024));
const fmtMiB = (b) => MiB(b).toFixed(2) + " MiB";

let state = { timeline: null, hoverTick: null };

const $ = (id) => document.getElementById(id);

async function api(path, params) {
  const url = path + "?" + new URLSearchParams(params).toString();
  const r = await fetch(url);
  return r.json();
}

function currentParams() {
  const p = {
    model: $("model").value,
    batch: $("batch").value,
    precision: $("precision").value,
    optimizer: $("optimizer").value,
    mode: $("mode").value,
    world_size: $("world").value,
    gpu: $("gpu").value,
  };
  const seq = $("seq").value;
  if (seq) p.seq_len = seq;
  const ck = $("checkpoint").value.trim();
  if (ck) p.checkpoint = ck;
  return p;
}

async function loadOptions() {
  const [models, gpus, prec] = await Promise.all([
    api("/api/models"), api("/api/gpus"), api("/api/precisions"),
  ]);
  $("model").innerHTML = models.map(m =>
    `<option value="${m.name}">${m.name}</option>`).join("");
  $("model").onchange = () => {
    const m = models.find(x => x.name === $("model").value);
    $("model-desc").textContent = m.description;
  };
  $("model").onchange();
  $("gpu").innerHTML = gpus.map(g =>
    `<option value="${g.name}">${g.name} (${g.memory_gib}G)</option>`)
    .join("");
  $("gpu").value = "A100-40G";
  $("precision").innerHTML = prec.precisions
    .map(p => `<option>${p}</option>`).join("");
  $("optimizer").innerHTML = prec.optimizers
    .map(p => `<option>${p}</option>`).join("");
}

async function refresh() {
  const data = await api("/api/timeline", currentParams());
  if (data.error) { alert(data.error); return; }
  state.timeline = data;
  renderCards(data);
  renderLegend();
  drawChart();
  renderTopTable(data);
  renderPerGpu(data);
}

function renderCards(d) {
  const verdict = $("verdict");
  if (d.ok) {
    verdict.textContent = "✓ 能跑";
    verdict.className = "card-value pass";
  } else {
    verdict.textContent = "✗ OOM";
    verdict.className = "card-value fail";
  }
  $("peak-reserved").textContent = fmtMiB(d.peak_reserved);
  $("peak-alloc").textContent = fmtMiB(d.peak_allocated);
  $("peak-frag").innerHTML =
    fmtMiB(d.peak_fragment) +
    ` (${(d.fragmentation_ratio * 100).toFixed(1)}%)` +
    `<div style="font-size:10px;color:var(--muted);font-weight:400">内${d.peak_internal_fragment != null ? fmtMiB(d.peak_internal_fragment) : "-"} / 外${d.peak_external_fragment != null ? fmtMiB(d.peak_external_fragment) : "-"}</div>`;
  $("peak-phase").innerHTML =
    (PHASE_LABELS[d.peak_phase] || d.peak_phase) +
    `<div style="font-size:11px;color:var(--muted)">tick ${d.peak_tick}</div>`;
  $("reuse").textContent = d.reuse_hits +
    ` / ${d.total_events}`;
}

function renderLegend() {
  const present = new Set();
  state.timeline.ticks.forEach(t =>
    CAT_ORDER.forEach(c => { if (t.by_category[c] > 0) present.add(c); }));
  $("legend").innerHTML = CAT_ORDER.filter(c => present.has(c)).map(c =>
    `<span class="item"><span class="swatch" style="background:${CAT_COLORS[c]}"></span>${CAT_LABELS[c]}</span>`
  ).join("");
}

function drawChart() {
  const canvas = $("chart");
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth;
  canvas.width = cssW * dpr;
  canvas.height = 340 * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const W = cssW, H = 340;
  const pad = { l: 64, r: 16, t: 14, b: 30 };
  const ticks = state.timeline.ticks;
  if (!ticks.length) return;

  const showReserved = $("show-reserved").checked;
  const maxY = Math.max(...ticks.map(t =>
    showReserved ? t.reserved : t.allocated));
  const x = (i) => pad.l + (i / Math.max(ticks.length - 1, 1)) *
    (W - pad.l - pad.r);
  const y = (v) => H - pad.b - (v / maxY) * (H - pad.t - pad.b);

  // 网格 + Y 轴
  ctx.strokeStyle = "#2c3a52"; ctx.fillStyle = "#8fa0b8";
  ctx.font = "10px sans-serif"; ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const v = maxY * g / 4, yy = y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(W - pad.r, yy);
    ctx.stroke();
    ctx.fillText(fmtMiB(v), 4, yy + 3);
  }

  // 阶段背景色带
  let segStart = 0;
  for (let i = 1; i <= ticks.length; i++) {
    if (i === ticks.length || ticks[i].phase !== ticks[segStart].phase) {
      if (ticks[segStart].phase === "backward" ||
          ticks[segStart].phase === "communication") {
        ctx.fillStyle = ticks[segStart].phase === "backward"
          ? "rgba(245,185,66,0.06)" : "rgba(255,209,102,0.08)";
        ctx.fillRect(x(segStart), pad.t, x(i - 1) - x(segStart) + 2,
          H - pad.t - pad.b);
      }
      segStart = i;
    }
  }

  // 堆叠面积
  CAT_ORDER.forEach(cat => {
    ctx.beginPath();
    let acc = 0;
    const lower = ticks.map(t => {
      const base = acc; acc += t.by_category[cat] || 0; return base;
    });
    ctx.moveTo(x(0), y(lower[0]));
    ticks.forEach((t, i) => ctx.lineTo(x(i), y(lower[i])));
    for (let i = ticks.length - 1; i >= 0; i--) {
      const top = lower[i] + (ticks[i].by_category[cat] || 0);
      ctx.lineTo(x(i), y(top));
    }
    ctx.closePath();
    ctx.fillStyle = CAT_COLORS[cat] + "cc";
    ctx.fill();
  });

  // reserved 轮廓线
  if (showReserved) {
    ctx.beginPath();
    ticks.forEach((t, i) => i ? ctx.lineTo(x(i), y(t.reserved))
      : ctx.moveTo(x(i), y(t.reserved)));
    ctx.strokeStyle = "#ff6b6b"; ctx.lineWidth = 1.6; ctx.stroke();
  }

  // 峰值竖线
  const pk = state.timeline.peak_tick;
  ctx.strokeStyle = "#ff6b6b"; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(x(pk), pad.t); ctx.lineTo(x(pk), H - pad.b);
  ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = "#ff6b6b";
  ctx.fillText("峰值", x(pk) + 4, pad.t + 10);

  // X 轴阶段标签
  ctx.fillStyle = "#8fa0b8";
  let s = 0;
  for (let i = 1; i <= ticks.length; i++) {
    if (i === ticks.length || ticks[i].phase !== ticks[s].phase) {
      const cx = (x(s) + x(i - 1)) / 2;
      ctx.fillText(PHASE_LABELS[ticks[s].phase] || ticks[s].phase,
        cx - 14, H - 10);
      s = i;
    }
  }

  // 悬停竖线
  if (state.hoverTick != null) {
    const i = state.hoverTick;
    ctx.strokeStyle = "#e6ecf5";
    ctx.beginPath(); ctx.moveTo(x(i), pad.t);
    ctx.lineTo(x(i), H - pad.b); ctx.stroke();
  }
}

function renderTopTable(d) {
  const tb = $("top-table").querySelector("tbody");
  tb.innerHTML = d.peak_top_tensors.map(t =>
    `<tr><td>${t.name}</td><td>${CAT_LABELS[t.category] || t.category}</td>
     <td>${fmtMiB(t.bytes)}</td></tr>`).join("");
}

function renderPerGpu(d) {
  const peaks = d.per_gpu_peak && d.per_gpu_peak.length
    ? d.per_gpu_peak : [d.peak_reserved];
  const max = Math.max(...peaks);
  $("per-gpu").innerHTML = peaks.map((p, i) =>
    `<div class="gpu-row"><span class="name">卡${i}</span>
      <div class="gpu-bar"><div class="gpu-fill" style="width:${p / max * 100}%"></div></div>
      <span class="gpu-val">${fmtMiB(p)}</span></div>`).join("");
}

// 悬停
$("chart").addEventListener("mousemove", (e) => {
  if (!state.timeline) return;
  const rect = e.target.getBoundingClientRect();
  const padL = 64, padR = 16;
  const frac = (e.clientX - rect.left - padL) /
    (rect.width - padL - padR);
  const idx = Math.round(frac * (state.timeline.ticks.length - 1));
  if (idx < 0 || idx >= state.timeline.ticks.length) return;
  state.hoverTick = idx;
  drawChart();
  const t = state.timeline.ticks[idx];
  const parts = CAT_ORDER.filter(c => t.by_category[c] > 0)
    .map(c => `${CAT_LABELS[c]}: ${fmtMiB(t.by_category[c])}`).join("　");
  $("hover-info").innerHTML =
    `tick ${t.tick} [${PHASE_LABELS[t.phase] || t.phase}]　` +
    `预留 ${fmtMiB(t.reserved)}　活跃 ${fmtMiB(t.allocated)}<br>${parts}`;
});
$("chart").addEventListener("mouseleave", () => {
  state.hoverTick = null; drawChart();
  $("hover-info").textContent = "悬停查看每个时刻的构成";
});

// 容量规划
$("btn-maxbatch").onclick = async () => {
  const r = await api("/api/solve/max-batch",
    { ...currentParams(), high: 1024 });
  $("solver-out").innerHTML = r.feasible
    ? `<span class="ok">最大批大小 = ${r.max_batch}</span>\n峰值 ${fmtMiB(r.peak_bytes)}\n剩余 ${fmtMiB(r.headroom_bytes)}\n${r.reason}\n搜索 ${r.search_log.length} 个点`
    : `<span class="bad">不可行</span>\n${r.reason}`;
};
$("btn-mingpu").onclick = async () => {
  const r = await api("/api/solve/min-gpus",
    { ...currentParams(), batch: $("batch").value, solve_mode: "zero" });
  $("solver-out").innerHTML = r.feasible
    ? `<span class="ok">最少 ${r.min_gpus} 张卡</span>\n峰值/卡 ${fmtMiB(r.peak_bytes)}\n${r.reason}`
    : `<span class="bad">${r.reason}</span>`;
};
$("btn-ckpt").onclick = async () => {
  const r = await api("/api/checkpoint", currentParams());
  if (r.error) { $("ckpt-out").textContent = r.error; return; }
  let s = `基线峰值 ${fmtMiB(r.baseline_peak)}\n\n逐层 (省MiB / 额外FLOP):\n`;
  s += r.options.map(o =>
    `· ${o.layer_name}: -${MiB(o.peak_saved_bytes).toFixed(1)}MiB / ` +
    `+${(o.extra_flops_ratio * 100).toFixed(1)}% ${o.note}`).join("\n");
  s += `\n\n贪心推荐: [${r.greedy.layers.join(", ")}]\n` +
    `共省 ${fmtMiB(r.greedy.total_saved)}, ` +
    `额外 FLOP ${(r.greedy.total_extra_flops / r.baseline_flops * 100).toFixed(1)}%`;
  $("ckpt-out").textContent = s;
};

// 验证结果
$("btn-load-val").onclick = async () => {
  const r = await api("/api/validation");
  if (!r.available) {
    $("val-summary").textContent = "尚未生成 results/validation.json";
    return;
  }
  const sm = r.summary;
  $("val-summary").innerHTML =
    `共 <b>${sm.total_cases}</b> 个组合 · ` +
    `能跑/不能跑判定错误 <b class="${sm.decision_mismatches ? "badge-bad" : "badge-ok"}">${sm.decision_mismatches}</b> 次 · ` +
    `判定准确率 <b>${(sm.decision_accuracy * 100).toFixed(1)}%</b> · ` +
    `峰值相对误差 最大 <b>${(sm.peak_error_max * 100).toFixed(3)}%</b> ` +
    `均值 <b>${(sm.peak_error_mean * 100).toFixed(3)}%</b> · ` +
    `精确字节匹配 <b>${sm.exact_byte_match_cases}/${sm.total_cases}</b>`;
  $("val-table").querySelector("tbody").innerHTML = r.results.map(x =>
    `<tr><td>${x.model}</td><td>${x.batch}</td><td>${x.precision}</td>
     <td>${x.optimizer}</td><td>${x.mode}</td><td>${x.world_size}</td>
     <td>${x.checkpoint.join("|")}</td>
     <td>${x.est_peak_mib}</td><td>${x.real_peak_mib}</td>
     <td>${(x.peak_rel_error * 100).toFixed(3)}%</td>
     <td class="${x.decision_ok ? "badge-ok" : "badge-bad"}">${x.decision_ok ? "一致" : "错误"}</td></tr>`
  ).join("");
};

// 事件绑定
["run", "model", "batch", "seq", "precision", "optimizer", "mode",
  "world", "gpu", "checkpoint"].forEach(id => {
  const el = $(id);
  if (id === "run") el.onclick = refresh;
  else el.addEventListener("change", refresh);
});
$("show-reserved").addEventListener("change", drawChart);

loadOptions().then(refresh);
