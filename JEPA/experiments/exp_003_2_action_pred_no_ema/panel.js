/**
 * exp_003_normalized_latent_jepa — Dashboard panel plugin.
 *
 * Globals provided by index.html:
 *   window.JEPA.arcRgb(idx)          → CSS rgb string
 *   window.JEPA.embRgb(v)            → [R,G,B]
 *   window.JEPA.drawHeatmapBar(canvas, values, h)
 *   window.JEPA.ARC_COLORS           → [[R,G,B] × 16]
 *
 * Exported API:
 *   render(data, el)           — called once when episode loads
 *   updateStep(step, t, el)    — called on every step change
 */

'use strict';

// ── Dark-theme palette ────────────────────────────────────────────────────────
const BG       = '#0d1117';
const BORDER   = '#21262d';
const TEXT_DIM = '#8b949e';
const TEXT     = '#c9d1d9';
const TEXT_HI  = '#e6edf3';
const ACCENT   = '#388bfd';

// ── Per-section state ─────────────────────────────────────────────────────────
let _saLayer     = 1;   // 1 or 2 (SA block index)
let _selSAPatch  = null;
let _percRound   = 0;
let _selLatent   = null;

// ── DOM helpers ───────────────────────────────────────────────────────────────
function _el(tag, attrs = {}, style = {}) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  Object.assign(el.style, style);
  return el;
}

function _section(title, content, buttons = null) {
  const wrap = _el('div', {}, { borderTop: `1px solid ${BORDER}`, padding: '10px 14px 14px' });
  const hdr  = _el('div', {}, { display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' });
  const lbl  = _el('div', {}, {
    fontSize: '10px', fontWeight: 'bold', letterSpacing: '.8px', color: ACCENT,
    textTransform: 'uppercase', fontFamily: "'Menlo','Monaco','Consolas',monospace",
  });
  lbl.textContent = title;
  hdr.appendChild(lbl);
  if (buttons) hdr.appendChild(buttons);
  wrap.appendChild(hdr);
  wrap.appendChild(content);
  return wrap;
}

function _toggleBtnGroup(labels, activeIdx, onChange) {
  const grp = _el('div', {}, { display: 'flex', gap: '4px' });
  const btns = labels.map((lbl, i) => {
    const b = _el('button', {}, {
      background: i === activeIdx ? ACCENT : '#21262d',
      color: i === activeIdx ? '#fff' : TEXT_DIM,
      border: `1px solid ${BORDER}`,
      borderRadius: '4px',
      padding: '2px 8px',
      fontSize: '10px',
      cursor: 'pointer',
      fontFamily: "'Menlo','Monaco','Consolas',monospace",
    });
    b.textContent = lbl;
    b.addEventListener('click', () => {
      btns.forEach((bb, j) => {
        bb.style.background = j === i ? ACCENT : '#21262d';
        bb.style.color = j === i ? '#fff' : TEXT_DIM;
      });
      onChange(i);
    });
    grp.appendChild(b);
    return b;
  });
  return grp;
}

// ── 2D heatmap: rows × cols, maps [0, max] → dark background → bright blue ──
// Cells with t=0 use a dark but visible blue; t=1 uses bright accent blue.
// Cell borders drawn via CSS box-shadow is unavailable on canvas, so we leave
// 1px gaps between cells using background fill (the BG pixel rows/cols).
function _drawHeatmap2D(canvas, matrix, cellW, cellH) {
  const rows = matrix.length, cols = matrix[0].length;
  const GAP = 1;
  const totalW = cols * cellW + (cols - 1) * GAP;
  const totalH = rows * cellH + (rows - 1) * GAP;
  canvas.width  = totalW;
  canvas.height = totalH;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(totalW, totalH);

  // Fill with background color first
  for (let i = 0; i < img.data.length; i += 4) {
    img.data[i]   = 13; img.data[i+1] = 17;
    img.data[i+2] = 23; img.data[i+3] = 255;
  }

  let maxV = 0;
  for (const row of matrix) for (const v of row) if (v > maxV) maxV = v;
  const sc = maxV > 1e-9 ? 1 / maxV : 1;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const t = Math.max(0, Math.min(1, matrix[r][c] * sc));
      // Color: near-bg dark navy (20,28,60) → bright accent blue (74,158,255)
      // Even t=0.05 shows as clearly distinct from background
      const R = Math.round(20  + t * (74  - 20));
      const G = Math.round(28  + t * (158 - 28));
      const B = Math.round(60  + t * (255 - 60));
      const x0 = c * (cellW + GAP);
      const y0 = r * (cellH + GAP);
      for (let dy = 0; dy < cellH; dy++) {
        for (let dx = 0; dx < cellW; dx++) {
          const base = ((y0 + dy) * totalW + x0 + dx) * 4;
          img.data[base]=R; img.data[base+1]=G; img.data[base+2]=B; img.data[base+3]=255;
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}

// ── Patch thumbnail ───────────────────────────────────────────────────────────
const THUMB = 56, TSCALE = 3.5;

function _drawPatch(canvas, frame, pi, overlayRgba = null, brightness = 1.0, label = null) {
  const pr = Math.floor(pi / 4), pc = pi % 4;
  canvas.width = canvas.height = THUMB;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(THUMB, THUMB);
  const bf = Math.max(0, Math.min(1, brightness));
  const scale = Math.round(TSCALE);
  const ARC = window.JEPA.ARC_COLORS || [];
  for (let r = 0; r < 16; r++) {
    for (let c = 0; c < 16; c++) {
      const colorIdx = frame[pr * 16 + r][pc * 16 + c];
      const [R, G, B] = ARC[colorIdx] || [80, 80, 80];
      for (let dy = 0; dy < scale && r * scale + dy < THUMB; dy++) {
        for (let dx = 0; dx < scale && c * scale + dx < THUMB; dx++) {
          const base = ((r * scale + dy) * THUMB + c * scale + dx) * 4;
          img.data[base]   = R * bf;
          img.data[base+1] = G * bf;
          img.data[base+2] = B * bf;
          img.data[base+3] = 255;
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  if (overlayRgba) { ctx.fillStyle = overlayRgba; ctx.fillRect(0, 0, THUMB, THUMB); }
  if (label !== null) {
    ctx.font = 'bold 9px monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    ctx.fillRect(THUMB - 28, THUMB - 14, 28, 13);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, THUMB - 2, THUMB - 2);
  }
}

// ── Patch hover tooltip ───────────────────────────────────────────────────────
let _tipEl      = null;
let _tipHmCanvas = null;

function _ensureTip() {
  if (_tipEl) return;
  _tipEl = _el('div', {}, {
    position: 'fixed', display: 'none', zIndex: '9999',
    background: '#1c2128', border: `1px solid ${ACCENT}`,
    borderRadius: '6px', padding: '8px 12px', maxWidth: '300px',
    boxShadow: '0 4px 20px rgba(0,0,0,.6)',
    fontFamily: "'Menlo','Monaco','Consolas',monospace",
    fontSize: '11px', color: TEXT, lineHeight: '1.55',
    pointerEvents: 'none',
  });
  document.body.appendChild(_tipEl);
}

function _drawTipHeatmap(emb) {
  if (!_tipHmCanvas) {
    _tipHmCanvas = document.createElement('canvas');
    _tipHmCanvas.style.cssText =
      'display:block;border-radius:2px;image-rendering:pixelated;margin:4px 0';
  }
  _tipHmCanvas.width = 256; _tipHmCanvas.height = 12;
  const ctx = _tipHmCanvas.getContext('2d');
  const img = ctx.createImageData(256, 12);
  for (let i = 0; i < 128; i++) {
    const [r, g, b] = window.JEPA.embRgb(emb[i]);
    for (let py = 0; py < 12; py++) {
      const base0 = (py * 256 + i * 2) * 4;
      const base1 = base0 + 4;
      img.data[base0]=r; img.data[base0+1]=g; img.data[base0+2]=b; img.data[base0+3]=255;
      img.data[base1]=r; img.data[base1+1]=g; img.data[base1+2]=b; img.data[base1+3]=255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return _tipHmCanvas;
}

function _showPatchTip(e, ps, pi, emb) {
  _ensureTip();
  const fmt  = v => v != null ? Number(v).toFixed(4) : '—';
  const fmtP = v => v != null ? (v * 100).toFixed(1) + '%' : '—';

  let html =
    `<b style="color:${TEXT_HI}">Patch ${pi}` +
    ` <span style="color:#555;font-weight:normal">(row ${Math.floor(pi/4)}, col ${pi%4})</span></b><br>` +
    `<span style="color:${TEXT_DIM}" title="L2 norm">Norm</span> <b>${fmt(ps.norm)}</b>` +
    ` &nbsp;<span style="color:${TEXT_DIM}" title="Mean across 128 dims">Mean</span> <b>${fmt(ps.mean)}</b>` +
    ` &nbsp;<span style="color:${TEXT_DIM}" title="Std across 128 dims">Std</span> <b>${fmt(ps.std)}</b><br>` +
    `<span style="color:${TEXT_DIM}">Min</span> <b>${fmt(ps.min_val)}</b>` +
    ` &nbsp;<span style="color:${TEXT_DIM}">Max</span> <b>${fmt(ps.max_val)}</b><br>` +
    `<span style="color:${TEXT_DIM}" title="-Σv²·log(v²)/log(128)">Entropy</span> <b>${fmt(ps.activation_entropy)}</b>`;

  _tipEl.innerHTML = html;

  if (emb) {
    const sepDiv = document.createElement('div');
    sepDiv.style.cssText =
      `font-size:9px;color:${TEXT_DIM};margin-top:4px;text-transform:uppercase;letter-spacing:.5px`;
    sepDiv.textContent = 'Embedding (128 dims, 2px/dim)';
    _tipEl.appendChild(sepDiv);
    _tipEl.appendChild(_drawTipHeatmap(emb));
    const f = v => Number(v).toFixed(4);
    const vecDiv = document.createElement('div');
    vecDiv.style.cssText = `font-size:10px;color:#a5d6ff;word-break:break-all;margin-bottom:4px`;
    vecDiv.textContent =
      `[${f(emb[0])}, ${f(emb[1])}, ${f(emb[2])}, …, ${f(emb[125])}, ${f(emb[126])}, ${f(emb[127])}]`;
    _tipEl.appendChild(vecDiv);
  }

  if (ps.cos_sim_prev != null) {
    const prevDiv = document.createElement('div');
    prevDiv.style.cssText = `border-top:1px solid #21262d;margin-top:4px;padding-top:4px`;
    prevDiv.innerHTML =
      `<div style="font-size:9px;color:${TEXT_DIM};text-transform:uppercase;letter-spacing:.5px;margin-bottom:2px">vs previous timestep</div>` +
      `<span style="color:${TEXT_DIM}">Cos-sim</span> <b>${fmt(ps.cos_sim_prev)}</b>` +
      ` &nbsp;<span style="color:${TEXT_DIM}">L2 Δ</span> <b>${fmt(ps.l2_dist_prev)}</b><br>` +
      `<span style="color:${TEXT_DIM}">Mean|Δ|/dim</span> <b>${fmt(ps.mean_abs_diff_prev)}</b>` +
      ` &nbsp;<span style="color:${TEXT_DIM}">Max|Δ|/dim</span> <b>${fmt(ps.max_abs_diff_prev)}</b>`;
    if (ps.pixel_change_frac != null) {
      prevDiv.innerHTML +=
        `<br><span style="color:${TEXT_DIM}">Pixel Δ frac</span> <b>${fmtP(ps.pixel_change_frac)}</b>`;
    }
    _tipEl.appendChild(prevDiv);
  }

  _tipEl.style.display = 'block';
  const TW = 300, TH = _tipEl.offsetHeight || 180;
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + TW > window.innerWidth)  x = e.clientX - TW - 10;
  if (y + TH > window.innerHeight) y = e.clientY - TH - 10;
  _tipEl.style.left = x + 'px'; _tipEl.style.top = y + 'px';
}

function _hideTip() { if (_tipEl) _tipEl.style.display = 'none'; }

// ════════════════════════════════════════════════════════════════════════════
// SECTION A — ENCODER SELF-ATTENTION
// ════════════════════════════════════════════════════════════════════════════

function _buildSASection(el) {
  const wrap = _el('div');

  const hint = _el('div', { id: 'p03-sa-hint' }, {
    fontSize: '10px', color: TEXT_DIM, marginBottom: '8px',
  });
  hint.textContent = 'Click a patch to see which other patches it attends to.';

  const grid = _el('div', { id: 'p03-sa-grid' }, {
    display: 'grid', gridTemplateColumns: `repeat(4, ${THUMB}px)`, gap: '3px',
  });

  for (let i = 0; i < 16; i++) {
    const cell = _el('div', {}, { position: 'relative', cursor: 'pointer' });
    const c = _el('canvas', {
      id: `p03-sa-${i}`, width: THUMB, height: THUMB,
      title: `Patch ${i} (row ${Math.floor(i/4)}, col ${i%4})`,
    }, { display: 'block', imageRendering: 'pixelated', border: `1px solid ${BORDER}` });
    cell.appendChild(c);
    grid.appendChild(cell);
  }

  wrap.appendChild(hint);
  wrap.appendChild(grid);

  // Embedding summary stats
  const summ = _el('div', {}, { marginTop: '10px', display: 'flex', gap: '16px', flexWrap: 'wrap' });
  [
    ['p03-es-cos',  'Pairwise cos-sim',
     'Mean cosine similarity between all 16 patch embedding pairs. Near 0 = diverse; near 1 = collapsed.'],
    ['p03-es-rank', 'Effective rank',
     'exp(entropy of SVD singular values). Range [1,16]. Higher = more diverse directions.'],
    ['p03-es-dead', 'Dead dims',
     'Dimensions where std across 16 patches < 0.01.'],
    ['p03-es-drift','Mean drift vs t-1',
     'Mean L2 distance between patch embeddings at t and t-1.'],
  ].forEach(([id, lbl, tip]) => {
    const row = _el('div', {}, { display: 'flex', flexDirection: 'column', alignItems: 'flex-start' });
    const ll = _el('span', {}, { fontSize: '9px', color: TEXT_DIM, cursor: 'help',
      borderBottom: `1px dotted ${BORDER}` });
    ll.textContent = lbl; ll.title = tip;
    const vv = _el('span', { id }, { fontSize: '11px', color: TEXT_HI,
      fontVariantNumeric: 'tabular-nums' });
    vv.textContent = '—';
    row.appendChild(ll); row.appendChild(vv);
    summ.appendChild(row);
  });
  wrap.appendChild(summ);

  el.appendChild(_section('ENCODER SELF-ATTENTION', wrap,
    _toggleBtnGroup(['Block 1', 'Block 2'], 0, i => { _saLayer = i + 1; _refreshSAGrid(); }),
  ));
}

let _saStep = null;

function _refreshSAGrid() {
  if (!_saStep) return;
  const frame = _saStep.frame;
  const attn  = _saLayer === 1 ? _saStep.encoder_sa_block1 : _saStep.encoder_sa_block2;
  const selAttn = (_selSAPatch !== null && attn) ? attn[_selSAPatch] : null;
  const maxW = selAttn ? Math.max(...selAttn, 1e-9) : 1;

  for (let i = 0; i < 16; i++) {
    const c = document.getElementById(`p03-sa-${i}`);
    if (!c) continue;
    if (selAttn) {
      const norm = selAttn[i] / maxW;
      const bf   = 0.18 + 0.82 * norm;
      const alpha = norm * 0.82;
      _drawPatch(c, frame, i, `rgba(255,90,0,${alpha.toFixed(2)})`, bf,
        (selAttn[i] * 100).toFixed(1) + '%');
    } else {
      _drawPatch(c, frame, i, null, 1.0);
    }
    c.style.border = i === _selSAPatch ? '2px solid #f0c040' : `1px solid ${BORDER}`;
  }

  const hint = document.getElementById('p03-sa-hint');
  if (hint) {
    hint.textContent = _selSAPatch !== null
      ? `Attention FROM Patch ${_selSAPatch} (row ${Math.floor(_selSAPatch/4)}, col ${_selSAPatch%4}) TO each patch (Block ${_saLayer})`
      : 'Click a patch to see which other patches it attends to.';
  }
}

function _initSAEvents() {
  const grid = document.getElementById('p03-sa-grid');
  if (!grid) return;

  grid.addEventListener('click', e => {
    const c = e.target.closest('canvas[id^="p03-sa-"]');
    if (!c) return;
    const i = parseInt(c.id.replace('p03-sa-', ''));
    _selSAPatch = (_selSAPatch === i) ? null : i;
    _refreshSAGrid();
  });

  grid.addEventListener('mousemove', e => {
    if (!_saStep) return;
    const c = e.target.closest('canvas[id^="p03-sa-"]');
    if (!c) { _hideTip(); return; }
    const i = parseInt(c.id.replace('p03-sa-', ''));
    const ps  = _saStep.per_patch_stats?.[i];
    const emb = _saStep.patch_embeddings?.[i];
    if (ps) _showPatchTip(e, ps, i, emb);
  });

  grid.addEventListener('mouseleave', _hideTip);
}

// ════════════════════════════════════════════════════════════════════════════
// SECTION B — PERCEIVER RESAMPLER
// ════════════════════════════════════════════════════════════════════════════

const LAT_BAR_W = 16;   // px wide per latent bar
const LAT_BAR_H = 256;  // px tall (2px per dim)

// Perceiver flow layout constants
const PERC_BAR_W  = 14;   // px per latent bar in the flow diagram
const PERC_BAR_H  = 128;  // px tall (1px per dim)
const PERC_BAR_GAP = 2;   // px gap between the 4 bars in a group
const PERC_CA_CW  = 8;    // cross-attn cell width  (16 cols × 8 = 128px)
const PERC_CA_CH  = 32;   // cross-attn/self-attn cell height (4 rows × 32 ≈ bar height)
const PERC_SA_CW  = 32;   // self-attn cell width   (4 cols × 32 = 128px)

// Build one round row: [input bars] → [cross-attn] → [self-attn] → [output bars]
function _buildPercRoundRow(wrap, roundIdx, inputLabel, outputLabel) {
  const row = _el('div', {}, {
    display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '14px',
  });

  // Vertical round label
  const rlbl = _el('div', {}, {
    fontSize: '9px', color: TEXT_DIM, writingMode: 'vertical-lr',
    transform: 'rotate(180deg)', alignSelf: 'center', whiteSpace: 'nowrap',
  });
  rlbl.textContent = `Round ${roundIdx}`;
  row.appendChild(rlbl);

  // Helper: group of 4 vertical latent bars
  const makeBars = (idPrefix, groupLbl) => {
    const wrap2 = _el('div', {}, { display: 'flex', flexDirection: 'column', alignItems: 'center' });
    const hdr = _el('div', {}, {
      fontSize: '8px', color: TEXT_DIM, marginBottom: '3px', whiteSpace: 'nowrap', textAlign: 'center',
    });
    hdr.textContent = groupLbl;
    wrap2.appendChild(hdr);
    const brow = _el('div', {}, { display: 'flex', gap: PERC_BAR_GAP + 'px' });
    for (let li = 0; li < 4; li++) {
      const col = _el('div', {}, { display: 'flex', flexDirection: 'column', alignItems: 'center' });
      const lbl = _el('div', {}, { fontSize: '8px', color: TEXT_DIM, marginBottom: '1px' });
      lbl.textContent = `L${li}`;
      col.appendChild(lbl);
      const c = _el('canvas', {
        id: `p03-perc-${idPrefix}-bar-${li}`, width: PERC_BAR_W, height: PERC_BAR_H,
      }, { display: 'block', imageRendering: 'pixelated', border: `1px solid ${BORDER}`, cursor: 'crosshair' });
      // Hover: show dim + value + norm
      c.addEventListener('mousemove', (e) => {
        _ensureTip();
        const el2 = document.getElementById(`p03-perc-${idPrefix}-bar-${li}`);
        if (!el2) return;
        const rect = el2.getBoundingClientRect();
        const my = (e.clientY - rect.top) * (el2.height / rect.height);
        const dimIdx = Math.min(127, Math.floor(my / (el2.height / 128)));
        // Find vec from current step
        const stepVecs = _percPercGroupVecs[idPrefix];
        const vec = stepVecs?.[li];
        if (!vec) { _tipEl.style.display = 'none'; return; }
        const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
        _tipEl.innerHTML =
          `<b style="color:${TEXT_HI}">${groupLbl} L${li}</b><br>` +
          `dim <b>${dimIdx}</b>: <b>${Number(vec[dimIdx]).toFixed(4)}</b><br>` +
          `norm=<b>${norm.toFixed(3)}</b>`;
        _tipEl.style.display = 'block';
        let x = e.clientX + 12, y2 = e.clientY + 12;
        if (x + 210 > window.innerWidth)  x = e.clientX - 220;
        if (y2 + 65 > window.innerHeight) y2 = e.clientY - 75;
        _tipEl.style.left = x + 'px'; _tipEl.style.top = y2 + 'px';
      });
      c.addEventListener('mouseleave', () => { if (_tipEl) _tipEl.style.display = 'none'; });
      col.appendChild(c);
      brow.appendChild(col);
    }
    wrap2.appendChild(brow);
    return wrap2;
  };

  // Helper: heatmap with header + hover
  const makeHeatmap = (id, hdrText, hoverFn) => {
    const wrap2 = _el('div', {}, { display: 'flex', flexDirection: 'column', alignItems: 'center' });
    const hdr = _el('div', {}, {
      fontSize: '8px', color: TEXT_DIM, marginBottom: '3px', whiteSpace: 'nowrap', textAlign: 'center',
    });
    hdr.textContent = hdrText;
    wrap2.appendChild(hdr);
    const c = _el('canvas', { id, width: '10', height: '10' }, {
      display: 'block', imageRendering: 'pixelated', border: `1px solid ${BORDER}`, cursor: 'crosshair',
    });
    c.addEventListener('mousemove', hoverFn);
    c.addEventListener('mouseleave', () => { if (_tipEl) _tipEl.style.display = 'none'; });
    wrap2.appendChild(c);
    return wrap2;
  };

  const arrow = () => {
    const a = _el('div', {}, {
      fontSize: '13px', color: TEXT_DIM, alignSelf: 'center', marginTop: '13px', userSelect: 'none',
    });
    a.textContent = '→';
    return a;
  };

  // Cross-attn hover: L{q} → Patch {k}
  const crossHover = (e) => {
    _ensureTip();
    const ca = _percStep?.[`perceiver_cross_attn_r${roundIdx}`];
    if (!ca) { _tipEl.style.display = 'none'; return; }
    const c = document.getElementById(`p03-perc-r${roundIdx}-cross`);
    const rect = c.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (c.width / rect.width);
    const my = (e.clientY - rect.top)  * (c.height / rect.height);
    const qRow = Math.min(3,  Math.floor(my / (PERC_CA_CH + 1)));
    const kCol = Math.min(15, Math.floor(mx / (PERC_CA_CW + 1)));
    const val = ca[qRow]?.[kCol];
    if (val == null) { _tipEl.style.display = 'none'; return; }
    _tipEl.innerHTML =
      `<b style="color:${TEXT_HI}">L${qRow} → Patch ${kCol}</b> (row ${Math.floor(kCol/4)}, col ${kCol%4})<br>` +
      `attention: <b>${(val * 100).toFixed(2)}%</b>`;
    _tipEl.style.display = 'block';
    let x = e.clientX + 12, y2 = e.clientY + 12;
    if (x + 230 > window.innerWidth)  x = e.clientX - 240;
    if (y2 + 50 > window.innerHeight) y2 = e.clientY - 60;
    _tipEl.style.left = x + 'px'; _tipEl.style.top = y2 + 'px';
  };

  // Self-attn hover: L{q} → L{k}
  const selfHover = (e) => {
    _ensureTip();
    const sa = _percStep?.[`perceiver_self_attn_r${roundIdx}`];
    if (!sa) { _tipEl.style.display = 'none'; return; }
    const c = document.getElementById(`p03-perc-r${roundIdx}-self`);
    const rect = c.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (c.width / rect.width);
    const my = (e.clientY - rect.top)  * (c.height / rect.height);
    const qRow = Math.min(3, Math.floor(my / (PERC_CA_CH + 1)));
    const kCol = Math.min(3, Math.floor(mx / (PERC_SA_CW + 1)));
    const val  = sa[qRow]?.[kCol];
    if (val == null) { _tipEl.style.display = 'none'; return; }
    _tipEl.innerHTML =
      `<b style="color:${TEXT_HI}">L${qRow} (query) → L${kCol} (key)</b><br>` +
      `attention: <b>${Number(val).toFixed(4)}</b>`;
    _tipEl.style.display = 'block';
    let x = e.clientX + 12, y2 = e.clientY + 12;
    if (x + 210 > window.innerWidth)  x = e.clientX - 220;
    if (y2 + 50 > window.innerHeight) y2 = e.clientY - 60;
    _tipEl.style.left = x + 'px'; _tipEl.style.top = y2 + 'px';
  };

  row.appendChild(makeBars(`r${roundIdx}-in`,         inputLabel));
  row.appendChild(arrow());
  row.appendChild(makeHeatmap(`p03-perc-r${roundIdx}-cross`, 'cross-attn\n(L→patches)', crossHover));
  row.appendChild(arrow());
  row.appendChild(makeBars(`r${roundIdx}-after-cross`, 'post-CA'));
  row.appendChild(arrow());
  row.appendChild(makeHeatmap(`p03-perc-r${roundIdx}-self`,  'self-attn\n(L→L)',        selfHover));
  row.appendChild(arrow());
  row.appendChild(makeBars(`r${roundIdx}-out`,         outputLabel));

  wrap.appendChild(row);
}

// Module-level cache so bar hover handlers can read current vecs without closing over step
const _percPercGroupVecs = {};

function _buildPerceiverSection(el) {
  const wrap = _el('div');
  _buildPercRoundRow(wrap, 0, 'h_{t-1}', 'inter');
  _buildPercRoundRow(wrap, 1, 'inter',   'h_t');
  el.appendChild(_section('PERCEIVER RESAMPLER', wrap));
}

function _drawLatentBar(canvas, vec) {
  canvas.width  = LAT_BAR_W;
  canvas.height = LAT_BAR_H;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(LAT_BAR_W, LAT_BAR_H);
  const H_PER_DIM = 2;
  for (let d = 0; d < 128; d++) {
    const [r, g, b] = window.JEPA.embRgb(vec[d]);
    for (let py = 0; py < H_PER_DIM; py++) {
      for (let px = 0; px < LAT_BAR_W; px++) {
        const base = ((d * H_PER_DIM + py) * LAT_BAR_W + px) * 4;
        img.data[base]=r; img.data[base+1]=g; img.data[base+2]=b; img.data[base+3]=255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}

function _drawLatentBarAt(canvas, vec, w, h) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);
  const hPerDim = h / 128;
  for (let d = 0; d < 128; d++) {
    const [r, g, b] = window.JEPA.embRgb(vec[d] ?? 0);
    const py0 = Math.floor(d * hPerDim);
    const py1 = Math.max(py0 + 1, Math.floor((d + 1) * hPerDim));
    for (let py = py0; py < py1; py++) {
      for (let px = 0; px < w; px++) {
        const base = (py * w + px) * 4;
        img.data[base]=r; img.data[base+1]=g; img.data[base+2]=b; img.data[base+3]=255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}

let _percStep = null;

function _refreshPerceiverPanel() {
  if (!_percStep) return;
  const step = _percStep;

  // Map group-prefix → array of 4 vecs (4, 128)
  const groups = {
    'r0-in':          step.perceiver_input_queries,
    'r0-after-cross': step.perceiver_after_cross_r0,
    'r0-out':         step.perceiver_inter_r0,
    'r1-in':          step.perceiver_inter_r0,
    'r1-after-cross': step.perceiver_after_cross_r1,
    'r1-out':         step.latent_vectors,
  };

  // Cache vecs for bar hover handlers
  Object.assign(_percPercGroupVecs, groups);

  // Draw latent bars for each group
  for (const [prefix, vecs] of Object.entries(groups)) {
    if (!vecs) continue;
    for (let li = 0; li < 4; li++) {
      const c = document.getElementById(`p03-perc-${prefix}-bar-${li}`);
      if (c && vecs[li]) _drawLatentBarAt(c, vecs[li], PERC_BAR_W, PERC_BAR_H);
    }
  }

  // Draw cross-attn heatmaps (4 rows × 16 cols)
  for (const r of [0, 1]) {
    const ca = step[`perceiver_cross_attn_r${r}`];
    const cc = document.getElementById(`p03-perc-r${r}-cross`);
    if (cc && ca) _drawHeatmap2D(cc, ca, PERC_CA_CW, PERC_CA_CH);

    const sa = step[`perceiver_self_attn_r${r}`];
    const sc = document.getElementById(`p03-perc-r${r}-self`);
    if (sc) {
      const mat = (sa && sa.length === 4) ? sa
        : [[.25,.25,.25,.25],[.25,.25,.25,.25],[.25,.25,.25,.25],[.25,.25,.25,.25]];
      _drawHeatmap2D(sc, mat, PERC_SA_CW, PERC_CA_CH);
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════
// SECTION C — PREDICTOR (4-column per-latent evolution)
// ════════════════════════════════════════════════════════════════════════════

const PRED_DIM_PX  = 2;   // px per latent dim (128 × 2 = 256px wide)
const PRED_BAR_H   = 10;  // px tall per time-step row
const PRED_BAR_GAP = 2;   // px vertical gap between rows
const PRED_LBL_W   = 44;  // px for the text label column

function _cosSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

function _buildPredictorSection(el) {
  const wrap = _el('div');

  const colsRow = _el('div', { id: 'p03-pred-cols' }, {
    display: 'flex', gap: '16px', alignItems: 'flex-start', flexWrap: 'wrap',
  });

  for (let li = 0; li < 4; li++) {
    const col = _el('div', {}, { display: 'flex', flexDirection: 'column' });

    // Header
    const hdr = _el('div', {}, {
      fontSize: '10px', color: TEXT_HI, marginBottom: '4px', fontWeight: 'bold',
      paddingLeft: PRED_LBL_W + 4 + 'px',
    });
    hdr.textContent = `L${li}`;
    col.appendChild(hdr);

    // Labels (left) + canvas (right)
    const vizRow = _el('div', {}, { display: 'flex', alignItems: 'flex-start', gap: '4px' });
    vizRow.appendChild(_el('div', { id: `p03-pred-lblcol-${li}` }, {
      display: 'flex', flexDirection: 'column', gap: PRED_BAR_GAP + 'px',
      width: PRED_LBL_W + 'px', flexShrink: '0',
    }));
    vizRow.appendChild(_el('canvas', { id: `p03-pred-col-${li}`, width: '10', height: '10' }, {
      display: 'block', imageRendering: 'pixelated', border: `1px solid ${BORDER}`, cursor: 'crosshair',
    }));
    col.appendChild(vizRow);

    // Metrics
    const metricsEl = _el('div', {}, {
      paddingLeft: PRED_LBL_W + 4 + 'px', marginTop: '4px', display: 'flex',
      flexDirection: 'column', gap: '1px',
    });
    [
      [`p03-pred-mse-${li}`,      'MSE —'],
      [`p03-pred-cos-pred-${li}`, 'pred·tgt —'],
      [`p03-pred-cos-ht-${li}`,   'h_t·tgt  —'],
    ].forEach(([id, text]) => {
      const d = _el('div', { id }, {
        fontSize: '9px', color: TEXT_DIM, fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap',
      });
      d.textContent = text;
      metricsEl.appendChild(d);
    });
    col.appendChild(metricsEl);
    colsRow.appendChild(col);
  }

  wrap.appendChild(colsRow);
  el.appendChild(_section('PREDICTOR — ODE EVOLUTION (hover for dim values)', wrap));
}

let _predStep  = null;
let _predTipEl = null;

function _drawPredCol(canvas, vecs) {
  const nBars = vecs.length;
  const W = 128 * PRED_DIM_PX;  // dims go left→right
  const H = nBars * PRED_BAR_H + (nBars - 1) * PRED_BAR_GAP;  // time goes top→bottom
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(W, H);

  for (let i = 0; i < img.data.length; i += 4) {
    img.data[i]=13; img.data[i+1]=17; img.data[i+2]=23; img.data[i+3]=255;
  }

  for (let b = 0; b < nBars; b++) {
    const vec = vecs[b];
    const y0 = b * (PRED_BAR_H + PRED_BAR_GAP);
    for (let d = 0; d < 128; d++) {
      const [r, g, bv] = window.JEPA.embRgb(vec[d] ?? 0);
      const x0 = d * PRED_DIM_PX;
      for (let py = y0; py < y0 + PRED_BAR_H; py++) {
        for (let px = x0; px < x0 + PRED_DIM_PX; px++) {
          const base = (py * W + px) * 4;
          img.data[base]=r; img.data[base+1]=g; img.data[base+2]=bv; img.data[base+3]=255;
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}

function _refreshPredictorPanel() {
  if (!_predStep) return;
  const step = _predStep;
  const traj = step.ode_trajectory;
  const tgt  = step.h_target_latents;

  for (let li = 0; li < 4; li++) {
    const el = document.getElementById(`p03-pred-mse-${li}`);
    const v  = step.per_latent_pred_error?.[li];
    if (el) el.textContent = `MSE ${v != null ? Number(v).toExponential(3) : '—'}`;
  }

  if (!traj || !tgt || !traj.length) return;

  const nSteps = traj.length;
  const stepLabels = traj.map((_, k) => {
    if (k === 0) return 'h_t';
    if (k === nSteps - 1) return 'pred';
    return `τ=${(k / (nSteps - 1)).toFixed(2)}`;
  });
  stepLabels.push('tgt');

  if (!_predTipEl) {
    _predTipEl = _el('div', {}, {
      position: 'fixed', display: 'none', zIndex: '9999',
      background: '#1c2128', border: `1px solid ${BORDER}`,
      borderRadius: '5px', padding: '6px 10px',
      fontSize: '10px', color: TEXT,
      fontFamily: "'Menlo','Monaco','Consolas',monospace",
      pointerEvents: 'none',
    });
    document.body.appendChild(_predTipEl);
  }
  const tip = _predTipEl;

  for (let li = 0; li < 4; li++) {
    const canvas = document.getElementById(`p03-pred-col-${li}`);
    if (!canvas) continue;

    const vecs = [...traj.map(t => t[li]), tgt[li]];
    _drawPredCol(canvas, vecs);

    // Step labels to the left (one per row, height matches PRED_BAR_H)
    const lblCol = document.getElementById(`p03-pred-lblcol-${li}`);
    if (lblCol) {
      lblCol.innerHTML = '';
      for (let b = 0; b < vecs.length; b++) {
        const lbl = _el('div', {}, {
          height: PRED_BAR_H + 'px', lineHeight: PRED_BAR_H + 'px',
          fontSize: '8px', color: TEXT_DIM, textAlign: 'right',
          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
        });
        lbl.textContent = stepLabels[b];
        lbl.title = stepLabels[b];
        lblCol.appendChild(lbl);
      }
    }

    // Cosine similarities
    const cosP = _cosSim(traj[nSteps - 1][li], tgt[li]);
    const cosH = _cosSim(traj[0][li], tgt[li]);
    const cosPEl = document.getElementById(`p03-pred-cos-pred-${li}`);
    const cosHEl = document.getElementById(`p03-pred-cos-ht-${li}`);
    if (cosPEl) cosPEl.textContent = `pred·tgt ${cosP.toFixed(4)}`;
    if (cosHEl) cosHEl.textContent = `h_t·tgt  ${cosH.toFixed(4)}`;

    // Hover: dim goes left→right (X), time goes top→bottom (Y)
    canvas.onmousemove = (e) => {
      const rect = canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (canvas.width  / rect.width);
      const my = (e.clientY - rect.top)  * (canvas.height / rect.height);
      const barIdx = Math.min(vecs.length - 1, Math.floor(my / (PRED_BAR_H + PRED_BAR_GAP)));
      const dimIdx = Math.min(127, Math.floor(mx / PRED_DIM_PX));
      const vec = vecs[barIdx];
      if (!vec || dimIdx < 0) { tip.style.display = 'none'; return; }
      const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
      tip.innerHTML =
        `<b style="color:${TEXT_HI}">L${li} — ${stepLabels[barIdx]}</b><br>` +
        `dim <b>${dimIdx}</b>: <b>${Number(vec[dimIdx]).toFixed(4)}</b><br>` +
        `norm=<b>${norm.toFixed(3)}</b>`;
      tip.style.display = 'block';
      let x = e.clientX + 12, y2 = e.clientY + 12;
      if (x + 220 > window.innerWidth)  x = e.clientX - 230;
      if (y2 + 70 > window.innerHeight) y2 = e.clientY - 80;
      tip.style.left = x + 'px'; tip.style.top = y2 + 'px';
    };
    canvas.onmouseleave = () => { tip.style.display = 'none'; };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// SECTION D — POLICY
// ════════════════════════════════════════════════════════════════════════════

function _drawActionBars(canvas, probs, taken, available) {
  const w = 220, h = 84;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = BG; ctx.fillRect(0, 0, w, h);
  const n = probs.length;
  const barH = Math.floor((h - (n + 1) * 4) / n);
  const avSet = new Set((available || []).map(a => a - 1));
  const colors = ['#388bfd','#f85149','#3fb950','#f0c040'];
  for (let i = 0; i < n; i++) {
    const y = 4 + i * (barH + 4);
    ctx.fillStyle = TEXT_DIM; ctx.font = '10px monospace';
    ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
    ctx.fillText(`A${i}`, 2, y + barH / 2);
    const barX = 24, barW = w - barX - 50;
    ctx.fillStyle = '#161b22'; ctx.fillRect(barX, y, barW, barH);
    const fill = Math.round(probs[i] * barW);
    ctx.fillStyle = i === taken ? '#f0c040' : (!avSet.has(i) ? '#484f58' : colors[i % colors.length]);
    ctx.fillRect(barX, y, fill, barH);
    if (i === taken) {
      ctx.fillStyle = '#fff'; ctx.font = 'bold 9px monospace';
      ctx.fillText('★', barX + fill + 2, y + barH / 2);
    }
    ctx.fillStyle = TEXT; ctx.font = '9px monospace'; ctx.textAlign = 'right';
    ctx.fillText((probs[i] * 100).toFixed(1) + '%', w - 2, y + barH / 2);
    ctx.textAlign = 'left';
  }
}

function _buildPolicySection(el) {
  const wrap = _el('div');
  const c = _el('canvas', { id: 'p03-action-bars', width: '220', height: '84' },
    { display: 'block' });
  const meta = _el('div', {}, {
    marginTop: '4px', fontSize: '11px', color: TEXT_DIM, display: 'flex', gap: '10px',
  });
  const entSpan = _el('span');
  entSpan.innerHTML = `Entropy <span id="p03-entropy" style="color:${TEXT_HI}">—</span>`;
  const actSpan = _el('span');
  actSpan.innerHTML = `Sampled <span id="p03-action-taken" style="color:${TEXT_HI}">—</span>`;
  meta.appendChild(entSpan); meta.appendChild(actSpan);
  wrap.appendChild(c); wrap.appendChild(meta);
  el.appendChild(_section('POLICY', wrap));
}

// ════════════════════════════════════════════════════════════════════════════
// EXPORTED API
// ════════════════════════════════════════════════════════════════════════════

export function render(data, el) {
  // Reset state on each new episode
  _saLayer    = 1;
  _selSAPatch = null;
  _saStep = _percStep = _predStep = null;

  el.innerHTML = '';
  el.style.fontFamily = "'Menlo','Monaco','Consolas',monospace";
  el.style.fontSize   = '12px';

  _buildPerceiverSection(el);
  _buildPredictorSection(el);
  _buildPolicySection(el);
  _buildSASection(el);

  _initSAEvents();

  if (data.timesteps?.length > 0) {
    _updateAll(data.timesteps[0]);
  }
}

export function updateStep(step, t, el) {
  _updateAll(step);
}

function _updateAll(step) {
  // SA
  _saStep = step;
  _refreshSAGrid();
  const es = step.embedding_summary;
  if (es) {
    const setText = (id, v, dp = 4) => {
      const el2 = document.getElementById(id);
      if (el2) el2.textContent = v != null ? Number(v).toFixed(dp) : '—';
    };
    setText('p03-es-cos',  es.mean_pairwise_cos_sim);
    setText('p03-es-rank', es.effective_rank, 2);
    const dd = document.getElementById('p03-es-dead');
    if (dd) dd.textContent = es.dead_dim_count != null ? es.dead_dim_count : '—';
    setText('p03-es-drift', es.mean_embedding_drift);
  }

  // Perceiver
  _percStep = step;
  _refreshPerceiverPanel();

  // Predictor
  _predStep = step;
  _refreshPredictorPanel();

  // Policy
  const probs   = step.action_probs;
  const taken   = step.action_taken;
  const avail   = step.available_actions;
  const entropy = step.action_entropy;
  const ac = document.getElementById('p03-action-bars');
  if (ac && probs) _drawActionBars(ac, probs, taken, avail);
  const entEl = document.getElementById('p03-entropy');
  if (entEl) entEl.textContent = entropy != null ? Number(entropy).toFixed(4) : '—';
  const actEl = document.getElementById('p03-action-taken');
  if (actEl) actEl.textContent = taken != null ? `A${taken} ★` : '—';
}
