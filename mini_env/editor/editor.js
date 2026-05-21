/* mini-LS20 Level Editor
 *
 * Static HTML/JS editor for designing 8x8-cell, 32x32-pixel levels for the
 * mini_env package. The preview canvas mirrors the Python renderer at
 * mini_env/renderer.py pixel-for-pixel (the env writes uint8 indices; the RGB
 * colors here come from the canonical 16-color ARC palette used elsewhere in
 * the repo).
 *
 * Visual style now matches LS20 Level 1 on arcprize.org:
 *   - grey walkable bg, near-black walls
 *   - blue player with orange band on the side matching its rotation
 *   - blue goal with a dark frame and an orange dot marking target rotation
 *   - white "+" cross
 *   - yellow energy bar, blue L-mark pattern preview in the UI strip
 *
 * Input: pointer events (works on mouse + trackpad + touch).
 */

"use strict";

// ─── Palette ──────────────────────────────────────────────────────────────
// 16-color ARC palette, mirrors JEPA/experiments/exp_003_*/debug_runner.py.
const ARC_PALETTE = [
  "#000000",   // 0  black
  "#0074D9",   // 1  blue       (player body, goal interior)
  "#FF4136",   // 2  red        (denial flash)
  "#2ECC40",   // 3  green
  "#FFDC00",   // 4  yellow     (energy bar)
  "#AAAAAA",   // 5  grey       (bg / walkable)
  "#F012BE",   // 6  magenta
  "#FF851B",   // 7  orange     (accent band / target-rotation dot)
  "#7FDBFF",   // 8  azure
  "#870C25",   // 9  maroon
  "#3D9970",   // 10 teal
  "#FFFFFF",   // 11 white      (cross)
  "#001F3F",   // 12 navy
  "#01FF70",   // 13 lime       (match-cue highlight)
  "#85144B",   // 14 burgundy
  "#014B65",   // 15 dark teal  (walls — near-black)
];

const ACCENT_IDX = 7;   // hardcoded — mirrors renderer.py:ACCENT

function idxToCss(i) {
  return ARC_PALETTE[i & 0x0f] || "#FF00FF";
}

// ─── Defaults ─────────────────────────────────────────────────────────────
const DEFAULT_PALETTE = {
  bg: 5,            // grey
  wall: 15,         // dark teal (near-black)
  player: 1,        // blue
  goal_frame: 0,    // black
  cross: 11,        // white
  preview_bg: 0,    // black
  highlight: 13,    // lime
  energy: 4,        // yellow
  denial_flash: 2,  // red
};

const GRID_COLS = 8;
const GRID_ROWS = 8;
const PLAYABLE_ROWS = 7;   // row 7 is UI strip
const TILE_PX = 4;
const FRAME_PX = 32;       // GRID_COLS * TILE_PX

function defaultState() {
  return {
    name: "level_01",
    step_limit: 42,
    player_rot: 270,
    goal_rot: 0,
    goal_gated: true,
    show_match_cue: true,
    walls: new Set(),                   // keys: "c,r"
    player: { c: 1, r: 5 },
    goal:   { c: 6, r: 1 },
    cross:  { c: 3, r: 3 },
  };
}

let state = defaultState();
let currentTool = "wall";
let isPainting = false;
let lastPainted = null;   // "c,r" of last cell painted in this drag

// ─── DOM hooks ────────────────────────────────────────────────────────────
const gridEl = document.getElementById("grid");
const canvas = document.getElementById("preview");
const ctx = canvas.getContext("2d");
const toastEl = document.getElementById("toast");
const errorBox = document.getElementById("error-box");

// ─── Grid build ───────────────────────────────────────────────────────────
function buildGrid() {
  gridEl.innerHTML = "";
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const div = document.createElement("div");
      div.className = "cell";
      div.dataset.c = c;
      div.dataset.r = r;
      if (r === PLAYABLE_ROWS) {
        div.classList.add("ui-strip");
        div.title = "UI strip — not editable";
      }
      gridEl.appendChild(div);
    }
  }
}

function keyOf(c, r) { return `${c},${r}`; }

function cellAt(c, r) {
  return gridEl.children[r * GRID_COLS + c];
}

function refreshCell(c, r) {
  if (r === PLAYABLE_ROWS) return;
  const el = cellAt(c, r);
  if (!el) return;
  el.className = "cell";  // reset classes
  el.dataset.c = c;
  el.dataset.r = r;
  // Cell-content class
  if (state.walls.has(keyOf(c, r))) el.classList.add("wall");
  if (state.player && state.player.c === c && state.player.r === r) {
    // Player tile is FIXED — never rotates. Rotation lives only in the
    // pattern preview in the UI strip.
    el.classList.add("player");
  }
  if (state.goal && state.goal.c === c && state.goal.r === r) {
    el.classList.add("goal");
    el.classList.add(`rot${state.goal_rot}`);
  }
  if (state.cross && state.cross.c === c && state.cross.r === r) {
    el.classList.add("cross");
  }
}

function refreshAllCells() {
  for (let r = 0; r < PLAYABLE_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      refreshCell(c, r);
    }
  }
}

// ─── Painting ─────────────────────────────────────────────────────────────
function paintCell(c, r) {
  if (r >= PLAYABLE_ROWS) return;
  if (c < 0 || c >= GRID_COLS || r < 0 || r >= PLAYABLE_ROWS) return;
  const k = keyOf(c, r);

  // Clear any piece already in this cell.
  state.walls.delete(k);
  if (state.player && state.player.c === c && state.player.r === r) state.player = null;
  if (state.goal   && state.goal.c   === c && state.goal.r   === r) state.goal = null;
  if (state.cross  && state.cross.c  === c && state.cross.r  === r) state.cross = null;

  switch (currentTool) {
    case "wall":
      state.walls.add(k);
      break;
    case "player":
      state.player = { c, r };   // singleton — old position cleared above only if same; we need to clear elsewhere
      break;
    case "goal":
      state.goal = { c, r };
      break;
    case "cross":
      state.cross = { c, r };
      break;
    case "erase":
      // Already cleared above.
      break;
  }

  // For singletons (player/goal/cross), make sure ONLY the new cell has them.
  // We do this by scanning all cells in the next refresh.
  refreshAllCells();
  drawPreview();
}

// Where exactly the pointer is in grid coordinates.
function cellFromPoint(x, y) {
  const el = document.elementFromPoint(x, y);
  if (!el || !el.classList.contains("cell")) return null;
  return { c: +el.dataset.c, r: +el.dataset.r };
}

function startPaint(e) {
  if (!e.target || !e.target.classList.contains("cell")) return;
  if (e.target.classList.contains("ui-strip")) return;
  isPainting = true;
  lastPainted = null;
  const c = +e.target.dataset.c;
  const r = +e.target.dataset.r;
  paintCell(c, r);
  lastPainted = keyOf(c, r);
  e.preventDefault();
  // Capture pointer so mouseout doesn't stop the drag.
  if (e.target.setPointerCapture && e.pointerId !== undefined) {
    try { gridEl.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ }
  }
}

function dragPaint(e) {
  if (!isPainting) return;
  const at = cellFromPoint(e.clientX, e.clientY);
  if (!at) return;
  if (at.r >= PLAYABLE_ROWS) return;
  const k = keyOf(at.c, at.r);
  if (k === lastPainted) return;     // don't re-paint the same cell repeatedly
  paintCell(at.c, at.r);
  lastPainted = k;
  e.preventDefault();
}

function endPaint() {
  isPainting = false;
  lastPainted = null;
}

gridEl.addEventListener("pointerdown", startPaint);
gridEl.addEventListener("pointermove", dragPaint);
window.addEventListener("pointerup", endPaint);
window.addEventListener("pointercancel", endPaint);

// Fallback for very old browsers without pointer events.
gridEl.addEventListener("mousedown", startPaint);
gridEl.addEventListener("mousemove", dragPaint);
window.addEventListener("mouseup", endPaint);

// ─── Tool selection ───────────────────────────────────────────────────────
document.getElementById("tools").addEventListener("click", (e) => {
  const btn = e.target.closest(".tool");
  if (!btn) return;
  document.querySelectorAll(".tool").forEach(b => b.classList.remove("selected"));
  btn.classList.add("selected");
  currentTool = btn.dataset.tool;
});

// ─── Form bindings ────────────────────────────────────────────────────────
document.getElementById("name-input").addEventListener("input", (e) => {
  state.name = e.target.value || "level";
});
document.getElementById("step-limit").addEventListener("input", (e) => {
  const v = parseInt(e.target.value, 10);
  state.step_limit = Number.isFinite(v) && v > 0 ? v : 42;
  drawPreview();
});
document.getElementById("player-rot").addEventListener("change", (e) => {
  state.player_rot = parseInt(e.target.value, 10);
  refreshAllCells();
  drawPreview();
});
document.getElementById("goal-rot").addEventListener("change", (e) => {
  state.goal_rot = parseInt(e.target.value, 10);
  refreshAllCells();
  drawPreview();
});
document.getElementById("goal-gated").addEventListener("change", (e) => {
  state.goal_gated = e.target.checked;
});
document.getElementById("show-match-cue").addEventListener("change", (e) => {
  state.show_match_cue = e.target.checked;
  drawPreview();
});

// ─── Preview renderer (port of mini_env/renderer.py) ──────────────────────
function px(imgData, x, y, cssColor) {
  if (x < 0 || x >= FRAME_PX || y < 0 || y >= FRAME_PX) return;
  const r = parseInt(cssColor.slice(1, 3), 16);
  const g = parseInt(cssColor.slice(3, 5), 16);
  const b = parseInt(cssColor.slice(5, 7), 16);
  const i = (y * FRAME_PX + x) * 4;
  imgData.data[i + 0] = r;
  imgData.data[i + 1] = g;
  imgData.data[i + 2] = b;
  imgData.data[i + 3] = 255;
}

function fillRect(imgData, x0, y0, w, h, cssColor) {
  for (let y = y0; y < y0 + h; y++)
    for (let x = x0; x < x0 + w; x++)
      px(imgData, x, y, cssColor);
}

// Build a 4x4 boolean mask for the rotation-0 reference L-mark, then rotate
// `n` times 90° CLOCKWISE. Matches Python's `np.rot90(arr, k=-n)`.
function rotateCW(mask) {
  const n = mask.length;
  const out = Array.from({ length: n }, () => Array(n).fill(false));
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++)
      out[c][n - 1 - r] = mask[r][c];
  return out;
}

function rotatedLMask(rotation) {
  const m0 = [
    [false, true,  false, false],
    [false, true,  true,  false],
    [false, false, false, false],
    [false, false, false, false],
  ];
  let m = m0;
  const steps = (((rotation % 360) + 360) / 90 | 0) % 4;
  for (let i = 0; i < steps; i++) m = rotateCW(m);
  return m;
}

function drawPlayerOnPreview(img, c, r, bodyCss) {
  // FIXED orientation: top 2 rows orange, bottom 2 rows blue. Does NOT rotate
  // — rotation is only visible in the preview L-mark in the UI strip.
  const accentCss = idxToCss(ACCENT_IDX);
  const x0 = c * TILE_PX, y0 = r * TILE_PX;
  fillRect(img, x0, y0, TILE_PX, TILE_PX, bodyCss);
  fillRect(img, x0, y0, TILE_PX, 2, accentCss);
}

function drawGoalOnPreview(img, c, r, bodyCss, rotation) {
  // Blue body + 3-pixel orange L-mark rotated by `rotation` — mirrors the
  // pattern preview, so when goal-L visually equals preview-L the puzzle is
  // solved.
  const accentCss = idxToCss(ACCENT_IDX);
  const x0 = c * TILE_PX, y0 = r * TILE_PX;
  fillRect(img, x0, y0, TILE_PX, TILE_PX, bodyCss);
  const mask = rotatedLMask(rotation);
  for (let dy = 0; dy < TILE_PX; dy++)
    for (let dx = 0; dx < TILE_PX; dx++)
      if (mask[dy][dx]) px(img, x0 + dx, y0 + dy, accentCss);
}

function drawPreview() {
  const pal = DEFAULT_PALETTE;
  const cssBg     = idxToCss(pal.bg);
  const cssWall   = idxToCss(pal.wall);
  const cssCross  = idxToCss(pal.cross);
  const cssGoalFr = idxToCss(pal.goal_frame);
  const cssHL     = idxToCss(pal.highlight);
  const cssPlayer = idxToCss(pal.player);
  const cssPBg    = idxToCss(pal.preview_bg);
  const cssEnergy = idxToCss(pal.energy);

  const img = ctx.createImageData(FRAME_PX, FRAME_PX);

  // 1. background
  fillRect(img, 0, 0, FRAME_PX, FRAME_PX, cssBg);

  // 2. walls
  for (const k of state.walls) {
    const [c, r] = k.split(",").map(Number);
    if (r >= PLAYABLE_ROWS) continue;
    fillRect(img, c * TILE_PX, r * TILE_PX, TILE_PX, TILE_PX, cssWall);
  }

  // 3. cross
  if (state.cross) {
    const c = state.cross.c, r = state.cross.r;
    const x0 = c * TILE_PX, y0 = r * TILE_PX;
    for (let dx = 0; dx < TILE_PX; dx++) {
      px(img, x0 + dx, y0 + 1, cssCross);
      px(img, x0 + dx, y0 + 2, cssCross);
    }
    for (let dy = 0; dy < TILE_PX; dy++) {
      px(img, x0 + 1, y0 + dy, cssCross);
      px(img, x0 + 2, y0 + dy, cssCross);
    }
  }

  // 4. goal (blue body + rotated orange L-mark)
  if (state.goal) {
    drawGoalOnPreview(img, state.goal.c, state.goal.r, cssPlayer, state.goal_rot);
  }

  // 5. match-cue ring (drawn AFTER goal, BEFORE player)
  if (state.show_match_cue && state.goal && state.player_rot === state.goal_rot) {
    const c = state.goal.c, r = state.goal.r;
    const x0 = c * TILE_PX, y0 = r * TILE_PX;
    for (let dx = -1; dx <= TILE_PX; dx++) {
      px(img, x0 + dx, y0 - 1,            cssHL);
      px(img, x0 + dx, y0 + TILE_PX,      cssHL);
    }
    for (let dy = -1; dy <= TILE_PX; dy++) {
      px(img, x0 - 1,            y0 + dy, cssHL);
      px(img, x0 + TILE_PX,      y0 + dy, cssHL);
    }
  }

  // 6. player (FIXED two-tone — does NOT rotate)
  if (state.player) {
    drawPlayerOnPreview(img, state.player.c, state.player.r, cssPlayer);
  }

  // 7. UI strip — LS20 layout: preview LEFT (cols 0-3), energy bar RIGHT (cols 4-31).
  fillRect(img, 0, 28, FRAME_PX, 4, cssPBg);

  // 7a. Pattern preview L-mark at rows 28..31, cols 0..3, in player colour, rotated by player_rot.
  const mask = rotatedLMask(state.player_rot);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      if (mask[r][c]) px(img, c, 28 + r, cssPlayer);
    }
  }

  // 7b. Energy bar at row 29, cols 4..31 (28 cols wide, editor preview = full energy).
  for (let x = 4; x < FRAME_PX; x++) px(img, x, 29, cssEnergy);

  ctx.putImageData(img, 0, 0);
}

// ─── Save / Load / Reset ──────────────────────────────────────────────────
function buildJson() {
  if (!state.player || !state.goal || !state.cross) {
    throw new Error("Level requires a player, goal, and cross before saving.");
  }
  const walls = Array.from(state.walls).map(k => k.split(",").map(Number));
  return {
    name: state.name,
    grid_cells: [GRID_COLS, GRID_ROWS],
    tile_px: TILE_PX,
    step_limit: state.step_limit,
    palette: { ...DEFAULT_PALETTE },
    walls,
    player_start: { cell: [state.player.c, state.player.r], rotation: state.player_rot },
    goal:         { cell: [state.goal.c,   state.goal.r],   rotation: state.goal_rot },
    cross:        { cell: [state.cross.c,  state.cross.r] },
    goal_gated: state.goal_gated,
    show_match_cue: state.show_match_cue,
  };
}

function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  setTimeout(() => toastEl.classList.remove("show"), 1800);
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
  setTimeout(() => { errorBox.hidden = true; }, 6000);
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

document.getElementById("save-btn").addEventListener("click", () => {
  clearError();
  try {
    const obj = buildJson();
    const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${state.name || "level"}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    showToast("✓ Downloaded");
  } catch (e) {
    showError(e.message || String(e));
  }
});

document.getElementById("load-btn").addEventListener("click", () => {
  document.getElementById("load-file").click();
});

document.getElementById("load-file").addEventListener("change", (e) => {
  clearError();
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const obj = JSON.parse(reader.result);
      validateAndLoad(obj);
      e.target.value = "";
      showToast("✓ Loaded");
    } catch (err) {
      showError("Failed to load: " + (err.message || String(err)));
      e.target.value = "";
    }
  };
  reader.onerror = () => showError("Failed to read file");
  reader.readAsText(file);
});

function validateAndLoad(obj) {
  const required = ["name", "grid_cells", "tile_px", "step_limit",
                    "palette", "walls", "player_start", "goal", "cross",
                    "goal_gated", "show_match_cue"];
  for (const k of required) {
    if (!(k in obj)) throw new Error(`Missing key: ${k}`);
  }
  if (!Array.isArray(obj.grid_cells) || obj.grid_cells[0] !== GRID_COLS || obj.grid_cells[1] !== GRID_ROWS) {
    throw new Error(`grid_cells must be [${GRID_COLS}, ${GRID_ROWS}]`);
  }
  if (!Array.isArray(obj.walls)) throw new Error("walls must be an array");
  for (const w of obj.walls) {
    if (!Array.isArray(w) || w.length !== 2) throw new Error("wall entry must be [col, row]");
    if (w[1] >= PLAYABLE_ROWS) throw new Error(`wall at row ${w[1]} is on the UI strip`);
  }
  const checkCell = (label, cell) => {
    if (!Array.isArray(cell) || cell.length !== 2) throw new Error(`${label}.cell must be [col, row]`);
    if (cell[0] < 0 || cell[0] >= GRID_COLS) throw new Error(`${label}.cell col out of range`);
    if (cell[1] < 0 || cell[1] >= PLAYABLE_ROWS) throw new Error(`${label}.cell row out of range or on UI strip`);
  };
  checkCell("player_start", obj.player_start.cell);
  checkCell("goal",         obj.goal.cell);
  checkCell("cross",        obj.cross.cell);
  const allowedRot = new Set([0, 90, 180, 270]);
  if (!allowedRot.has(obj.player_start.rotation)) throw new Error("player_start.rotation must be 0/90/180/270");
  if (!allowedRot.has(obj.goal.rotation))         throw new Error("goal.rotation must be 0/90/180/270");

  state.name           = obj.name;
  state.step_limit     = obj.step_limit;
  state.player_rot     = obj.player_start.rotation;
  state.goal_rot       = obj.goal.rotation;
  state.goal_gated     = !!obj.goal_gated;
  state.show_match_cue = !!obj.show_match_cue;
  state.walls = new Set(obj.walls.map(([c, r]) => keyOf(c, r)));
  state.player = { c: obj.player_start.cell[0], r: obj.player_start.cell[1] };
  state.goal   = { c: obj.goal.cell[0],         r: obj.goal.cell[1] };
  state.cross  = { c: obj.cross.cell[0],        r: obj.cross.cell[1] };

  syncFormFromState();
  refreshAllCells();
  drawPreview();
}

document.getElementById("reset-btn").addEventListener("click", () => {
  clearError();
  state = defaultState();
  syncFormFromState();
  refreshAllCells();
  drawPreview();
  showToast("Reset");
});

function syncFormFromState() {
  document.getElementById("name-input").value = state.name;
  document.getElementById("step-limit").value = state.step_limit;
  document.getElementById("player-rot").value = String(state.player_rot);
  document.getElementById("goal-rot").value   = String(state.goal_rot);
  document.getElementById("goal-gated").checked     = state.goal_gated;
  document.getElementById("show-match-cue").checked = state.show_match_cue;
}

// ─── Boot ─────────────────────────────────────────────────────────────────
buildGrid();
syncFormFromState();
refreshAllCells();
drawPreview();
