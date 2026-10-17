#!/usr/bin/env python3
"""
Bondad de Ajuste Chi-Cuadrado con Dados — Pipeline de Visión por Computadora
=============================================================================
Une la detección de dados en tiempo real (OpenCV / Roboflow Inference) con
pruebas de hipótesis de Bondad de Ajuste Chi-Cuadrado (χ²).

Todas las estructuras de detección, anotaciones, HUD y exportación de
fotogramas usan la librería **supervision** de Roboflow (``import supervision as sv``).

Uso
---
    python main.py                  # Modo OpenCV (predeterminado, sin configuración)
    python main.py --mode roboflow  # Modo Roboflow Inference

Controles (ventana en vivo)
---------------------------
    ESPACIO  Registrar la tirada actual de 5 dados en SQLite y guardar el fotograma
    S        Sembrar tiradas simuladas + 3 fotogramas de muestra
    T        Alternar entre cámara en vivo y dados sintéticos
    A        Cerrar el video → análisis χ² completo + panel de resultados
    Q        Salir
"""
from __future__ import annotations

import argparse
import datetime
from collections import Counter, deque
import math
import os
import random
import re
import sqlite3
import sys
import textwrap

import cv2
import matplotlib
import numpy as np
import supervision as sv
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# The report uses box-drawing / Greek characters; on Windows a redirected or
# legacy console defaults to cp1252 and would raise UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ═══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
DB_PATH: str = "dice_experiments.db"
CAPTURES_DIR: str = "linkedin_captures"
DASHBOARD_PATH: str = "dashboard_bondad_ajuste.png"
ALPHA: float = 0.05
MIN_EXPECTED: float = 5.0          # pooling threshold: every E_i >= 5
MIN_ROLLS: int = 30                # n/6 >= 5 so Test 1 satisfies E_i >= 5
DEFAULT_RF_MODEL: str = "dice-yolov5/2"
# OpenCV pip counter tuning
PIP_BLACKHAT_FRAC: float = 0.35   # black-hat kernel as a fraction of die size
PIP_MIN_CONTRAST: int = 25        # min darkness of a pip vs. its face (gray lvls)
FACE_TOLERANCES: tuple[int, ...] = (25, 35, 45)  # top-face flood-fill ranges
FACE_EDGE_MARGIN: float = 0.06    # pip centre must be this far (x die size) inside the face
STABLE_FRAMES: int = 10           # live readings voted on before logging
FRAME_W: int = 960
FRAME_H: int = 540
DIE_SIZE: int = 80
DIE_GAP: int = 30

# ═══════════════════════════════════════════════════════════════════════
#  SQLITE HELPERS
# ═══════════════════════════════════════════════════════════════════════

def init_db() -> sqlite3.Connection:
    """Create (or open) the SQLite database and ensure the *rolls* table exists."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rolls (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            d1        INT,
            d2        INT,
            d3        INT,
            d4        INT,
            d5        INT,
            total_sum INT,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    return conn


def insert_roll(conn: sqlite3.Connection, values: list[int]) -> None:
    """Insert a single 5-dice roll into the database."""
    ts = datetime.datetime.now().isoformat()
    conn.execute(
        "INSERT INTO rolls (d1,d2,d3,d4,d5,total_sum,timestamp) "
        "VALUES (?,?,?,?,?,?,?)",
        (*values, sum(values), ts),
    )
    conn.commit()


def count_rolls(conn: sqlite3.Connection) -> int:
    """Return total number of logged rolls."""
    return conn.execute("SELECT COUNT(*) FROM rolls").fetchone()[0]


def fetch_all_rolls(conn: sqlite3.Connection) -> list[tuple]:
    """Fetch every roll as ``(d1, d2, d3, d4, d5, total_sum)``."""
    return conn.execute(
        "SELECT d1, d2, d3, d4, d5, total_sum FROM rolls"
    ).fetchall()


def seed_rolls(conn: sqlite3.Connection, n: int = 100) -> None:
    """Generate and insert *n* realistic 5-dice rolls."""
    for _ in range(n):
        vals = [random.randint(1, 6) for _ in range(5)]
        insert_roll(conn, vals)


# ═══════════════════════════════════════════════════════════════════════
#  SYNTHETIC DICE FRAME GENERATOR
# ═══════════════════════════════════════════════════════════════════════

# Pip (dot) positions on each die face, expressed as fractions of die size.
PIP_LAYOUTS: dict[int, list[tuple[float, float]]] = {
    1: [(0.50, 0.50)],
    2: [(0.28, 0.28), (0.72, 0.72)],
    3: [(0.28, 0.28), (0.50, 0.50), (0.72, 0.72)],
    4: [(0.28, 0.28), (0.72, 0.28), (0.28, 0.72), (0.72, 0.72)],
    5: [(0.28, 0.28), (0.72, 0.28), (0.50, 0.50), (0.28, 0.72), (0.72, 0.72)],
    6: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.50), (0.75, 0.50),
        (0.25, 0.75), (0.75, 0.75)],
}


def _draw_die_on_frame(
    frame: np.ndarray, cx: int, cy: int, size: int, pips: int
) -> None:
    """Render a single die face centred at ``(cx, cy)``."""
    half = size // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    # Die body (white square)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 245, 245), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (170, 170, 170), 2)
    # Rounded corners (small filled circles at corners)
    corner_r = 6
    for corner in [(x1 + corner_r, y1 + corner_r), (x2 - corner_r, y1 + corner_r),
                   (x1 + corner_r, y2 - corner_r), (x2 - corner_r, y2 - corner_r)]:
        cv2.circle(frame, corner, corner_r, (245, 245, 245), -1)
    # Pips
    pip_r = max(size // 10, 4)
    for px, py in PIP_LAYOUTS.get(pips, PIP_LAYOUTS[1]):
        dot_x = int(x1 + px * size)
        dot_y = int(y1 + py * size)
        cv2.circle(frame, (dot_x, dot_y), pip_r, (25, 25, 25), -1)


def generate_synthetic_frame(
    values: list[int] | None = None,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """
    Render 5 dice on a dark-background frame.

    Returns
    -------
    frame : np.ndarray  — BGR image (FRAME_H × FRAME_W × 3)
    values : list[int]  — pip values [v1..v5]
    xyxy : np.ndarray   — bounding boxes (5 × 4, float32)
    """
    if values is None:
        values = [random.randint(1, 6) for _ in range(5)]

    frame = np.full((FRAME_H, FRAME_W, 3), (35, 42, 52), dtype=np.uint8)
    # Subtle noise for realism
    noise = np.random.randint(0, 8, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)

    # Banner text
    cv2.putText(
        frame, "DADOS SINTETICOS",
        (FRAME_W // 2 - 150, 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (180, 190, 205), 2, cv2.LINE_AA,
    )

    total_width = 5 * DIE_SIZE + 4 * DIE_GAP
    start_x = (FRAME_W - total_width) // 2 + DIE_SIZE // 2
    cy = FRAME_H // 2
    xyxy_list: list[list[int]] = []
    for i, v in enumerate(values):
        cx = start_x + i * (DIE_SIZE + DIE_GAP)
        _draw_die_on_frame(frame, cx, cy, DIE_SIZE, v)
        half = DIE_SIZE // 2
        xyxy_list.append([cx - half, cy - half, cx + half, cy + half])

    return frame, values, np.array(xyxy_list, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════
#  DICE DETECTOR  (Plug-and-Play: OpenCV | Roboflow)
# ═══════════════════════════════════════════════════════════════════════

class DiceDetector:
    """
    Detects up to 5 dice in a BGR frame and returns an ``sv.Detections``
    object whose ``class_id`` and ``data['pips']`` hold the face values.
    """

    _WORD_TO_PIPS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}

    def __init__(
        self, mode: str = "opencv", model_id: str = DEFAULT_RF_MODEL,
    ) -> None:
        self.mode = mode
        self._rf_model = None

        if mode == "roboflow":
            try:
                from inference import get_model  # type: ignore[import-untyped]

                self._rf_model = get_model(
                    model_id=model_id,
                    api_key=os.environ.get("ROBOFLOW_API_KEY"),
                )
                print(f"[INFO] Modelo de Roboflow Inference '{model_id}' cargado.")
            except Exception as exc:
                print(f"[AVISO] No se pudo cargar Roboflow ({exc}). Se usa OpenCV.")
                self.mode = "opencv"

    # ── public API ────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run detection and return ``sv.Detections`` with pip counts."""
        if self.mode == "roboflow" and self._rf_model is not None:
            return self._detect_roboflow(frame)
        return self._detect_opencv(frame)

    # ── Roboflow backend ──────────────────────────────────────────
    def _detect_roboflow(self, frame: np.ndarray) -> sv.Detections:
        results = self._rf_model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        if len(detections) == 0:
            return sv.Detections.empty()

        names = detections.data.get("class_name")
        pips = np.array(
            [
                self._pips_from_class(
                    str(names[i]) if names is not None else "",
                    int(detections.class_id[i]),
                )
                for i in range(len(detections))
            ],
            dtype=int,
        )
        # Drop anything that is not a readable face (1–6)
        valid = (pips >= 1) & (pips <= 6)
        detections = detections[valid]
        pips = pips[valid]
        if len(detections) == 0:
            return sv.Detections.empty()

        # Keep the 5 most confident dice, then order them left→right
        if len(detections) > 5 and detections.confidence is not None:
            top = np.argsort(-detections.confidence)[:5]
            detections, pips = detections[top], pips[top]
        order = np.argsort(detections.xyxy[:, 0])[:5]
        detections, pips = detections[order], pips[order]

        detections.class_id = pips.copy()
        detections.data["pips"] = pips
        return detections

    @classmethod
    def _pips_from_class(cls, name: str, class_id: int) -> int:
        """Map a model class to a face value: '5', 'dice-5', 'five' → 5.
        Falls back to the 0-indexed convention class_id 0..5 → 1..6."""
        digit = re.search(r"[1-6]", name)
        if digit:
            return int(digit.group())
        lowered = name.lower()
        for word, val in cls._WORD_TO_PIPS.items():
            if word in lowered:
                return val
        return class_id + 1

    # ── OpenCV backend ────────────────────────────────────────────
    def _detect_opencv(self, frame: np.ndarray) -> sv.Detections:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Threshold to find bright dice on dark surface
        _, thresh = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        dice: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 600:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            if 0.45 < aspect < 2.2:
                dice.append(((x, y, x + w, y + h), cnt))

        # Sort left→right, keep at most 5
        dice.sort(key=lambda d: d[0][0])
        dice = dice[:5]

        if not dice:
            return sv.Detections.empty()

        boxes = [box for box, _ in dice]
        xyxy = np.array(boxes, dtype=np.float32)
        pips_arr = np.array(
            [self._count_pips(frame, box, cnt) for box, cnt in dice],
            dtype=int,
        )
        return sv.Detections(
            xyxy=xyxy,
            confidence=np.full(len(boxes), 0.90),
            class_id=pips_arr.copy(),
            data={"pips": pips_arr},
        )

    # ── pip counter (top-face segmentation + black-hat) ──────────
    @staticmethod
    def _count_pips(
        frame: np.ndarray,
        box: tuple[int, int, int, int],
        contour: np.ndarray,
    ) -> int:
        """
        Count the pips on the die's TOP face only. Returns 0 if unreadable.

        An oblique camera also sees 1–2 side faces of each die, and counting
        every dark blob in the ROI is what overcounted. Pipeline:
          1. Die mask = filled contour of the detected die (no background).
          2. Pip candidates via black-hat: dark spots smaller than the kernel
             stand out regardless of how each face is shaded; then filtered
             by area (relative to die size) and circularity.
          3. Top face = fixed-range flood fill seeded at the brightest body
             pixel in the upper part of the die. It stops at the shading edge
             between faces; its convex hull is used since a projected square
             face is convex (recovers pips that notch the region's border).
          4. Count candidates whose centroid lies on that face, discarding
             specks much smaller than the median pip.
        """
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        s = min(h, w)
        if s < 12:
            return 0

        # 1. Die mask
        die = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(die, [contour], -1, 255, -1, offset=(-x1, -y1))
        inner = cv2.erode(
            die, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        if cv2.countNonZero(inner) == 0:
            return 0

        # 2. Pip candidates (black-hat → Otsu inside the die)
        k = max(5, int(s * PIP_BLACKHAT_FRAC)) | 1
        blackhat = cv2.morphologyEx(
            gray, cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
        )
        bh_vals = blackhat[inner > 0].reshape(-1, 1)
        otsu_t, _ = cv2.threshold(
            bh_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        pip_mask = np.where(
            (blackhat > max(otsu_t, PIP_MIN_CONTRAST)) & (inner > 0), 255, 0,
        ).astype(np.uint8)
        pip_mask = cv2.morphologyEx(
            pip_mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

        n_lbl, labels, lbl_stats, centroids = cv2.connectedComponentsWithStats(
            pip_mask,
        )
        candidates: list[tuple[float, float, int]] = []
        for i in range(1, n_lbl):
            area = int(lbl_stats[i, cv2.CC_STAT_AREA])
            if not (s * s * 0.004 < area < s * s * 0.08):
                continue
            blob = np.where(labels == i, 255, 0).astype(np.uint8)
            cnts, _ = cv2.findContours(
                blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
            )
            perim = cv2.arcLength(cnts[0], True)
            circularity = 4.0 * math.pi * area / max(perim * perim, 1.0)
            if circularity < 0.45:
                continue
            candidates.append(
                (float(centroids[i][0]), float(centroids[i][1]), area),
            )
        if not candidates:
            return 0

        # 3. Top-face segmentation by region growing
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        body = (inner > 0) & (
            cv2.dilate(pip_mask, np.ones((5, 5), np.uint8)) == 0
        )
        ys, xs = np.nonzero(body)
        if len(ys) == 0:
            return 0
        upper = ys <= ys.min() + 0.35 * (ys.max() - ys.min())
        j = int(np.argmax(np.where(upper, smooth[ys, xs].astype(int), -1)))
        seed = (int(xs[j]), int(ys[j]))

        src = smooth.copy()
        src[inner == 0] = 0  # walls off the background
        die_area = cv2.countNonZero(inner)
        face = np.zeros((h, w), dtype=np.uint8)
        for tol in FACE_TOLERANCES:
            ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(
                src, ff_mask, seed, 255, loDiff=tol, upDiff=tol,
                flags=4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
                | (255 << 8),
            )
            face = ff_mask[1:-1, 1:-1]
            if cv2.countNonZero(face) >= 0.20 * die_area:
                break

        hull = cv2.convexHull(cv2.findNonZero(face))

        # 4. Pips whose centre lies clearly inside the top face (a margin
        #    keeps side-face pips that graze the hull edge out), minus specks
        margin = FACE_EDGE_MARGIN * s
        top = [
            c for c in candidates
            if cv2.pointPolygonTest(hull, (c[0], c[1]), True) >= margin
        ]
        if top:
            median_area = float(np.median([c[2] for c in top]))
            top = [c for c in top if c[2] >= 0.35 * median_area]
        count = len(top)
        return count if 1 <= count <= 6 else 0


# ═══════════════════════════════════════════════════════════════════════
#  SUPERVISION ANNOTATION + HUD HELPERS
# ═══════════════════════════════════════════════════════════════════════

def build_annotated_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    box_ann: sv.RoundBoxAnnotator,
    lbl_ann: sv.LabelAnnotator,
    rolls_logged: int,
    current_sum: int,
    mode_tag: str,
) -> np.ndarray:
    """
    Annotate *frame* with bounding boxes, per-die labels, and a
    bottom telemetry HUD bar — all using the ``supervision`` API.
    """
    # Per-die labels
    labels = [
        f"Dado {i + 1}: {p}" if 1 <= p <= 6 else f"Dado {i + 1}: ?"
        for i, p in enumerate(detections.data.get("pips", []))
    ]

    out = box_ann.annotate(scene=frame.copy(), detections=detections)
    out = lbl_ann.annotate(scene=out, detections=detections, labels=labels)

    # ── Telemetry HUD bar ─────────────────────────────────────────
    hud_text = (
        f"Tiradas registradas (n): {rolls_logged}  |  "
        f"Suma actual de 5 dados: {current_sum}  |  "
        f"Modo: {mode_tag}"
    )
    out = sv.draw_filled_rectangle(
        scene=out,
        rect=sv.Rect(x=0, y=FRAME_H - 44, width=FRAME_W, height=44),
        color=sv.Color(r=0, g=0, b=0),
        opacity=0.72,
    )
    out = sv.draw_text(
        scene=out,
        text=hud_text,
        text_anchor=sv.Point(x=FRAME_W // 2, y=FRAME_H - 22),
        text_color=sv.Color(r=0, g=255, b=140),
        text_scale=0.55,
        text_thickness=2,
        text_padding=6,
    )
    return out


# ═══════════════════════════════════════════════════════════════════════
#  CHI-SQUARE (χ²) GOODNESS-OF-FIT ENGINE
# ═══════════════════════════════════════════════════════════════════════

_SEP_W = 96


def _banner(title: str) -> None:
    print()
    print("═" * _SEP_W)
    print(f"  {title}")
    print("═" * _SEP_W)


def _step(title: str) -> None:
    print(f"\n  ▸ {title}")


def _print_table(
    headers: list[str],
    rows: list[list[str]],
    footer: list[str] | None = None,
    indent: int = 4,
) -> None:
    """Box-drawn table. Cells are pre-formatted strings; first column is
    left-aligned, numeric columns right-aligned."""
    every = [headers] + rows + ([footer] if footer else [])
    widths = [max(len(r[c]) for r in every) for c in range(len(headers))]
    pad = " " * indent

    def rule(left: str, mid: str, right: str) -> str:
        return pad + left + mid.join("─" * (w + 2) for w in widths) + right

    def fmt(row: list[str]) -> str:
        cells = [
            f" {cell:<{w}} " if c == 0 else f" {cell:>{w}} "
            for c, (cell, w) in enumerate(zip(row, widths))
        ]
        return pad + "│" + "│".join(cells) + "│"

    print(rule("┌", "┬", "┐"))
    print(fmt(headers))
    print(rule("├", "┼", "┤"))
    for row in rows:
        print(fmt(row))
    if footer:
        print(rule("├", "┼", "┤"))
        print(fmt(footer))
    print(rule("└", "┴", "┘"))


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _fmt_interval(lo: float, hi: float, closed_right: bool) -> str:
    return f"[{lo:g}, {hi:g}{']' if closed_right else ')'}"


def _print_decision(
    chi2_calc: float, chi2_crit: float, p_val: float, df: int, conclusion: str,
    first_step: int,
) -> tuple[bool, str]:
    rejected = chi2_calc > chi2_crit
    decision = "SE RECHAZA H0" if rejected else "NO SE RECHAZA H0"
    _step(f"Paso {first_step} · Valor crítico y valor p")
    print(
        f"    χ²_crít = χ²({1 - ALPHA:.2f}; gl = {df}) = {chi2_crit:.4f}"
        f"   (probabilidad acumulada {1 - ALPHA:.2f}, α = {ALPHA})"
    )
    print(f"    valor p = P(χ²_{df} ≥ {chi2_calc:.4f}) = {p_val:.6f}")
    _step(f"Paso {first_step + 1} · Decisión")
    print("    Regla: se rechaza H0 si χ²_calc > χ²_crít  (equivalente: valor p < α)")
    cmp_stat = ">" if rejected else "≤"
    cmp_p = "<" if p_val < ALPHA else "≥"
    print(
        f"    {chi2_calc:.4f} {cmp_stat} {chi2_crit:.4f}   y   "
        f"p = {p_val:.4f} {cmp_p} {ALPHA}   →   >>> {decision} <<<"
    )
    for line in textwrap.wrap(conclusion, _SEP_W - 6):
        print(f"    {line}")
    return rejected, decision


# ── Test 1: Discrete Uniform GoF (single die d1) ─────────────────────

def _test_uniform(d1_values: list[int], n: int) -> dict:
    """H0: die d1 is fair → P(face = x_i) = 1/6 for x_i ∈ {1..6}."""
    _banner("PRUEBA 1 — Bondad de Ajuste Uniforme Discreta  ·  un solo dado (d1)")
    print("\n  Hipótesis")
    print("    H0: d1 es un dado justo — X ~ Uniforme{1,…,6},  P(X = x_i) = 1/6 para cada cara")
    print("    H1: d1 no es justo — P(X = x_i) ≠ 1/6 para al menos una cara")
    print(
        f"    Nivel de significación α = {ALPHA}  →  valor crítico con "
        f"probabilidad acumulada {1 - ALPHA:.2f}"
    )

    faces = list(range(1, 7))
    observed = [d1_values.count(f) for f in faces]
    p_e = 1.0 / 6.0
    expected = [n * p_e] * 6
    contribs = [(o - e) ** 2 / e for o, e in zip(observed, expected)]
    chi2_calc = sum(contribs)
    r, k = 6, 0
    df = r - k - 1
    chi2_crit = float(stats.chi2.ppf(1 - ALPHA, df))
    p_val = float(stats.chi2.sf(chi2_calc, df))

    _step(f"Paso 1 · Tabla de frecuencias  (n = {n} tiradas de d1)")
    _print_table(
        ["Cara x_i", "O_i", "P_e", "E_i = n·P_e", "O_i - E_i", "(O_i - E_i)^2 / E_i"],
        [
            [str(f), str(o), "1/6 = 0.166667", f"{e:.4f}", f"{o - e:+.4f}", f"{c:.4f}"]
            for f, o, e, c in zip(faces, observed, expected, contribs)
        ],
        ["Σ", str(sum(observed)), f"{6 * p_e:.6f}", f"{sum(expected):.4f}",
         f"{sum(o - e for o, e in zip(observed, expected)):+.4f}", f"{chi2_calc:.4f}"],
    )
    ok = n * p_e >= MIN_EXPECTED
    print(
        f"    Validez: toda E_i = n/6 = {n * p_e:.4f} "
        f"{'≥' if ok else '<'} {MIN_EXPECTED:g}  "
        f"{'✔ la aproximación χ² es válida' if ok else '✖ la aproximación χ² NO es confiable'}"
    )

    _step("Paso 2 · Grados de libertad")
    print(f"    r = {r} categorías (caras),  k = {k} parámetros estimados (H0 especifica P_e por completo)")
    print(f"    gl = r − k − 1 = {r} − {k} − 1 = {df}")

    _step("Paso 3 · Estadístico de prueba")
    print("    χ²_calc = Σ (O_i − E_i)² / E_i")
    print(f"            = {' + '.join(f'{c:.4f}' for c in contribs)}")
    print(f"            = {chi2_calc:.4f}")

    conclusion = "Conclusión: " + _lower_first(_conclusion_es_t1(
        {"rejected": chi2_calc > chi2_crit},
    ))
    rejected, decision = _print_decision(
        chi2_calc, chi2_crit, p_val, df, conclusion, first_step=4,
    )

    return dict(
        faces=faces, observed=observed, expected=expected, contribs=contribs,
        p_e=p_e, chi2_calc=chi2_calc, chi2_crit=chi2_crit, df=df, r=r, k=k,
        p_val=p_val, decision=decision, rejected=rejected,
        conclusion=conclusion, n=n,
    )


# ── Pooling algorithm (E_i >= 5) ─────────────────────────────────────

def _pool_bins(
    intervals: list[tuple[float, float]],
    observed: list[int],
    expected: list[float],
    min_e: float = MIN_EXPECTED,
) -> tuple[list[tuple[float, float]], list[int], list[float]]:
    """
    Merge adjacent categories (summing O_i and E_i) until every E_i >= min_e.

    1. Left tail merged inward, 2. right tail merged inward,
    3. safety net: any interior bin still below min_e is merged with its
       smaller neighbour (can only happen with very small n).
    """
    ints = list(intervals)
    obs = list(observed)
    exp = list(expected)
    last = len(intervals) - 1

    def label(i: int) -> str:
        return _fmt_interval(ints[i][0], ints[i][1], ints[i][1] == intervals[last][1])

    def merge(i: int, j: int, reason: str) -> None:
        a, b = min(i, j), max(i, j)
        merged_e = exp[a] + exp[b]
        print(
            f"    [AGRUPACIÓN] {reason}: {label(a)} (E={exp[a]:.4f}) + "
            f"{label(b)} (E={exp[b]:.4f})  →  "
            f"{_fmt_interval(ints[a][0], ints[b][1], ints[b][1] == intervals[last][1])}"
            f" (E={merged_e:.4f})"
        )
        ints[a] = (ints[a][0], ints[b][1])
        obs[a] += obs[b]
        exp[a] = merged_e
        del ints[b], obs[b], exp[b]

    while len(exp) > 1 and exp[0] < min_e:
        merge(0, 1, "cola izquierda")
    while len(exp) > 1 and exp[-1] < min_e:
        merge(len(exp) - 1, len(exp) - 2, "cola derecha  ")
    while len(exp) > 1 and min(exp) < min_e:
        i = int(np.argmin(exp))
        if i == 0:
            j = 1
        elif i == len(exp) - 1:
            j = i - 1
        else:
            j = i - 1 if exp[i - 1] <= exp[i + 1] else i + 1
        merge(i, j, "interior      ")
    if len(exp) == len(expected):
        print(f"    No hace falta agrupar — toda E_i ≥ {min_e:g}.")
    return ints, obs, exp


# ── Test 2: Normal GoF on sum of 5 dice (CLT) ────────────────────────

def _test_normal_clt(sums: list[int], n: int) -> dict | None:
    """
    H0: the sum of 5 dice follows N(mu_hat, sigma_hat).

    Uses grouped-data estimation (k = 2) and mandatory pooling (E_i >= 5).
    """
    _banner("PRUEBA 2 — Bondad de Ajuste Normal vía TCL  ·  S = d1 + d2 + d3 + d4 + d5")
    print("\n  Hipótesis")
    print("    H0: S ~ Normal(μ, σ), con μ y σ estimados a partir de los datos agrupados")
    print("    H1: S no sigue una distribución Normal")
    print(
        f"    Nivel de significación α = {ALPHA}  →  valor crítico con "
        f"probabilidad acumulada {1 - ALPHA:.2f}"
    )
    print(
        "    Referencia TCL (dados justos): E[S] = 5 × 3.5 = 17.5,  "
        f"DE[S] = √(5 × 35/12) = {math.sqrt(5 * 35 / 12):.4f}"
    )

    x_min = min(sums)
    x_max = max(sums)
    R = x_max - x_min
    if R == 0:
        print("\n    [ERROR] Todas las sumas son iguales — no se pueden construir intervalos.")
        return None

    # ── Step 1: Interval construction ─────────────────────────────
    # Amplitude rule: h = ceil(R / r_init), always rounded UP, so that
    # r_init * h >= R and the sample extremes are fully covered.
    r_init = 6
    h = math.ceil(R / r_init)

    # Semi-open intervals [L_inf, L_sup) from X_min; the last one is closed
    # on the right so X_max is never left out.
    intervals: list[tuple[float, float]] = []
    left = float(x_min)
    while True:
        right = left + h
        intervals.append((left, right))
        if right >= x_max:
            break
        left = right
    r_bins = len(intervals)

    observed: list[int] = []
    for idx, (lo, hi) in enumerate(intervals):
        if idx == r_bins - 1:
            cnt = sum(1 for s in sums if lo <= s <= hi)
        else:
            cnt = sum(1 for s in sums if lo <= s < hi)
        observed.append(cnt)
    labels = [
        _fmt_interval(lo, hi, i == r_bins - 1) for i, (lo, hi) in enumerate(intervals)
    ]

    _step("Paso 1 · Construcción de clases")
    print(f"    X_min = {x_min},  X_max = {x_max},  R = X_max − X_min = {R}")
    print(
        f"    h = ⌈R / r_init⌉ = ⌈{R} / {r_init}⌉ = ⌈{R / r_init:.4f}⌉ = {h}"
        "   (siempre redondeado HACIA ARRIBA para cubrir ambos extremos)"
    )
    print(
        f"    {r_bins} clases de amplitud h = {h} desde X_min = {x_min} hasta "
        f"{intervals[-1][1]:g}; la última clase es cerrada, así X_max queda incluido."
    )

    # ── Step 2: Parameter estimation from grouped data (k = 2) ────
    marks = [(lo + hi) / 2.0 for lo, hi in intervals]
    sum_fx = sum(f * x for f, x in zip(observed, marks))
    mu_hat = sum_fx / n
    sq_dev = [f * (x - mu_hat) ** 2 for f, x in zip(observed, marks)]
    sum_sq = sum(sq_dev)
    sigma_hat = math.sqrt(sum_sq / (n - 1))

    _step("Paso 2 · Estimación de parámetros con datos agrupados (k = 2)")
    _print_table(
        ["Clase", "x_i = (L_inf+L_sup)/2", "f_i = O_i", "f_i·x_i", "f_i·(x_i - mu_hat)^2"],
        [
            [labels[i], f"{marks[i]:.2f}", str(observed[i]),
             f"{observed[i] * marks[i]:.2f}", f"{sq_dev[i]:.4f}"]
            for i in range(r_bins)
        ],
        ["Σ", "", str(sum(observed)), f"{sum_fx:.2f}", f"{sum_sq:.4f}"],
    )
    print(f"    μ̂ = Σ f_i·x_i / n = {sum_fx:.2f} / {n} = {mu_hat:.4f}")
    print(
        f"    σ̂ = √( Σ f_i·(x_i − μ̂)² / (n − 1) ) = √( {sum_sq:.4f} / {n - 1} ) "
        f"= {sigma_hat:.4f}"
    )
    if sigma_hat == 0:
        print("    [ERROR] σ̂ = 0 — distribución degenerada.")
        return None

    # ── Step 3: Expected probabilities (NO infinite tails) ────────
    z_inf = [(lo - mu_hat) / sigma_hat for lo, _ in intervals]
    z_sup = [(hi - mu_hat) / sigma_hat for _, hi in intervals]
    phi_inf = [float(stats.norm.cdf(z)) for z in z_inf]
    phi_sup = [float(stats.norm.cdf(z)) for z in z_sup]
    pe_list = [s - i for s, i in zip(phi_sup, phi_inf)]
    expected = [n * pe for pe in pe_list]

    _step(
        f"Paso 3 · Frecuencias esperadas bajo N(μ̂ = {mu_hat:.4f}, σ̂ = {sigma_hat:.4f})"
        " — límites de clase finitos, sin colas ±∞"
    )
    print("    Z_inf = (L_inf − μ̂)/σ̂,  Z_sup = (L_sup − μ̂)/σ̂,  "
          "P_e = Φ(Z_sup) − Φ(Z_inf),  E_i = n·P_e")
    print("    Tabla inicial (sin agrupar):")
    _print_table(
        ["[L_inf, L_sup)", "x_i", "O_i", "Z_inf", "Z_sup",
         "Phi(Z_sup)", "Phi(Z_inf)", "P_e", "E_i"],
        [
            [labels[i], f"{marks[i]:.2f}", str(observed[i]),
             f"{z_inf[i]:+.4f}", f"{z_sup[i]:+.4f}",
             f"{phi_sup[i]:.6f}", f"{phi_inf[i]:.6f}",
             f"{pe_list[i]:.6f}", f"{expected[i]:.4f}"]
            for i in range(r_bins)
        ],
        ["Σ", "", str(sum(observed)), "", "", "", "",
         f"{sum(pe_list):.6f}", f"{sum(expected):.4f}"],
    )
    print(
        f"    ΣP_e = {sum(pe_list):.6f} < 1 y ΣE_i = {sum(expected):.4f} < n = {n}: "
        "la probabilidad fuera de [L_1, L_r] no se asigna (límites finitos)."
    )
    flagged = [labels[i] for i in range(r_bins) if expected[i] < MIN_EXPECTED]
    if flagged:
        print(f"    Clases con E_i < {MIN_EXPECTED:g}: {', '.join(flagged)}")

    # ── Step 4: Mandatory pooling (E_i < 5) ──────────────────────
    _step(f"Paso 4 · Agrupación — se fusionan clases adyacentes mientras alguna E_i < {MIN_EXPECTED:g}")
    int_p, obs_p, exp_p = _pool_bins(intervals, observed, expected)
    r_final = len(obs_p)
    contribs = [(o - e) ** 2 / e for o, e in zip(obs_p, exp_p)]
    chi2_calc = sum(contribs)
    pooled_labels = [
        _fmt_interval(lo, hi, i == r_final - 1) for i, (lo, hi) in enumerate(int_p)
    ]
    print("    Tabla final (agrupada):")
    _print_table(
        ["Categoría", "O_i", "E_i", "O_i - E_i", "(O_i - E_i)^2 / E_i"],
        [
            [pooled_labels[i], str(obs_p[i]), f"{exp_p[i]:.4f}",
             f"{obs_p[i] - exp_p[i]:+.4f}", f"{contribs[i]:.4f}"]
            for i in range(r_final)
        ],
        ["Σ", str(sum(obs_p)), f"{sum(exp_p):.4f}",
         f"{sum(o - e for o, e in zip(obs_p, exp_p)):+.4f}", f"{chi2_calc:.4f}"],
    )

    # ── Step 5: Degrees of freedom ────────────────────────────────
    k = 2  # mu_hat and sigma_hat
    df = r_final - k - 1
    _step("Paso 5 · Grados de libertad")
    print(f"    r = {r_final} categorías tras agrupar,  k = {k} parámetros estimados (μ̂, σ̂)")
    print(f"    gl = r − k − 1 = {r_final} − 2 − 1 = {df}")
    if df < 1:
        print(
            f"    [ERROR] gl = {df} < 1 — se necesitan r ≥ 4 categorías tras agrupar."
        )
        print("    La Prueba 2 no es calculable — registrá más tiradas y volvé a ejecutar.")
        return None

    chi2_crit = float(stats.chi2.ppf(1 - ALPHA, df))
    p_val = float(stats.chi2.sf(chi2_calc, df))

    _step("Paso 6 · Estadístico de prueba")
    print("    χ²_calc = Σ (O_i − E_i)² / E_i   (sobre las categorías agrupadas)")
    print(f"            = {' + '.join(f'{c:.4f}' for c in contribs)}")
    print(f"            = {chi2_calc:.4f}")

    conclusion = "Conclusión: " + _lower_first(_conclusion_es_t2({
        "rejected": chi2_calc > chi2_crit,
        "mu_hat": mu_hat,
        "sigma_hat": sigma_hat,
    }))
    rejected, decision = _print_decision(
        chi2_calc, chi2_crit, p_val, df, conclusion, first_step=7,
    )

    return dict(
        intervals=intervals, observed=observed, expected=expected, marks=marks,
        z_inf=z_inf, z_sup=z_sup, phi_inf=phi_inf, phi_sup=phi_sup, pe=pe_list,
        intervals_pooled=int_p, obs_pooled=obs_p, exp_pooled=exp_p,
        contribs=contribs, chi2_calc=chi2_calc, chi2_crit=chi2_crit, df=df,
        k=k, p_val=p_val, decision=decision, rejected=rejected,
        conclusion=conclusion, mu_hat=mu_hat, sigma_hat=sigma_hat, h=h,
        r_final=r_final, n=n, x_min=x_min, x_max=x_max,
    )


# ── Orchestrator ──────────────────────────────────────────────────────

def run_analysis(conn: sqlite3.Connection) -> None:
    """Query the database and run both chi-square tests + dashboard."""
    rows = fetch_all_rolls(conn)
    n = len(rows)
    if n < MIN_ROLLS:
        print(
            f"  [ERROR] n = {n}. Se necesitan al menos {MIN_ROLLS} tiradas para que "
            f"E_i = n/6 >= 5 en la Prueba 1. Presioná [S] para sembrar."
        )
        return

    d1_values = [r[0] for r in rows]
    sums = [r[5] for r in rows]

    t1 = _test_uniform(d1_values, n)
    t2 = _test_normal_clt(sums, n)
    if t2 is None:
        print("\n  [AVISO] La Prueba 2 no pudo completarse — el panel muestra solo la Prueba 1.")

    print()
    print("═" * _SEP_W)
    print("  RESUMEN")
    print("═" * _SEP_W)
    print(
        f"    Prueba 1 (Uniforme, d1): χ²_calc = {t1['chi2_calc']:.4f}  vs  "
        f"χ²_crít = {t1['chi2_crit']:.4f}  (gl = {t1['df']}),  "
        f"p = {t1['p_val']:.4f}  →  {t1['decision']}"
    )
    if t2 is not None:
        print(
            f"    Prueba 2 (Normal, suma): χ²_calc = {t2['chi2_calc']:.4f}  vs  "
            f"χ²_crít = {t2['chi2_crit']:.4f}  (gl = {t2['df']}),  "
            f"p = {t2['p_val']:.4f}  →  {t2['decision']}"
        )
    else:
        print("    Prueba 2 (Normal, suma): no calculable (gl < 1 tras agrupar)")
    print()

    _generate_dashboard(t1, t2, n)


# ═══════════════════════════════════════════════════════════════════════
#  PUBLICATION-READY DASHBOARD  (dark theme, 2 subplots + summary panels)
# ═══════════════════════════════════════════════════════════════════════

# Dark-mode chart palette. Observed / Expected / curve are fixed categorical
# slots (blue, orange, aqua); decision colours are status colours and are
# always paired with an icon + label, never colour alone.
_C = {
    "surface": "#1a1a19",
    "panel": "#232321",
    "text": "#f0efec",
    "text2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "border": "#3a3a37",
    "observed": "#3987e5",
    "expected": "#d95926",
    "curve": "#199e70",
    "good": "#0ca30c",
    "critical": "#d03b3b",
}


def _fmt_p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


# Thin surface-coloured halo keeps value labels legible over the curve
_HALO = [patheffects.withStroke(linewidth=3, foreground=_C["surface"])]


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(_C["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_C["border"])
    ax.tick_params(colors=_C["text2"], labelsize=10.5, length=0, pad=6)
    ax.yaxis.grid(True, color=_C["grid"], linewidth=0.9)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _summary_panel(
    ax: plt.Axes,
    heading: str,
    stats_rows: list[tuple[str, str]],
    result: dict | None,
    footnote: str,
    conclusion: str,
) -> None:
    """Executive-summary box under each chart: key numbers, decision badge,
    plain-language conclusion."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0,rounding_size=0.025",
        transform=ax.transAxes, facecolor=_C["panel"], edgecolor=_C["border"],
        linewidth=1.0, mutation_aspect=0.35, clip_on=False,
    ))
    ax.text(0.025, 0.88, heading, fontsize=10.5, color=_C["muted"],
            fontweight="bold", va="center")

    # Key numbers: two columns of label → value
    per_col = math.ceil(len(stats_rows) / 2)
    for idx, (lbl, val) in enumerate(stats_rows):
        col, row = divmod(idx, per_col)
        x0 = 0.025 + col * 0.255
        y = 0.72 - row * 0.15
        ax.text(x0, y, lbl, fontsize=12, color=_C["text2"], va="center")
        ax.text(x0 + 0.225, y, val, fontsize=12, color=_C["text"],
                va="center", ha="right", family="monospace")

    # Divider between the numbers and the verdict
    ax.plot([0.53, 0.53], [0.06, 0.82], color=_C["border"], linewidth=1.0)
    xr = 0.555

    if result is None:
        ax.text(xr, 0.80, "✖  NO CALCULABLE", fontsize=12.5, color="white",
                fontweight="bold", va="center",
                bbox=dict(boxstyle="round,pad=0.45", facecolor=_C["critical"],
                          edgecolor="none"))
        body = conclusion
    else:
        rejected = result["rejected"]
        badge = "✖  SE RECHAZA H₀" if rejected else "✔  NO SE RECHAZA H₀"
        ax.text(xr, 0.80, badge, fontsize=12.5, color="white",
                fontweight="bold", va="center",
                bbox=dict(boxstyle="round,pad=0.45",
                          facecolor=_C["critical"] if rejected else _C["good"],
                          edgecolor="none"))
        cmp_stat = ">" if rejected else "≤"
        cmp_p = "<" if result["p_val"] < ALPHA else "≥"
        ax.text(
            xr, 0.61,
            f"$\\chi^2_{{calc}}$ = {result['chi2_calc']:.4f}  {cmp_stat}  "
            f"$\\chi^2_{{crít}}$ = {result['chi2_crit']:.4f}",
            fontsize=11.5, color=_C["text2"], va="center",
        )
        ax.text(
            xr, 0.48,
            f"$p$ {'' if result['p_val'] < 1e-4 else '= '}"
            f"{_fmt_p(result['p_val'])}  {cmp_p}  $\\alpha$ = {ALPHA}",
            fontsize=11.5, color=_C["text2"], va="center",
        )
        body = conclusion
    ax.text(xr, 0.37, textwrap.fill(body, 48), fontsize=10.5,
            color=_C["text"], va="top", linespacing=1.4)
    ax.text(0.025, 0.05, textwrap.fill(footnote, 58), fontsize=9,
            color=_C["muted"], va="bottom", style="italic", linespacing=1.35)


def _conclusion_es_t1(t1: dict) -> str:
    if t1["rejected"]:
        return (f"Con α = {ALPHA}, las frecuencias de d1 difieren "
                f"significativamente de 1/6 por cara: hay evidencia de que el "
                f"dado (o la detección) está sesgado.")
    return (f"Con α = {ALPHA} no hay evidencia significativa de que d1 esté "
            f"sesgado; sus frecuencias son compatibles con un dado justo.")


def _conclusion_es_t2(t2: dict | None) -> str:
    if t2 is None:
        return ("Quedan menos de 4 categorías tras agrupar, por lo que "
                "gl = r − 3 < 1. Registrá más tiradas y volvé a ejecutar.")
    if t2["rejected"]:
        return (f"Con α = {ALPHA}, la distribución de la suma de 5 dados "
                f"difiere significativamente de la Normal ajustada.")
    return (f"Con α = {ALPHA}, la suma de 5 dados es compatible con una "
            f"Normal(μ̂ = {t2['mu_hat']:.2f}, σ̂ = {t2['sigma_hat']:.2f}), "
            f"como predice el Teorema Central del Límite.")


def _plot_test1(ax: plt.Axes, t1: dict) -> None:
    _style_axes(ax)
    faces = t1["faces"]
    obs = t1["observed"]
    exp = t1["expected"]
    x = np.arange(len(faces))
    bw = 0.38
    ax.bar(x - bw / 2, obs, bw, color=_C["observed"], edgecolor=_C["surface"],
           linewidth=1.5, label="Observadas $O_i$")
    ax.bar(x + bw / 2, exp, bw, color=_C["expected"], edgecolor=_C["surface"],
           linewidth=1.5, label=f"Esperadas $E_i = n/6$ = {exp[0]:.2f}")
    top = max(max(obs), exp[0])
    for xi, o in zip(x, obs):
        ax.text(xi - bw / 2, o + top * 0.015, str(o), ha="center", va="bottom",
                fontsize=10, color=_C["text2"])
    ax.set_ylim(0, top * 1.28)
    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in faces])
    ax.set_xlabel("Cara del dado $x_i$ (d1)", fontsize=11.5, color=_C["text2"], labelpad=8)
    ax.set_ylabel("Frecuencia (tiradas)", fontsize=11.5, color=_C["text2"], labelpad=8)
    ax.set_title("Prueba 1 · Uniforme discreta — ¿es justo el dado d1?", loc="left",
                 fontsize=14.5, color=_C["text"], fontweight="bold", pad=14)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=10.5,
              labelcolor=_C["text2"], handlelength=1.2, borderaxespad=0.2)


def _plot_test2(ax: plt.Axes, t2: dict | None) -> None:
    _style_axes(ax)
    ax.set_title("Prueba 2 · Ajuste normal (TCL) — ¿la suma de 5 dados es normal?",
                 loc="left", fontsize=14.5, color=_C["text"], fontweight="bold",
                 pad=14)
    if t2 is None:
        ax.text(0.5, 0.5, "No calculable: gl = r − 3 < 1 tras agrupar\n"
                "Registrá más tiradas y volvé a ejecutar", transform=ax.transAxes,
                ha="center", va="center", fontsize=13, color=_C["text2"])
        ax.set_xticks([])
        ax.set_yticks([])
        return

    int_p = t2["intervals_pooled"]
    obs_p = t2["obs_pooled"]
    exp_p = t2["exp_pooled"]
    h = t2["h"]
    n = t2["n"]
    mu, sig = t2["mu_hat"], t2["sigma_hat"]

    # Heights are "frequency per class of width h": an unmerged class shows
    # its raw count, a merged class (width m·h) is drawn at count/m. This
    # keeps every bar on the same scale as the curve n·h·φ(x) (one axis).
    heights: list[float] = []
    for i, ((lo, hi), o, e) in enumerate(zip(int_p, obs_p, exp_p)):
        w = hi - lo
        gap = w * 0.06
        half = (w - 2 * gap) / 2
        ho, he = o * h / w, e * h / w
        heights += [ho, he]
        ax.bar(lo + gap, ho, half, align="edge", color=_C["observed"],
               edgecolor=_C["surface"], linewidth=1.5,
               label="Observadas $O_i$ (agrupadas)" if i == 0 else None)
        ax.bar(lo + gap + half, he, half, align="edge", color=_C["expected"],
               edgecolor=_C["surface"], linewidth=1.5,
               label="Esperadas $E_i = n\\,P_e$ (agrupadas)" if i == 0 else None)
        ax.text(lo + gap + half / 2, ho, f"{o}", ha="center", va="bottom",
                fontsize=9.5, color=_C["text2"], path_effects=_HALO, zorder=5)
        ax.text(lo + gap + 1.5 * half, he, f"{e:.1f}", ha="center",
                va="bottom", fontsize=9.5, color=_C["text2"],
                path_effects=_HALO, zorder=5)

    x_curve = np.linspace(int_p[0][0] - h, int_p[-1][1] + h, 500)
    y_curve = stats.norm.pdf(x_curve, mu, sig) * n * h
    ax.plot(x_curve, y_curve, color=_C["curve"], linewidth=2.2,
            label="Curva normal $n\\,h\\,\\phi(x;\\,\\hat{\\mu},\\hat{\\sigma})$")
    ax.axvline(mu, color=_C["muted"], linewidth=1.0, linestyle=(0, (4, 3)))
    top = max(max(heights), float(y_curve.max()))
    ax.text(mu, top * 1.07, f" $\\hat{{\\mu}}$ = {mu:.2f}", color=_C["text2"],
            fontsize=10, va="bottom", ha="left")

    edges = sorted({e for iv in int_p for e in iv})
    ax.set_xticks(edges)
    ax.set_xticklabels([f"{e:g}" for e in edges])
    ax.set_xlim(x_curve[0], x_curve[-1])
    ax.set_ylim(0, top * 1.28)
    ax.set_xlabel("Suma de 5 dados  $S$  (marcas = límites de las categorías agrupadas)",
                  fontsize=11.5, color=_C["text2"], labelpad=8)
    ax.set_ylabel(f"Frecuencia por clase de amplitud h = {h:g}", fontsize=11.5,
                  color=_C["text2"], labelpad=8)
    ax.legend(loc="upper left", ncol=1, frameon=False, fontsize=10.5,
              labelcolor=_C["text2"], handlelength=1.2, borderaxespad=0.2)


def _generate_dashboard(t1: dict, t2: dict | None, n: int) -> None:
    rc = {
        "figure.facecolor": _C["surface"],
        "savefig.facecolor": _C["surface"],
        "text.color": _C["text"],
        "axes.labelcolor": _C["text2"],
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(19, 11))
        gs = fig.add_gridspec(
            2, 2, height_ratios=[3.0, 1.4], left=0.05, right=0.975,
            top=0.86, bottom=0.03, hspace=0.26, wspace=0.16,
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        s1 = fig.add_subplot(gs[1, 0])
        s2 = fig.add_subplot(gs[1, 1])

        fig.text(0.05, 0.955, "Prueba de Bondad de Ajuste χ² — Experimento con Dados",
                 fontsize=22, fontweight="bold", color=_C["text"])
        fig.text(
            0.05, 0.918,
            f"n = {n} tiradas (SQLite)  ·  α = {ALPHA}  ·  valores críticos con "
            f"probabilidad acumulada {1 - ALPHA:.2f}  ·  Prueba 1: k = 0  ·  "
            f"Prueba 2: k = 2 ($\\hat{{\\mu}}$, $\\hat{{\\sigma}}$ de datos "
            f"agrupados), límites de clase finitos, agrupación hasta $E_i \\geq 5$",
            fontsize=11.5, color=_C["text2"],
        )

        _plot_test1(ax1, t1)
        _plot_test2(ax2, t2)

        _summary_panel(
            s1, "RESUMEN · PRUEBA 1",
            [
                ("$n$", f"{t1['n']}"),
                ("$gl$  (r − k − 1)", f"{t1['df']}"),
                ("$P_e$", "1/6"),
                ("$\\chi^2_{calc}$", f"{t1['chi2_calc']:.4f}"),
                (f"$\\chi^2_{{crít}}$ ({1 - ALPHA:.2f})", f"{t1['chi2_crit']:.4f}"),
                ("valor $p$", _fmt_p(t1["p_val"])),
            ],
            t1,
            f"Todas las $E_i$ = n/6 = {n / 6:.2f} ≥ 5: la aproximación χ² es válida.",
            _conclusion_es_t1(t1),
        )
        if t2 is not None:
            t2_rows = [
                ("$\\hat{\\mu}$", f"{t2['mu_hat']:.4f}"),
                ("$\\hat{\\sigma}$", f"{t2['sigma_hat']:.4f}"),
                ("$r$ (tras agrupar)", f"{t2['r_final']}"),
                ("$k$ (estimados)", "2"),
                ("$gl$  (r − 3)", f"{t2['df']}"),
                ("$\\chi^2_{calc}$", f"{t2['chi2_calc']:.4f}"),
                (f"$\\chi^2_{{crít}}$ ({1 - ALPHA:.2f})", f"{t2['chi2_crit']:.4f}"),
                ("valor $p$", _fmt_p(t2["p_val"])),
            ]
            note = ("Categorías fusionadas a escala frecuencia × h / amplitud "
                    "(como la curva); etiquetas = $O_i$, $E_i$ reales.")
        else:
            t2_rows = [("$r$ (tras agrupar)", "< 4"), ("$gl$  (r − 3)", "< 1")]
            note = "Agrupar hasta $E_i \\geq 5$ dejó muy pocas categorías para la prueba."
        _summary_panel(s2, "RESUMEN · PRUEBA 2", t2_rows, t2, note,
                       _conclusion_es_t2(t2))

        saved = _save_figure(fig, DASHBOARD_PATH)
        plt.close(fig)
    if saved:
        print(f"  [PANEL] Guardado en -> {os.path.abspath(saved)}")


def _save_figure(fig: plt.Figure, target: str) -> str | None:
    """
    Write to a temp file, then swap it in. On Windows the target can be
    locked (open in Photos, OneDrive syncing): then keep the new image under
    a timestamped name instead of crashing at the end of the analysis.
    """
    root, ext = os.path.splitext(target)
    tmp = f"{root}.tmp{ext}"
    try:
        fig.savefig(tmp, dpi=200, facecolor=_C["surface"], format="png")
    except OSError as exc:
        print(f"  [ERROR] No se pudo escribir el panel ({exc}).")
        return None
    try:
        os.replace(tmp, target)
        return target
    except OSError:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = f"{root}_{stamp}{ext}"
        os.replace(tmp, alt)
        print(
            f"  [AVISO] '{target}' está bloqueado (¿abierto en otro programa, "
            f"p. ej. Fotos?). El panel nuevo se guardó como '{alt}'."
        )
        return alt


# ═══════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

class _SpanishHelpFormatter(argparse.HelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):  # type: ignore[override]
        super().add_usage(usage, actions, groups, prefix="uso: ")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bondad de ajuste chi-cuadrado con dados — pipeline de visión por computadora",
        formatter_class=_SpanishHelpFormatter,
        add_help=False,
    )
    parser._optionals.title = "opciones"
    parser.add_argument(
        "-h", "--help", action="help",
        help="mostrar esta ayuda y salir",
    )
    parser.add_argument(
        "--mode",
        choices=["opencv", "roboflow"],
        default="opencv",
        help="Motor de detección (predeterminado: opencv)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Índice de la cámara web (predeterminado: 0)",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_RF_MODEL,
        help=f"ID del modelo de Roboflow para --mode roboflow (predeterminado: "
             f"{DEFAULT_RF_MODEL}); la clave de API se lee de ROBOFLOW_API_KEY",
    )
    args = parser.parse_args()

    conn = init_db()
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    detector = DiceDetector(mode=args.mode, model_id=args.model_id)

    cap = cv2.VideoCapture(args.camera)
    use_cam = cap.isOpened()
    if use_cam:
        print(f"[INFO] Cámara (índice {args.camera}) abierta exitosamente. Presioná [T] en cualquier momento para alternar con modo sintético.")
    else:
        print(f"[INFO] No se pudo abrir la cámara {args.camera} — usando generador de dados sintéticos.")

    # Supervision annotators (>= 0.20.0 syntax)
    box_annotator = sv.RoundBoxAnnotator(
        thickness=2,
        color_lookup=sv.ColorLookup.CLASS,
    )
    label_annotator = sv.LabelAnnotator(
        text_position=sv.Position.TOP_LEFT,
        text_scale=0.6,
        color_lookup=sv.ColorLookup.CLASS,
    )

    rolls_logged = count_rolls(conn)
    # Recent live readings: a roll is only logged once it is stable
    history: deque[tuple[int, ...]] = deque(maxlen=STABLE_FRAMES)
    synth_frame, _, _ = generate_synthetic_frame()

    print(
        "\n"
        "┌────────────────────────────────────────────────────────────┐\n"
        "│  CONTROLES                                                 │\n"
        "│  ESPACIO  Registrar tirada de 5 dados y guardar fotograma  │\n"
        "│  S        Sembrar tiradas simuladas (completa a 100 o +100)│\n"
        "│  T        Alternar cámara en vivo / dados sintéticos       │\n"
        "│  A        Analizar — prueba Chi-Cuadrado y guardar panel   │\n"
        "│  Q        Salir                                            │\n"
        "└────────────────────────────────────────────────────────────┘\n"
    )

    while True:
        if use_cam:
            ret, frame = cap.read()
            if not ret:
                print("[AVISO] Transmisión de cámara interrumpida — cambiando a modo sintético.")
                use_cam = False
                continue
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            mode_tag = "CV"
        else:
            frame = synth_frame.copy()
            mode_tag = "Sim"

        detections = detector.detect(frame)

        if len(detections) > 0:
            current_values = [int(x) for x in detections.data.get("pips", [])[:5]]
        else:
            current_values = []

        current_sum = sum(current_values)
        if use_cam:
            history.append(tuple(current_values))

        annotated = build_annotated_frame(
            frame, detections, box_annotator, label_annotator,
            rolls_logged, current_sum, mode_tag,
        )

        cv2.imshow("Dice Chi-Square CV Pipeline (supervision)", annotated)
        key = cv2.waitKey(20) & 0xFF

        # ── SPACE: Log roll ───────────────────────────────────────
        if key == ord(" "):
            if not use_cam:
                synth_frame, _, _ = generate_synthetic_frame()
                detections = detector.detect(synth_frame)
                current_values = [int(x) for x in detections.data.get("pips", [])[:5]]
                annotated = build_annotated_frame(
                    synth_frame, detections, box_annotator, label_annotator,
                    rolls_logged + 1, sum(current_values), mode_tag,
                )
            else:
                # Majority vote over the last frames filters per-frame
                # flicker (glare, motion blur, hand still in view).
                reading, votes = (
                    Counter(history).most_common(1)[0] if history else ((), 0)
                )
                if votes < 0.7 * STABLE_FRAMES:
                    print(
                        f"  [OMITIDA] Lectura inestable ({votes}/{STABLE_FRAMES} "
                        f"frames coinciden) — tirada NO registrada. "
                        f"Esperá a que los dados estén quietos y presioná SPACE."
                    )
                    continue
                current_values = list(reading)

            # Never pad or guess values: that would inject fabricated
            # observations into the chi-square sample.
            if len(current_values) != 5 or not all(
                1 <= v <= 6 for v in current_values
            ):
                print(
                    f"  [OMITIDA] Lectura {current_values} — se necesitan 5 dados "
                    f"legibles (1–6). Tirada NO registrada; reacomodá y "
                    f"presioná SPACE."
                )
                continue

            insert_roll(conn, current_values[:5])
            rolls_logged = count_rolls(conn)
            ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            with sv.ImageSink(
                target_dir_path=CAPTURES_DIR, overwrite=False
            ) as sink:
                sink.save_image(
                    image=annotated,
                    image_name=f"tirada_{rolls_logged:04d}_{ts_str}.png",
                )
            print(
                f"  [TIRADA {rolls_logged:>4}]  "
                f"{current_values[:5]}  ->  Suma = {sum(current_values[:5])}"
            )

        # ── S: Seed rolls (Fast-Forward) ──────────────────────────
        elif key in (ord("s"), ord("S")):
            curr_n = count_rolls(conn)
            to_seed = (100 - curr_n) if 0 < curr_n < 100 else 100
            seed_rolls(conn, n=to_seed)
            rolls_logged = count_rolls(conn)

            with sv.ImageSink(
                target_dir_path=CAPTURES_DIR, overwrite=False
            ) as sink:
                for sample_idx in range(1, 4):
                    sf, _, _ = generate_synthetic_frame()
                    sd = detector.detect(sf)
                    s_vals = [int(x) for x in sd.data.get("pips", [])]
                    sf_ann = build_annotated_frame(
                        sf, sd, box_annotator, label_annotator,
                        rolls_logged, sum(s_vals), "Sim",
                    )
                    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    sink.save_image(
                        image=sf_ann,
                        image_name=f"muestra_semilla_{sample_idx}_{ts_str}.png",
                    )
            print(
                f"  [SEMILLA] +{to_seed} tiradas inyectadas en SQLite (Total n = {rolls_logged}) "
                f"y 3 muestras guardadas en '{CAPTURES_DIR}/'."
            )

        # ── T: Toggle Webcam / Synthetic Mode ─────────────────────
        elif key in (ord("t"), ord("T")):
            if cap.isOpened():
                use_cam = not use_cam
            else:
                use_cam = False
            history.clear()
            if not use_cam:
                synth_frame, _, _ = generate_synthetic_frame()
            print(f"  [MODO] Cambiado a {'Cámara en Vivo (CV)' if use_cam else 'Modo Sintético (Sim)'}")

        # ── A: Analyze ────────────────────────────────────────────
        elif key in (ord("a"), ord("A")):
            print("\n  [ANALIZAR] Cerrando video — ejecutando motor Chi-Cuadrado...\n")
            break

        # ── Q / ESC: Quit ─────────────────────────────────────────
        elif key in (ord("q"), ord("Q"), 27):
            print("  [SALIR] Saliendo limpiamente.")
            if cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            conn.close()
            return

    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

    run_analysis(conn)
    conn.close()


if __name__ == "__main__":
    main()