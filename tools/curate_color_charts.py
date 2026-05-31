"""
Interactive curator for one film/digital pair → LUT chart.

Loads a single pair, runs a fast initial pass to seed ~N candidate patches,
then opens a side-by-side viewer where you can:
  - Drag a patch to reposition it (Lab medians re-sample on release)
  - Right-click a patch to delete it
  - Click empty space to add a new patch
  - Press 'S' (or use the Save button) to render and write the chart TIFFs

The fast pass skips the hue and local-delta filters that the CLI uses —
those are what *you* are now doing by eye — so it returns quickly.

Usage:
  python tools/curate_color_charts.py \
      --film-dir path/to/film \
      --digital-dir path/to/digital \
      --out-dir path/to/charts \
      --pair 03 [--grid 8x6] [--patch-sample 12] [--initial-candidates 80]
"""
import argparse
import sys
from pathlib import Path

# Make tools/ importable and apply the project's numpy 2.0 shim.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / 'tools'))
import core  # noqa: F401

import cv2
import colour
import numpy as np

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (QImage, QPixmap, QPen, QBrush, QColor,
                           QPainter, QKeySequence, QShortcut)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsItem,
    QPushButton, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QStatusBar, QFileDialog,
)

import build_color_charts as bcc


# ---------------------------------------------------------------------------
# Display preview
# ---------------------------------------------------------------------------

def _acescct_ap1_to_srgb8(img: np.ndarray) -> np.ndarray:
    """Convert an ACEScct/AP1 float image to an 8-bit sRGB preview."""
    linear = bcc._acescct_decode(img)
    ap1 = colour.RGB_COLOURSPACES['ACEScg']
    srgb = colour.RGB_COLOURSPACES['sRGB']
    rgb = colour.RGB_to_RGB(linear, ap1, srgb)
    rgb = np.clip(rgb, 0.0, 1.0)
    encoded = colour.cctf_encoding(rgb, 'sRGB')
    return (np.clip(encoded, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_preview(img: np.ndarray, colorspace: str) -> np.ndarray:
    if colorspace in ('acescct_ap1', 'acescct_rec2020'):
        # acescct_rec2020 is rare here; treat its preview as AP1 — it's only
        # a visual aid for picking patches.
        return _acescct_ap1_to_srgb8(img)
    if colorspace == 'rec2020_g24':
        return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def np_rgb_to_qpixmap(arr: np.ndarray) -> QPixmap:
    h, w, _ = arr.shape
    # QImage needs contiguous data; cv2 ops may produce non-contiguous slices.
    buf = np.ascontiguousarray(arr)
    img = QImage(buf.data, w, h, w * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(img)


# ---------------------------------------------------------------------------
# Patch model
# ---------------------------------------------------------------------------

class Patch:
    """A sample location with independent film and digital centers.
    Normal drag on either rect moves both in lockstep; Shift+drag moves only
    the rect being dragged, letting you correct small alignment slop between
    the film and digital images."""

    def __init__(self, scene, image_w, gap, sample_size,
                 film_b, digital_b, y, x):
        self.scene = scene
        self.image_w = image_w
        self.gap = gap
        self.sample_size = sample_size
        self.film_b = film_b
        self.digital_b = digital_b
        self.fy, self.fx = int(y), int(x)
        self.dy, self.dx = int(y), int(x)
        self.film_color = np.zeros(3, dtype=np.float32)
        self.digital_color = np.zeros(3, dtype=np.float32)

        s = sample_size
        self.film_item = _PatchRect(self, 'film', QRectF(0, 0, s, s))
        self.dig_item = _PatchRect(self, 'digital', QRectF(0, 0, s, s))
        for it in (self.film_item, self.dig_item):
            it.setPen(QPen(QColor(80, 255, 120, 230), 2))
            it.setBrush(QBrush(QColor(80, 255, 120, 40)))
        scene.addItem(self.film_item)
        scene.addItem(self.dig_item)
        self._sync_position()
        self.resample()

    def partner(self, side: str) -> '_PatchRect':
        return self.dig_item if side == 'film' else self.film_item

    def _sync_position(self):
        s = self.sample_size
        # _syncing guards prevent the programmatic moves from triggering
        # itemChange's "mirror partner" logic.
        self.film_item._syncing = True
        self.dig_item._syncing = True
        self.film_item.setPos(self.fx - s / 2, self.fy - s / 2)
        self.dig_item.setPos(self.image_w + self.gap + self.dx - s / 2,
                             self.dy - s / 2)
        self.film_item._syncing = False
        self.dig_item._syncing = False

    def update_centers_from_rects(self):
        s = self.sample_size
        fp = self.film_item.pos()
        self.fx = int(round(fp.x() + s / 2))
        self.fy = int(round(fp.y() + s / 2))
        dp = self.dig_item.pos()
        self.dx = int(round(dp.x() - self.image_w - self.gap + s / 2))
        self.dy = int(round(dp.y() + s / 2))

    def resample(self):
        half = self.sample_size // 2
        fh, fw = self.film_b.shape[:2]
        if half <= self.fx < fw - half and half <= self.fy < fh - half:
            fm, _, _ = bcc.extract_patch(self.film_b, self.fy, self.fx,
                                         self.sample_size)
            self.film_color = fm
        else:
            self.film_color = np.zeros(3, dtype=np.float32)
        dh, dw = self.digital_b.shape[:2]
        if half <= self.dx < dw - half and half <= self.dy < dh - half:
            dm, _, _ = bcc.extract_patch(self.digital_b, self.dy, self.dx,
                                         self.sample_size)
            self.digital_color = dm
        else:
            self.digital_color = np.zeros(3, dtype=np.float32)

    def remove_from_scene(self):
        self.scene.removeItem(self.film_item)
        self.scene.removeItem(self.dig_item)


class _PatchRect(QGraphicsRectItem):
    """A draggable rect (film or digital side). During a normal drag, mirrors
    motion to its partner; during a Shift-drag, moves alone."""

    def __init__(self, owner: Patch, side: str, rect: QRectF):
        super().__init__(rect)
        self.owner = owner
        self.side = side
        self._shift_drag = False
        self._syncing = False
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

    def itemChange(self, change, value):
        if (change == QGraphicsItem.ItemPositionChange
                and not self._syncing and not self._shift_drag):
            delta = value - self.pos()
            partner = self.owner.partner(self.side)
            partner._syncing = True
            partner.setPos(partner.pos() + delta)
            partner._syncing = False
        return super().itemChange(change, value)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self.owner.scene.window.delete_patch(self.owner)
            ev.accept()
            return
        self._shift_drag = bool(ev.modifiers() & Qt.ShiftModifier)
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if ev.button() == Qt.LeftButton:
            self.owner.update_centers_from_rects()
            self.owner.resample()
            self._shift_drag = False


# ---------------------------------------------------------------------------
# Scene / view
# ---------------------------------------------------------------------------

class CurateScene(QGraphicsScene):
    def __init__(self, window):
        super().__init__()
        self.window = window  # for callbacks

    def mousePressEvent(self, ev):
        # If the click lands on (or very near) any patch rect, let the rect
        # handle it. Otherwise add a new patch at the click location.
        if self._patch_at(ev.scenePos()) is not None:
            super().mousePressEvent(ev)
            return
        if ev.button() == Qt.LeftButton:
            self.window.add_patch_at_scene(ev.scenePos())
            ev.accept()
            return
        super().mousePressEvent(ev)

    def _patch_at(self, scene_pos):
        # Slop region around the click — a few pixels of tolerance so users
        # don't accidentally add a stray patch when they meant to grab one.
        slop = max(4, self.window.args.patch_sample // 4)
        rect = QRectF(scene_pos.x() - slop, scene_pos.y() - slop,
                      slop * 2, slop * 2)
        for it in self.items(rect):
            if isinstance(it, _PatchRect):
                return it
        return None


class CurateView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHints(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom = 1.0
        self._panning = False
        self._pan_start = QPointF()

    def wheelEvent(self, ev):
        factor = 1.25 if ev.angleDelta().y() > 0 else 1 / 1.25
        self._zoom *= factor
        self.scale(factor, factor)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = ev.position()
            self.setCursor(Qt.ClosedHandCursor)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._panning:
            delta = ev.position() - self._pan_start
            self._pan_start = ev.position()
            self.horizontalScrollBar().setValue(
                int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(
                int(self.verticalScrollBar().value() - delta.y()))
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.unsetCursor()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class CurateWindow(QMainWindow):
    def __init__(self, args, pairs, start_idx):
        super().__init__()
        self.args = args
        self.pairs = pairs
        self.pair_idx = start_idx
        self.patches: list[Patch] = []

        # --- UI chrome --------------------------------------------------
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        bar = QHBoxLayout()
        bar.setContentsMargins(8, 6, 8, 6)
        prev_btn = QPushButton("◀ Prev")
        prev_btn.clicked.connect(self.prev_pair)
        bar.addWidget(prev_btn)
        next_btn = QPushButton("Next ▶")
        next_btn.clicked.connect(self.next_pair)
        bar.addWidget(next_btn)
        bar.addSpacing(20)
        self.pair_label = QLabel("")
        bar.addWidget(self.pair_label)
        bar.addSpacing(20)
        self.count_label = QLabel("Patches: 0")
        bar.addWidget(self.count_label)
        bar.addStretch(1)
        info = QLabel("Click empty to add · Drag patch to move both · "
                      "Shift+drag to move one side · Right-click to delete · "
                      "Middle-drag to pan · Wheel to zoom · S save · N/P step")
        info.setStyleSheet("color: #888;")
        bar.addWidget(info)
        bar.addSpacing(20)
        save_btn = QPushButton("Save chart")
        save_btn.clicked.connect(self.save_chart)
        bar.addWidget(save_btn)
        layout.addLayout(bar)

        # View placeholder; load_pair() creates a fresh scene each time.
        self.view: CurateView | None = None
        self._view_layout = layout
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        QShortcut(QKeySequence("S"), self, self.save_chart)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_chart)
        QShortcut(QKeySequence("F"), self, self.fit_view)
        QShortcut(QKeySequence("N"), self, self.next_pair)
        QShortcut(QKeySequence("P"), self, self.prev_pair)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_pair)
        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_pair)

        self.load_pair(self.pair_idx)

    # ----- pair loading ------------------------------------------------

    def load_pair(self, idx: int):
        idx = max(0, min(idx, len(self.pairs) - 1))
        self.pair_idx = idx
        stem, film_path, dig_path = self.pairs[idx]
        self.stem = stem
        self.setWindowTitle(
            f"Curate — pair {stem} ({idx + 1}/{len(self.pairs)})")
        self.pair_label.setText(
            f"Pair {stem}  ({idx + 1}/{len(self.pairs)})")

        film = bcc.load_film(film_path)
        digital = bcc.load_digital(
            dig_path, boost_ev=self.args.exposure_boost_ev)
        if film.shape[:2] != digital.shape[:2]:
            digital = cv2.resize(digital, (film.shape[1], film.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        if self.args.blur_sigma > 0:
            self.film_b = cv2.GaussianBlur(
                film, (0, 0), sigmaX=self.args.blur_sigma)
            self.digital_b = cv2.GaussianBlur(
                digital, (0, 0), sigmaX=self.args.blur_sigma)
        else:
            self.film_b, self.digital_b = film, digital
        self.film = film
        self.digital = digital
        self.image_h, self.image_w = film.shape[:2]
        self.gap = max(40, self.image_w // 50)

        film_prev = to_preview(film, self.args.film_colorspace)
        dig_prev = to_preview(digital, self.args.digital_colorspace)
        film_pix = np_rgb_to_qpixmap(film_prev)
        dig_pix = np_rgb_to_qpixmap(dig_prev)

        self.scene = CurateScene(self)
        self.scene.setBackgroundBrush(QBrush(QColor(28, 28, 30)))
        self.scene.addItem(QGraphicsPixmapItem(film_pix))
        dig_item = QGraphicsPixmapItem(dig_pix)
        dig_item.setPos(self.image_w + self.gap, 0)
        self.scene.addItem(dig_item)
        self.scene.setSceneRect(0, 0, 2 * self.image_w + self.gap,
                                self.image_h)

        if self.view is not None:
            self._view_layout.removeWidget(self.view)
            self.view.deleteLater()
        self.view = CurateView(self.scene)
        self._view_layout.addWidget(self.view)
        self.patches = []

        if self.args.initial_candidates > 0:
            self.seed_initial_patches()
        self.refresh_count()
        self.fit_view()

    def next_pair(self):
        if self.pair_idx + 1 < len(self.pairs):
            self.load_pair(self.pair_idx + 1)

    def prev_pair(self):
        if self.pair_idx > 0:
            self.load_pair(self.pair_idx - 1)

    # ----- patch lifecycle ---------------------------------------------

    def add_patch_at_scene(self, scene_pos: QPointF):
        sx, sy = scene_pos.x(), scene_pos.y()
        if 0 <= sx < self.image_w:
            x = sx
        elif self.image_w + self.gap <= sx < 2 * self.image_w + self.gap:
            x = sx - self.image_w - self.gap
        else:
            return  # clicked in the gap or outside
        y = sy
        if not (0 <= y < self.image_h):
            return
        self._add_patch(int(x), int(y))
        self.refresh_count()

    def _add_patch(self, x, y):
        p = Patch(self.scene, self.image_w, self.gap, self.args.patch_sample,
                  self.film_b, self.digital_b, y, x)
        self.patches.append(p)
        return p

    def delete_patch(self, patch: Patch):
        if patch not in self.patches:
            return
        patch.remove_from_scene()
        self.patches.remove(patch)
        self.refresh_count()

    def refresh_count(self):
        n = len(self.patches)
        target = self.args.grid[0] * self.args.grid[1]
        self.count_label.setText(
            f"Patches: {n}   (chart will use {min(n, target)} of {target})")

    # ----- initial seeding ---------------------------------------------

    def seed_initial_patches(self):
        """Fast pass — large stride, no hue/delta filters."""
        stride = self.args.stride or max(self.args.patch_sample, 8) * 2
        cands, _ = bcc.collect_candidates(
            self.film, self.digital,
            patch_detect=self.args.patch_detect,
            patch_sample=self.args.patch_sample,
            blur_sigma=self.args.blur_sigma,
            border_frac=0.0,
            flatness_pct=self.args.flatness_pct,
            stride=stride,
            sample_flatness_pct=self.args.sample_flatness_pct,
            max_quad_disagreement=self.args.max_quad_disagreement,
            max_hue_diff_deg=180.0,    # skip hue filter
            hue_chroma_min=0.0,
            max_delta_mad=0.0,         # skip local-delta filter
            border_px=self.args.border_px,
            film_colorspace=self.args.film_colorspace,
            digital_colorspace=self.args.digital_colorspace,
            flatness_metric=self.args.flatness_metric,
            detect_blur_sigma=self.args.detect_blur_sigma,
        )
        if not cands:
            self.statusBar().showMessage("No candidates from fast pass — "
                                         "click to add patches manually.")
            return

        # k-means in Lab to spread initial seeds across the candidate gamut.
        # We don't yet know the (y,x) for each candidate because
        # collect_candidates only returns colors. Re-run the same grid here
        # so we can pair colors with positions and pick spread seeds.
        positions = self._candidate_positions(stride)
        if len(positions) != len(cands):
            # Defensive: positions should equal cands by construction; if not,
            # fall back to plain truncation.
            positions = positions[:len(cands)]
            cands = cands[:len(positions)]

        film_colors = np.stack([c[0] for c in cands])
        lab = bcc.rgb_to_lab(film_colors, self.args.film_colorspace)
        n = min(self.args.initial_candidates, len(cands))
        sel = bcc.kmeans_select(lab, n)
        for i in sel:
            y, x = positions[int(i)]
            self._add_patch(int(x), int(y))

    def _candidate_positions(self, stride):
        """Re-walk the same grid collect_candidates uses, returning the
        (y, x) for each candidate it would have kept at the *detect* stage.
        Mirrors the relevant prefix of collect_candidates exactly."""
        film_d = self.film
        digital_d = self.digital
        if self.args.detect_blur_sigma > 0 and self.args.detect_blur_sigma != self.args.blur_sigma:
            film_d = cv2.GaussianBlur(self.film, (0, 0),
                                      sigmaX=self.args.detect_blur_sigma)
            digital_d = cv2.GaussianBlur(self.digital, (0, 0),
                                         sigmaX=self.args.detect_blur_sigma)
        elif self.args.blur_sigma > 0:
            film_d = cv2.GaussianBlur(self.film, (0, 0),
                                      sigmaX=self.args.blur_sigma)
            digital_d = cv2.GaussianBlur(self.digital, (0, 0),
                                         sigmaX=self.args.blur_sigma)
        std_f = bcc.flatness_map(film_d, self.args.patch_detect,
                                 self.args.flatness_metric,
                                 self.args.film_colorspace)
        std_d = bcc.flatness_map(digital_d, self.args.patch_detect,
                                 self.args.flatness_metric,
                                 self.args.digital_colorspace)
        h, w = self.image_h, self.image_w
        by = bx = self.args.border_px if self.args.border_px > 0 else 0
        half = self.args.patch_sample // 2
        y0, y1 = max(by, half), min(h - by, h - half)
        x0, x1 = max(bx, half), min(w - bx, w - half)
        grid_y = np.arange(y0, y1, stride)
        grid_x = np.arange(x0, x1, stride)
        valid_f = std_f[y0:y1, x0:x1]
        valid_d = std_d[y0:y1, x0:x1]
        thr_f = np.percentile(valid_f, self.args.flatness_pct)
        thr_d = np.percentile(valid_d, self.args.flatness_pct)

        # Same first pass, but also keep sample-flatness gate the way
        # collect_candidates does (it gates with `out` after that step).
        raw_positions = []
        raw_stats = []
        for y in grid_y:
            for x in grid_x:
                if std_f[y, x] > thr_f or std_d[y, x] > thr_d:
                    continue
                fm, fs, fq = bcc.extract_patch(
                    cv2.GaussianBlur(self.film, (0, 0),
                                     sigmaX=self.args.blur_sigma)
                    if self.args.blur_sigma > 0 else self.film,
                    int(y), int(x), self.args.patch_sample)
                dm, ds, dq = bcc.extract_patch(
                    cv2.GaussianBlur(self.digital, (0, 0),
                                     sigmaX=self.args.blur_sigma)
                    if self.args.blur_sigma > 0 else self.digital,
                    int(y), int(x), self.args.patch_sample)
                raw_positions.append((int(y), int(x)))
                raw_stats.append((fs, ds, max(fq, dq)))
        if not raw_positions:
            return []
        f_stds = np.array([r[0] for r in raw_stats])
        d_stds = np.array([r[1] for r in raw_stats])
        thr_fs = np.percentile(f_stds, self.args.sample_flatness_pct)
        thr_ds = np.percentile(d_stds, self.args.sample_flatness_pct)
        kept = [pos for pos, (fs, ds, qm) in zip(raw_positions, raw_stats)
                if fs <= thr_fs and ds <= thr_ds
                and qm <= self.args.max_quad_disagreement]
        return kept

    # ----- view helpers ------------------------------------------------

    def fit_view(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.view._zoom = 1.0

    # ----- save --------------------------------------------------------

    def save_chart(self):
        cols, rows = self.args.grid
        target = rows * cols
        if not self.patches:
            self.statusBar().showMessage("No patches to save.")
            return

        film_colors = np.stack([p.film_color for p in self.patches])
        dig_colors = np.stack([p.digital_color for p in self.patches])
        lab = bcc.rgb_to_lab(film_colors, self.args.film_colorspace)

        if len(self.patches) < target:
            sel = np.concatenate([
                np.arange(len(self.patches)),
                np.random.choice(len(self.patches),
                                 target - len(self.patches), replace=True),
            ])
        else:
            sel = bcc.kmeans_select(lab, target)

        film_sel = film_colors[sel]
        dig_sel = dig_colors[sel]
        lab_sel = bcc.rgb_to_lab(film_sel, self.args.film_colorspace)
        order = bcc.order_for_grid(lab_sel, rows, cols)

        film_chart = bcc.render_chart(film_sel[order], rows, cols,
                                      self.args.cell_px)
        dig_chart = bcc.render_chart(dig_sel[order], rows, cols,
                                     self.args.cell_px)

        out_dir = self.args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        film_out = out_dir / f"{self.stem}_film.tif"
        dig_out = out_dir / f"{self.stem}_digital.tif"
        bcc.save_film_chart(film_chart, film_out)
        bcc.save_digital_chart(dig_chart, dig_out)
        self.statusBar().showMessage(
            f"Saved {film_out.name} + {dig_out.name} "
            f"({target} swatches from {len(self.patches)} patches)")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--film-dir', type=Path, required=True)
    ap.add_argument('--digital-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--pair', type=str, default='',
                    help='chart key for the pair to start on (e.g. "03"). '
                         'If omitted, opens the first pair and lets you step '
                         'through the folder with N/P.')
    ap.add_argument('--grid', type=bcc.parse_grid, default=(8, 6))
    ap.add_argument('--patch-detect', type=int, default=16)
    ap.add_argument('--patch-sample', type=int, default=24)
    ap.add_argument('--blur-sigma', type=float, default=1.5)
    ap.add_argument('--detect-blur-sigma', type=float, default=5.0)
    ap.add_argument('--flatness-pct', type=float, default=50.0)
    ap.add_argument('--sample-flatness-pct', type=float, default=70.0)
    ap.add_argument('--max-quad-disagreement', type=float, default=0.04)
    ap.add_argument('--cell-px', type=int, default=80)
    ap.add_argument('--border-px', type=int, default=100)
    cs_choices = ['srgb', 'rec2020_g24', 'acescct_rec2020', 'acescct_ap1']
    ap.add_argument('--film-colorspace', choices=cs_choices,
                    default='acescct_ap1')
    ap.add_argument('--digital-colorspace', choices=cs_choices,
                    default='acescct_ap1')
    ap.add_argument('--flatness-metric', choices=['rgb', 'chroma'],
                    default='chroma')
    ap.add_argument('--exposure-boost-ev', type=float, default=0.0)
    ap.add_argument('--stride', type=int, default=None,
                    help='spacing between candidate centers; large = faster '
                         'fewer seeds (default: 2 × patch_sample)')
    ap.add_argument('--initial-candidates', type=int, default=0,
                    help='if > 0, run the fast pass and seed this many '
                         'patches via kmeans-spread before opening the UI. '
                         'Default 0 — open with no patches and pick manually.')
    args = ap.parse_args()

    pairs = bcc.pair_files(args.film_dir, args.digital_dir)
    if not pairs:
        print("No matching pairs found.")
        sys.exit(1)

    start_idx = 0
    wanted = args.pair.strip()
    if wanted:
        for i, (stem, _, _) in enumerate(pairs):
            if stem == wanted:
                start_idx = i
                break
            try:
                if str(int(stem)) == wanted:
                    start_idx = i
                    break
            except ValueError:
                pass
        else:
            print(f"No pair matches --pair {args.pair!r}. "
                  f"Available: {', '.join(p[0] for p in pairs)}")
            sys.exit(1)

    app = QApplication(sys.argv)
    w = CurateWindow(args, pairs, start_idx)
    w.resize(1600, 900)
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
