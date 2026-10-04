
import hashlib
import html
import importlib.util
import math
import os
import sys
import tempfile
import time
import traceback

from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QAction, QBrush, QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen,
    QPixmap, QRadialGradient,
)
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QAbstractButton, QApplication, QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QInputDialog,
    QLabel, QMainWindow, QMenu, QMessageBox, QPushButton, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

import numpy as np
from scipy import signal

import dsp

# ---------- Palette néon (change les couleurs ici) ----------
C1, C2, C3, WARN = "#00e5ff", "#ff3df2", "#7c5cff", "#ffd166"   # cyan, magenta, violet, ambre
BG0, BG1 = "#07070d", "#0e0b1f"
TEXT, MUTED = "#eef0ff", "#8b8fb0"
OK, BUSY, ERR = "#34d399", "#ffd166", "#ff6b6b"
SPEC_STOPS = [(0.00, (5, 6, 18)), (0.22, (12, 30, 84)), (0.45, (0, 140, 190)),
              (0.65, (0, 229, 255)), (0.85, (255, 61, 242)), (1.00, (255, 238, 250))]
LUT = np.stack([np.interp(np.linspace(0, 1, 256), [s[0] for s in SPEC_STOPS], [s[1][c] for s in SPEC_STOPS])
                for c in range(3)], axis=1).astype(np.uint8)

QSS = f"""
* {{ color: {TEXT}; }}
QLabel {{ background: transparent; }}
QFrame#hero, QFrame#glass, QFrame#module {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(255,255,255,20), stop:1 rgba(255,255,255,6));
    border: 1px solid rgba(255,255,255,30); border-radius: 18px; }}
QLabel#cap {{ color: {MUTED}; font-size: 10px; }}
QLabel#val {{ font-size: 12px; font-weight: 600; }}
QLabel#hint {{ color: {MUTED}; font-size: 11px; }}
QSpinBox {{ background: rgba(255,255,255,14); border: 1px solid rgba(255,255,255,34); border-radius: 12px;
    padding: 5px 10px; min-width: 62px; font-weight: 600; }}
QSpinBox:hover {{ border-color: {C1}; }}
QSpinBox::up-button, QSpinBox::down-button {{ width: 0; }}
QToolButton, QPushButton#ghost {{ background: rgba(255,255,255,14); border: 1px solid rgba(255,255,255,34);
    border-radius: 14px; padding: 8px 16px; font-weight: 600; }}
QToolButton:hover, QPushButton#ghost:hover {{ border-color: {C1}; background: rgba(0,229,255,28); }}
QToolButton::menu-indicator {{ image: none; }}
QPushButton[seg="true"] {{ background: rgba(255,255,255,10); border: 1px solid rgba(255,255,255,30);
    padding: 6px 14px; font-size: 11px; font-weight: 700; border-radius: 0; }}
QPushButton[seg="true"][pos="first"] {{ border-top-left-radius: 12px; border-bottom-left-radius: 12px; }}
QPushButton[seg="true"][pos="last"] {{ border-top-right-radius: 12px; border-bottom-right-radius: 12px; }}
QPushButton[seg="true"]:hover {{ border-color: {C1}; }}
QPushButton[seg="true"]:checked {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00b8d4, stop:1 #c026d3);
    border-color: transparent; color: white; }}
QPushButton[seg="true"]:disabled {{ color: #5b5e7d; }}
QMenu {{ background: #14122a; border: 1px solid rgba(255,255,255,40); border-radius: 10px; padding: 6px; }}
QMenu::item {{ padding: 7px 20px; border-radius: 6px; }}
QMenu::item:selected {{ background: rgba(0,229,255,45); }}
QMenu::separator {{ height: 1px; background: rgba(255,255,255,30); margin: 5px 8px; }}
QMessageBox {{ background: #14122a; }}
"""

NOTES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
DEFAULT_DUR = 15
SPEC_ROWS = 192
BANDS = 44


def qc(hexcol, a=255):
    c = QColor(hexcol)
    c.setAlpha(a)
    return c


def fmt_hz(v):
    return f"{v / 1000:.1f} kHz" if v >= 1000 else f"{v:.0f} Hz"


def fmt_t(s):
    m = int(s // 60)
    return f"{m}:{s - 60 * m:04.1f}"


def note_name(f):
    midi = 69 + 12 * np.log2(f / 440.0)
    n = int(round(midi))
    return f"{NOTES[n % 12]}{n // 12 - 1} {int(round((midi - n) * 100)):+d}¢"


def spaced(label, size=10, spacing=1.6, bold=True):
    f = label.font()
    f.setPointSizeF(size)
    f.setBold(bold)
    f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    label.setFont(f)
    return label


# ======================= Traitement (threads) =======================
_DEMUCS = {}
_STAGES = {"sig": None, "items": []}
HAS_DEMUCS = importlib.util.find_spec("demucs") is not None


def _sig(y):
    return hashlib.md5(np.ascontiguousarray(y[:, ::1009]).tobytes()).hexdigest() + str(y.shape)


def _apply(y, sr, s):
    t = s["type"]
    if t == "gain":
        return y * 10 ** (s["db"] / 20)
    if t == "filter":
        if s["kind"] == "bandpass":
            if s["f1"] < s["f2"]:
                return dsp.butter_filter(y, sr, "bandpass", [s["f1"], s["f2"]], s["order"])
            return y
        return dsp.butter_filter(y, sr, s["kind"], s["f1"], s["order"])
    if t == "pitch":
        return dsp.pitch_shift(y, sr, s["st"])
    if t == "reverb":
        return dsp.reverb(y, sr, s["decay"], s["wet"], s["pre"])
    if t == "sep":
        mode = s["mode"]
        if "Demucs" in mode:
            if not HAS_DEMUCS:
                raise RuntimeError("Demucs n'est pas installé : pip install demucs")
            key = _sig(y)
            if key not in _DEMUCS:
                _DEMUCS[key] = dsp.separate_demucs(np.ascontiguousarray(y), sr)
            v, i = _DEMUCS[key]
        else:
            v, i = dsp.split_center(y, sr)
        return v if mode.startswith("Voix") else i
    return y


def process_chain(x, sr, specs, use_cache=True):
    """Effets actifs dans l'ordre -> (signal, infos). Un effet qui plante est ignoré (avertissement).
    Cache des étapes : seul l'effet modifié (et les suivants) est recalculé."""
    active = [s for s in specs if s["on"]]
    keys = [repr({k: v for k, v in s.items() if k != "on"}) for s in active]
    stages = []
    if use_cache:
        sig = _sig(x)
        if _STAGES["sig"] != sig:
            _STAGES["sig"], _STAGES["items"] = sig, []
        stages = _STAGES["items"]
    k = 0
    while k < len(active) and k < len(stages) and stages[k][0] == keys[k]:
        k += 1
    done = list(stages[:k])
    y = done[-1][1] if done else x
    times = []
    for i in range(k, len(active)):
        t0, warn = time.perf_counter(), None
        try:
            y = _apply(y, sr, active[i])
        except Exception as e:  # noqa: BLE001
            warn = f"{active[i]['type']} ignoré : {e}"
        times.append((active[i]["type"], time.perf_counter() - t0))
        done.append((keys[i], y, warn))
    if use_cache:
        _STAGES["items"] = done
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32), dict(times=times, warns=[d[2] for d in done if d[2]])


def compute_spec(b, sr):
    """Spectrogramme (temps, SPEC_ROWS) en dB relatifs, échelle de fréquence logarithmique 40 Hz-20 kHz."""
    mono = b.mean(axis=0)
    f, _, S = signal.spectrogram(mono, sr, window="hann", nperseg=4096, noverlap=3072, mode="magnitude")
    targets = np.logspace(np.log10(40), np.log10(20000), SPEC_ROWS)
    idx = np.clip(np.searchsorted(f, targets), 0, len(f) - 1)
    db = 20 * np.log10(S[idx] + 1e-10)
    db -= db.max()
    return db.T.astype(np.float32)


def chain_job(x, sr, specs):
    y, info = process_chain(x, sr, specs)
    return y, info, compute_spec(y, sr)


def excerpt_job(path, start, dur, total=None):
    if total is None:
        total, _ = dsp.audio_info(path)
    start = int(min(start, max(int(total) - 1, 0)))
    dur = float(max(1.0, min(dur, total - start)))
    x, sr = dsp.load_excerpt(path, start, dur)
    return dict(path=path, total=total, start=start, x=x, spec=compute_spec(x, sr))


def open_job(path, start, dur):
    total, _ = dsp.audio_info(path)
    return excerpt_job(path, start, dur, total)


def filter_response(kind, order, f1, f2, sr=dsp.SR):
    if kind == "bandpass":
        if not f1 < f2:
            return None
        wn = [f1 / (sr / 2), f2 / (sr / 2)]
    else:
        wn = f1 / (sr / 2)
    sos = signal.butter(order, wn, btype=kind, output="sos")
    f = np.logspace(np.log10(20), np.log10(20000), 240)
    _, h = signal.sosfreqz(sos, worN=f, fs=sr)
    return f, np.clip(40 * np.log10(np.abs(h) + 1e-9), -36, 6)  # sosfiltfilt => |H|^2


def fine_env(b, F=1536):
    mono = np.abs(b).max(axis=0)
    F = min(F, len(mono))
    k = len(mono) // F
    return mono[: F * k].reshape(F, k).max(axis=1)


class Worker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args

    def run(self):
        try:
            self.done.emit(self.fn(*self.args))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")


# ======================= Widgets dessinés à la main =======================
class Backdrop(QWidget):
    """Fond sombre avec lueurs néon."""

    def paintEvent(self, _):
        p = QPainter(self)
        w, h = self.width(), self.height()
        g = QLinearGradient(0, 0, 0, h)
        g.setColorAt(0, QColor(BG1))
        g.setColorAt(1, QColor(BG0))
        p.fillRect(self.rect(), g)
        for cx, cy, rad, col in ((w * 0.92, -h * 0.05, w * 0.55, C2), (w * 0.05, h * 1.0, w * 0.6, C1),
                                 (w * 0.55, h * 0.45, w * 0.4, C3)):
            rg = QRadialGradient(cx, cy, rad)
            rg.setColorAt(0, qc(col, 52))
            rg.setColorAt(1, qc(col, 0))
            p.fillRect(self.rect(), QBrush(rg))


class Knob(QWidget):
    """Potard rotatif : glisser = régler, Maj = précis, molette, double-clic = défaut."""
    valueChanged = pyqtSignal(float)

    def __init__(self, caption, lo, hi, val, step=0.0, fmt=None, log=False, bipolar=False, size=84):
        super().__init__()
        self.caption, self.lo, self.hi, self.step, self.log, self.bipolar = caption, lo, hi, step, log, bipolar
        self.default, self.size_, self.dim = val, size, False
        self.fmt = fmt or (lambda v: f"{v:g}")
        self.setFixedSize(size + 12, size + 40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Glisser = régler · Maj = précis · molette · double-clic = réinitialiser")
        self.frac = self._to_frac(val)
        self._last = self.value()
        self._drag = None

    def _to_frac(self, v):
        f = np.log(v / self.lo) / np.log(self.hi / self.lo) if self.log else (v - self.lo) / (self.hi - self.lo)
        return float(np.clip(f, 0, 1))

    def value(self):
        f = float(np.clip(self.frac, 0, 1))
        v = self.lo * (self.hi / self.lo) ** f if self.log else self.lo + f * (self.hi - self.lo)
        if self.step:
            v = round(v / self.step) * self.step
        return float(np.clip(round(v, 6), self.lo, self.hi))

    def set_value(self, v):
        self._set_frac(self._to_frac(v))

    def _set_frac(self, f):
        self.frac = float(np.clip(f, 0, 1))
        v = self.value()
        if v != self._last:
            self._last = v
            self.valueChanged.emit(v)
        self.update()

    def mousePressEvent(self, e):
        self._drag = (e.position().y(), self.frac)

    def mouseMoveEvent(self, e):
        if self._drag:
            fine = 700 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 220
            self._set_frac(self._drag[1] + (self._drag[0] - e.position().y()) / fine)

    def mouseReleaseEvent(self, _):
        self._drag = None

    def mouseDoubleClickEvent(self, _):
        self.set_value(self.default)

    def wheelEvent(self, e):
        d = e.angleDelta().y() / 120
        step = (self.step / (self.hi - self.lo)) if (self.step and not self.log) else 0.03
        self._set_frac(self.frac + d * step)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.dim:
            p.setOpacity(0.36)
        s, ox, oy, m = self.size_, 6, 2, 9
        cx, cy = ox + s / 2, oy + s / 2
        arc = QRectF(ox + m, oy + m, s - 2 * m, s - 2 * m)
        cap = Qt.PenCapStyle.RoundCap
        p.setPen(QPen(qc("#ffffff", 26), 6, Qt.PenStyle.SolidLine, cap))
        p.drawArc(arc, 225 * 16, -270 * 16)
        f = self._to_frac(self.value())
        start, span = (90, -(f - 0.5) * 270) if self.bipolar else (225, -270 * f)
        if abs(span) > 0.6:
            for alpha, width in ((70, 12), (255, 5)):
                g = QLinearGradient(arc.left(), 0, arc.right(), 0)
                g.setColorAt(0, qc(C1, alpha))
                g.setColorAt(1, qc(C2, alpha))
                p.setPen(QPen(QBrush(g), width, Qt.PenStyle.SolidLine, cap))
                p.drawArc(arc, int(start * 16), int(span * 16))
        r = s / 2 - m - 10
        rg = QRadialGradient(cx - r * 0.35, cy - r * 0.45, r * 1.5)
        rg.setColorAt(0, QColor("#3a3c68"))
        rg.setColorAt(1, QColor("#0d0e22"))
        p.setPen(QPen(qc("#ffffff", 40), 1))
        p.setBrush(QBrush(rg))
        p.drawEllipse(QPointF(cx, cy), r, r)
        a = math.radians(225 - 270 * f)
        p.setPen(QPen(QColor("white"), 3, Qt.PenStyle.SolidLine, cap))
        p.drawLine(QPointF(cx + r * 0.38 * math.cos(a), cy - r * 0.38 * math.sin(a)),
                   QPointF(cx + r * 0.86 * math.cos(a), cy - r * 0.86 * math.sin(a)))
        p.setPen(qc(TEXT))
        font = p.font()
        font.setPointSizeF(10)
        font.setBold(True)
        p.setFont(font)
        p.drawText(QRectF(0, oy + s + 1, self.width(), 18), Qt.AlignmentFlag.AlignCenter, self.fmt(self.value()))
        p.setPen(qc(MUTED))
        font.setPointSizeF(7.5)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.4)
        p.setFont(font)
        p.drawText(QRectF(0, oy + s + 18, self.width(), 16), Qt.AlignmentFlag.AlignCenter, self.caption)


class Led(QAbstractButton):
    def __init__(self):
        super().__init__()
        self.setCheckable(True)
        self.setFixedSize(26, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Activer / désactiver ce module")

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(13, 13)
        if self.isChecked():
            rg = QRadialGradient(c, 13)
            rg.setColorAt(0, qc(C1, 120))
            rg.setColorAt(1, qc(C1, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(rg))
            p.drawEllipse(c, 13, 13)
            p.setBrush(qc(C1))
            p.drawEllipse(c, 5, 5)
            p.setBrush(qc("#ffffff", 200))
            p.drawEllipse(QPointF(11.5, 11.5), 1.8, 1.8)
        else:
            p.setPen(QPen(qc("#ffffff", 70), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, 5, 5)


class PlayButton(QAbstractButton):
    def __init__(self):
        super().__init__()
        self.setFixedSize(72, 72)
        self.playing = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(36, 36)
        rg = QRadialGradient(c, 36)
        rg.setColorAt(0.6, qc(C2, 90))
        rg.setColorAt(1, qc(C2, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(rg))
        p.drawEllipse(c, 36, 36)
        g = QLinearGradient(8, 8, 64, 64)
        g.setColorAt(0, qc(C1))
        g.setColorAt(1, qc(C2))
        p.setBrush(QBrush(g))
        p.drawEllipse(c, 27, 27)
        p.setBrush(QColor("#0b0a1a"))
        if self.playing:
            p.drawRoundedRect(QRectF(27, 25, 6.5, 22), 2, 2)
            p.drawRoundedRect(QRectF(38.5, 25, 6.5, 22), 2, 2)
        else:
            path = QPainterPath()
            path.moveTo(30, 24)
            path.lineTo(30, 48)
            path.lineTo(50, 36)
            path.closeSubpath()
            p.drawPath(path)


class Segmented(QWidget):
    changed = pyqtSignal(int)

    def __init__(self, labels):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.btns = []
        for i, text in enumerate(labels):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setProperty("seg", True)
            b.setProperty("pos", "first" if i == 0 else "last" if i == len(labels) - 1 else "mid")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            self.group.addButton(b, i)
            lay.addWidget(b)
            self.btns.append(b)
        self.btns[0].setChecked(True)
        self.group.idClicked.connect(self.changed)

    def index(self):
        return self.group.checkedId()

    def set_index(self, i):
        self.btns[i].setChecked(True)


class WaveBars(QWidget):
    """Forme d'onde en barres néon (barres jouées = dégradé, à venir = atténuées)."""
    seek = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(130)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cur = self.ghost = None
        self.progress = 0.0

    def set_data(self, cur, ghost):
        self.cur, self.ghost = cur, ghost
        self.update()

    def set_progress(self, f):
        self.progress = f
        self.update()

    @staticmethod
    def _reduce(fine, n):
        edges = np.linspace(0, len(fine), n + 1).astype(int)
        return np.maximum.reduceat(fine, edges[:-1])

    def _emit(self, e):
        self.seek.emit(float(np.clip(e.position().x() / max(self.width(), 1), 0, 1)))

    def mousePressEvent(self, e):
        self._emit(e)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._emit(e)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        mid = h / 2
        p.setPen(QPen(qc("#ffffff", 22), 1))
        p.drawLine(QPointF(0, mid), QPointF(w, mid))
        if self.cur is None or len(self.cur) < 8:
            return
        n = max(24, int(w // 6))
        bw = w / n
        cur = self._reduce(self.cur, n)
        ghost = self._reduce(self.ghost, n) if self.ghost is not None else None
        norm = max(float(cur.max()), float(ghost.max()) if ghost is not None else 0.0, 1e-6)
        g_on, g_off = QLinearGradient(0, 0, w, 0), QLinearGradient(0, 0, w, 0)
        for g, a in ((g_on, 255), (g_off, 70)):
            g.setColorAt(0, qc(C1, a))
            g.setColorAt(0.55, qc(C3, a))
            g.setColorAt(1, qc(C2, a))
        b_on, b_off, b_gh = QBrush(g_on), QBrush(g_off), QBrush(qc("#ffffff", 26))
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            x = i * bw + bw * 0.18
            if ghost is not None:
                gh = max(1.5, (ghost[i] / norm) ** 0.75 * h * 0.46)
                p.setBrush(b_gh)
                p.drawRoundedRect(QRectF(x - 0.5, mid - gh, bw * 0.64 + 1, 2 * gh), 2, 2)
            hh = max(1.5, (cur[i] / norm) ** 0.75 * h * 0.46)
            p.setBrush(b_on if (i + 0.5) / n <= self.progress else b_off)
            p.drawRoundedRect(QRectF(x, mid - hh, bw * 0.64, 2 * hh), 2, 2)
        x = self.progress * w
        p.setPen(QPen(qc("#ffffff", 230), 2))
        p.drawLine(QPointF(x, 2), QPointF(x, h - 2))


class SpecView(QWidget):
    """Spectrogramme néon, fréquence en échelle log (graves en bas)."""
    seek = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pix = None
        self.progress = 0.0

    def set_image(self, db):  # db : (temps, lignes)
        idx = np.clip((db + 78.0) / 78.0 * 255, 0, 255).astype(np.uint8)
        arr = np.ascontiguousarray(LUT[idx.T[::-1]])  # (lignes, temps, 3), aigus en haut
        h, w, _ = arr.shape
        self.pix = QPixmap.fromImage(QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888))
        self.update()

    def set_progress(self, f):
        self.progress = f
        self.update()

    def _emit(self, e):
        self.seek.emit(float(np.clip(e.position().x() / max(self.width(), 1), 0, 1)))

    def mousePressEvent(self, e):
        self._emit(e)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._emit(e)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        w, h = self.width(), self.height()
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0, 0, w, h), 12, 12)
        p.setClipPath(clip)
        p.fillRect(self.rect(), QColor("#05060f"))
        if self.pix is not None:
            p.drawPixmap(QRectF(0, 0, w, h), self.pix, QRectF(self.pix.rect()))
        lo, hi = np.log10(40), np.log10(20000)
        font = p.font()
        font.setPointSizeF(8)
        p.setFont(font)
        for f, lab in ((100, "100"), (1000, "1k"), (10000, "10k")):
            y = h * (1 - (np.log10(f) - lo) / (hi - lo))
            p.setPen(QPen(qc("#ffffff", 34), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(0, y), QPointF(w, y))
            p.setPen(qc("#ffffff", 150))
            p.drawText(QPointF(8, y - 3), lab + " Hz" if f == 100 else lab)
        x = self.progress * w
        p.setPen(QPen(qc("#ffffff", 235), 2))
        p.drawLine(QPointF(x, 0), QPointF(x, h))


class Bars(QWidget):
    """Analyseur de spectre à barres néon avec retenue de crête."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.vals = np.full(BANDS, -90.0)
        self.peaks = np.full(BANDS, -90.0)

    def set_vals(self, v):
        self.vals = v
        self.peaks = np.maximum(v, self.peaks - 2.5)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        for db in (-20, -40, -60):
            y = h * (1 - (db + 80) / 80)
            p.setPen(QPen(qc("#ffffff", 20), 1))
            p.drawLine(QPointF(0, y), QPointF(w, y))
        g = QLinearGradient(0, h, 0, 0)
        g.setColorAt(0, qc(C1))
        g.setColorAt(0.55, qc(C3))
        g.setColorAt(1, qc(C2))
        gg = QLinearGradient(0, h, 0, 0)
        gg.setColorAt(0, qc(C1, 40))
        gg.setColorAt(1, qc(C2, 40))
        bw = w / BANDS
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(BANDS):
            fr = float(np.clip((self.vals[i] + 80) / 80, 0, 1))
            x, bh = i * bw + bw * 0.16, max(2.0, fr * (h - 4))
            p.setBrush(QBrush(gg))
            p.drawRoundedRect(QRectF(x - 2, h - bh - 2, bw * 0.68 + 4, bh + 4), 4, 4)
            p.setBrush(QBrush(g))
            p.drawRoundedRect(QRectF(x, h - bh, bw * 0.68, bh), 2, 2)
            pk = float(np.clip((self.peaks[i] + 80) / 80, 0, 1))
            p.setBrush(qc("#ffffff", 190))
            p.drawRoundedRect(QRectF(x, h - pk * (h - 4) - 3, bw * 0.68, 2.5), 1, 1)


class HMeter(QWidget):
    """Deux vumètres horizontaux L / R."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(44)
        self.rms, self.pk = [-90.0, -90.0], [-90.0, -90.0]

    def set_levels(self, rms, pk):
        self.rms, self.pk = rms, pk
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width() - 22
        g = QLinearGradient(20, 0, 20 + w, 0)
        g.setColorAt(0, qc(C1))
        g.setColorAt(0.7, qc(C3))
        g.setColorAt(0.9, qc(C2))
        g.setColorAt(1, qc(WARN))
        font = p.font()
        font.setPointSizeF(8)
        font.setBold(True)
        p.setFont(font)
        for i, name in enumerate("LR"):
            y = 4 + i * 20
            p.setPen(qc(MUTED))
            p.drawText(QRectF(0, y, 16, 14), Qt.AlignmentFlag.AlignCenter, name)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qc("#ffffff", 20))
            p.drawRoundedRect(QRectF(20, y + 2, w, 9), 4, 4)
            fr = float(np.clip((self.rms[i] + 60) / 60, 0, 1))
            p.setBrush(QBrush(g))
            p.drawRoundedRect(QRectF(20, y + 2, max(3.0, w * fr), 9), 4, 4)
            pf = float(np.clip((self.pk[i] + 60) / 60, 0, 1))
            p.setBrush(qc("#ffffff", 220))
            p.drawRect(QRectF(20 + w * pf - 1.5, y + 1, 3, 11))


class CurveView(QWidget):
    """Réponse en fréquence du filtre."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(74)
        self.data = None

    def set_curve(self, d):
        self.data = d
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc("#000000", 70))
        p.drawRoundedRect(QRectF(0, 0, w, h), 10, 10)
        p.setPen(QPen(qc("#ffffff", 22), 1))
        for f in (100, 1000, 10000):
            x = w * (np.log10(f) - np.log10(20)) / 3
            p.drawLine(QPointF(x, 6), QPointF(x, h - 6))
        if self.data is None:
            return
        f, db = self.data
        xs = w * (np.log10(f) - np.log10(20)) / 3
        ys = 8 + (6 - db) / 42 * (h - 16)
        path = QPainterPath(QPointF(xs[0], ys[0]))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(x, y)
        fill = QPainterPath(path)
        fill.lineTo(xs[-1], h - 8)
        fill.lineTo(xs[0], h - 8)
        fill.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(C1, 40))
        p.drawPath(fill)
        g = QLinearGradient(0, 0, w, 0)
        g.setColorAt(0, qc(C1))
        g.setColorAt(1, qc(C2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QBrush(g), 2.2))
        p.drawPath(path)


# ======================= Modules du rack =======================
class Module(QFrame):
    changed = pyqtSignal()

    def __init__(self, title, on=False):
        super().__init__()
        self.setObjectName("module")
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(6)
        head = QHBoxLayout()
        self.title = spaced(QLabel(title), 9.5, 2.0)
        self.title.setStyleSheet(f"color: {MUTED};")
        self.led = Led()
        self.led.setChecked(on)
        head.addWidget(self.title, 1)
        head.addWidget(self.led)
        v.addLayout(head)
        self.body = QVBoxLayout()
        self.body.setSpacing(6)
        v.addLayout(self.body, 1)
        self.knobs = []
        self.led.toggled.connect(self._power)

    @property
    def on(self):
        return self.led.isChecked()

    def knob(self, *a, **k):
        kn = Knob(*a, **k)
        kn.valueChanged.connect(self._touch)
        self.knobs.append(kn)
        return kn

    def _touch(self, *_):
        if not self.led.isChecked():
            self.led.setChecked(True)  # toucher un potard active le module
        else:
            self.changed.emit()

    def _power(self, on):
        self.refresh_dim()
        self.changed.emit()

    def refresh_dim(self):
        for k in self.knobs:
            k.dim = not self.on
            k.update()


class SepModule(Module):
    def __init__(self):
        super().__init__("SÉPARATION")
        self.what = Segmented(["VOIX", "INSTRU"])
        self.engine = Segmented(["DSP", "DEMUCS"])
        self.hint = QLabel()
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        cap1, cap2 = QLabel("ISOLER"), QLabel("MOTEUR")
        for c in (cap1, cap2):
            c.setObjectName("cap")
        for w in (cap1, self.what, cap2, self.engine, self.hint):
            self.body.addWidget(w)
        self.body.addStretch(1)
        if not HAS_DEMUCS:
            self.engine.btns[1].setEnabled(False)
        self.what.changed.connect(lambda _: self._touch())
        self.engine.changed.connect(lambda _: self._touch())
        self.engine.changed.connect(self._hint)
        self._hint()

    def _hint(self, *_):
        if self.engine.index() == 1:
            self.hint.setText("Réseau de neurones : précis mais lent sans GPU.")
        elif HAS_DEMUCS:
            self.hint.setText("Centre stéréo (Mid/Side) : instantané, approximatif.")
        else:
            self.hint.setText("Centre stéréo (Mid/Side) : instantané.\nDemucs : pip install demucs")

    def spec(self):
        mode = ("Voix" if self.what.index() == 0 else "Instru") + (" (Demucs)" if self.engine.index() == 1
                                                                 else " (centre, DSP)")
        return dict(type="sep", on=self.on, mode=mode)


class FilterModule(Module):
    KINDS = ["highpass", "lowpass", "bandpass"]

    def __init__(self):
        super().__init__("FILTRE")
        self.kind = Segmented(["PASSE-HAUT", "PASSE-BAS", "BANDE"])
        self.body.addWidget(self.kind)
        self.k1 = self.knob("COUPURE", 20, 20000, 200, fmt=fmt_hz, log=True, size=72)
        self.k2 = self.knob("HAUTE", 20, 20000, 4000, fmt=fmt_hz, log=True, size=72)
        self.ko = self.knob("ORDRE", 1, 8, 4, step=1, fmt=lambda v: f"{v:.0f}", size=72)
        row = QHBoxLayout()
        row.setSpacing(0)
        for k in (self.k1, self.k2, self.ko):
            row.addWidget(k)
        self.body.addLayout(row)
        self.curve = CurveView()
        self.body.addWidget(self.curve)
        self.kind.changed.connect(self._on_kind)
        for k in (self.k1, self.k2, self.ko):
            k.valueChanged.connect(self._refresh)
        self._refresh()
        self.refresh_dim()

    def _on_kind(self, i):
        if i == 1:
            self.k1.set_value(4000)
        elif i == 0:
            self.k1.set_value(200)
        elif i == 2:
            self.k1.set_value(200)
        self._refresh()
        self._touch()

    def _refresh(self, *_):
        bp = self.kind.index() == 2
        self.k2.setVisible(bp)
        self.k1.caption = "BASSE" if bp else "COUPURE"
        self.k1.update()
        s = self.spec()
        self.curve.set_curve(filter_response(s["kind"], s["order"], s["f1"], s["f2"]))

    def spec(self):
        return dict(type="filter", on=self.on, kind=self.KINDS[self.kind.index()], order=int(self.ko.value()),
                    f1=self.k1.value(), f2=self.k2.value())


class PitchModule(Module):
    def __init__(self):
        super().__init__("PITCH")
        self.k = self.knob("DEMI-TONS", -12, 12, 0, step=0.5, bipolar=True, size=118,
                           fmt=lambda v: f"{v:+.1f} st")
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.k)
        row.addStretch(1)
        self.body.addLayout(row)
        self.info = QLabel()
        self.info.setObjectName("hint")
        self.info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.body.addWidget(self.info)
        self.body.addStretch(1)
        self.k.valueChanged.connect(self._info)
        self._info()
        self.refresh_dim()

    def _info(self, *_):
        v = self.k.value()
        self.info.setText(f"fréquence ×{2 ** (v / 12):.3f}  ·  {v * 100:+.0f} cents")

    def spec(self):
        return dict(type="pitch", on=self.on, st=self.k.value())


class ReverbModule(Module):
    def __init__(self):
        super().__init__("REVERB")
        self.kd = self.knob("TAILLE", 0.2, 5, 1.5, step=0.1, fmt=lambda v: f"{v:.1f} s", size=76)
        self.kw = self.knob("MIX", 0, 1, 0.3, step=0.01, fmt=lambda v: f"{v * 100:.0f} %", size=76)
        self.kp = self.knob("PRÉ-DÉLAI", 0, 100, 20, step=1, fmt=lambda v: f"{v:.0f} ms", size=76)
        row = QHBoxLayout()
        row.setSpacing(0)
        for k in (self.kd, self.kw, self.kp):
            row.addWidget(k)
        self.body.addStretch(1)
        self.body.addLayout(row)
        self.body.addStretch(1)
        self.refresh_dim()

    def spec(self):
        return dict(type="reverb", on=self.on, decay=self.kd.value(), wet=self.kw.value(), pre=self.kp.value())


class GainModule(Module):
    def __init__(self):
        super().__init__("GAIN SORTIE")
        self.k = self.knob("NIVEAU", -24, 24, 0, step=0.5, bipolar=True, size=112, fmt=lambda v: f"{v:+.1f} dB")
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.k)
        row.addStretch(1)
        self.body.addStretch(1)
        self.body.addLayout(row)
        self.body.addStretch(1)
        self.refresh_dim()

    def spec(self):
        return dict(type="gain", on=self.on, db=self.k.value())


# ======================= Fenêtre principale =======================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Lab")
        self.resize(1480, 940)
        self.setAcceptDrops(True)
        self.sr = dsp.SR
        self.paths, self.path, self.want = [], None, None
        self.total, self.start = 0.0, 0
        self.orig = self.res = None
        self.view = "res"
        self.cursor_t, self.dur_buf = 0.0, 0.0
        self.specs = {"orig": None, "res": None}
        self.wavs, self.n, self.gen, self.load_gen = {}, 0, 0, 0
        self.tmp = tempfile.mkdtemp(prefix="audiolab_")
        self.workers = set()
        self.chain_busy = self.chain_pending = False
        self.resume = None
        self.player = QMediaPlayer()
        self.aout = QAudioOutput()
        self.player.setAudioOutput(self.aout)
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.tick)
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(700)
        self.debounce.timeout.connect(self.run_chain)
        self.exc_debounce = QTimer(self)
        self.exc_debounce.setSingleShot(True)
        self.exc_debounce.setInterval(400)
        self.exc_debounce.timeout.connect(self.refresh_excerpt)
        f = np.fft.rfftfreq(4096, 1 / dsp.SR)
        edges = np.logspace(np.log10(30), np.log10(18000), BANDS + 1)
        self.b_lo = np.searchsorted(f, edges[:-1])
        self.b_hi = np.maximum(np.searchsorted(f, edges[1:]), self.b_lo + 1)
        self._build()
        self.player.playbackStateChanged.connect(self.on_state)
        self.player.mediaStatusChanged.connect(self.on_media_status)
        self.set_status("Glisse un fichier audio ici ou ouvre la bibliothèque", "ok")

    # ----- UI -----
    def _build(self):
        root = Backdrop()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(26, 18, 26, 20)
        v.setSpacing(14)

        # En-tête
        head = QHBoxLayout()
        logo = QLabel(f"<span style='color:{C1}'>AUDIO</span><span style='color:{C2}'>LAB</span>")
        spaced(logo, 17, 4.0)
        sub = QLabel("Hamza Azroud")
        spaced(sub, 7.5, 3.0)
        sub.setStyleSheet(f"color: {MUTED};")
        lg = QVBoxLayout()
        lg.setSpacing(0)
        lg.addWidget(logo)
        lg.addWidget(sub)
        self.title = QLabel("Aucun fichier")
        self.title.setTextFormat(Qt.TextFormat.RichText)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_lib = QToolButton()
        self.btn_lib.setText("♫  Bibliothèque ▾")
        self.btn_lib.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.lib_menu = QMenu(self.btn_lib)
        self.lib_menu.aboutToShow.connect(self.fill_library)
        self.btn_lib.setMenu(self.lib_menu)
        self.btn_exp = QToolButton()
        self.btn_exp.setText("⇩  Exporter ▾")
        self.btn_exp.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        em = QMenu(self.btn_exp)
        em.addAction("Extrait traité (WAV)...", self.export_excerpt)
        em.addAction("Fichier entier traité (WAV)...", self.export_full)
        self.btn_exp.setMenu(em)
        head.addLayout(lg)
        head.addStretch(1)
        head.addWidget(self.title, 2)
        head.addStretch(1)
        head.addWidget(self.btn_lib)
        head.addSpacing(8)
        head.addWidget(self.btn_exp)
        v.addLayout(head)

        # Hero (onde + spectrogramme) + analyseur
        mid = QHBoxLayout()
        mid.setSpacing(14)
        hero = QFrame()
        hero.setObjectName("hero")
        hv = QVBoxLayout(hero)
        hv.setContentsMargins(18, 14, 18, 16)
        hv.setSpacing(10)
        hh = QHBoxLayout()
        lab1 = spaced(QLabel("FORME D'ONDE  ·  SPECTROGRAMME"), 8.5, 2.0)
        lab1.setStyleSheet(f"color: {MUTED};")
        self.lbl_range = QLabel("")
        self.lbl_range.setObjectName("cap")
        hh.addWidget(lab1, 1)
        hh.addWidget(self.lbl_range)
        hv.addLayout(hh)
        self.wave = WaveBars()
        self.specv = SpecView()
        hv.addWidget(self.wave, 0)
        hv.addWidget(self.specv, 1)
        self.wave.seek.connect(self.on_seek)
        self.specv.seek.connect(self.on_seek)
        mid.addWidget(hero, 1)

        glass = QFrame()
        glass.setObjectName("glass")
        glass.setFixedWidth(340)
        gv = QVBoxLayout(glass)
        gv.setContentsMargins(18, 14, 18, 16)
        gv.setSpacing(10)
        lab2 = spaced(QLabel("ANALYSE  ·  AU CURSEUR"), 8.5, 2.0)
        lab2.setStyleSheet(f"color: {MUTED};")
        gv.addWidget(lab2)
        self.bars = Bars()
        gv.addWidget(self.bars, 1)
        self.hm = HMeter()
        gv.addWidget(self.hm)
        self.lbl_peak = QLabel("—")
        self.lbl_peak.setObjectName("val")
        self.lbl_corr = QLabel("—")
        self.lbl_corr.setObjectName("val")
        for cap, lab in (("PIC", self.lbl_peak), ("STÉRÉO", self.lbl_corr)):
            r = QHBoxLayout()
            c = QLabel(cap)
            c.setObjectName("cap")
            r.addWidget(c)
            r.addStretch(1)
            r.addWidget(lab)
            gv.addLayout(r)
        mid.addWidget(glass)
        v.addLayout(mid, 1)

        # Transport
        tr = QHBoxLayout()
        tr.setSpacing(14)
        self.btn_play = PlayButton()
        self.btn_stop = QToolButton()
        self.btn_stop.setText("■")
        self.btn_stop.setFixedSize(44, 44)
        self.lbl_time = QLabel("0:00.0")
        spaced(self.lbl_time, 15, 1.0)
        self.lbl_time.setTextFormat(Qt.TextFormat.RichText)
        self.ab = Segmented(["ORIGINAL", "RÉSULTAT"])
        self.ab.set_index(1)
        self.sp_start = QSpinBox()
        self.sp_start.setSuffix(" s")
        self.sp_dur = QSpinBox()
        self.sp_dur.setSuffix(" s")
        self.sp_dur.setRange(1, 1)
        self.lbl_status = QLabel()
        self.lbl_status.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_status.setObjectName("hint")
        tr.addWidget(self.btn_play)
        tr.addWidget(self.btn_stop)
        tr.addWidget(self.lbl_time)
        tr.addSpacing(10)
        tr.addWidget(self.ab)
        tr.addStretch(1)
        c1, c2 = QLabel("DÉBUT"), QLabel("DURÉE")
        for c in (c1, c2):
            c.setObjectName("cap")
        tr.addWidget(c1)
        tr.addWidget(self.sp_start)
        tr.addWidget(c2)
        tr.addWidget(self.sp_dur)
        tr.addSpacing(16)
        tr.addWidget(self.lbl_status)
        v.addLayout(tr)

        self.lbl_err = QLabel()
        self.lbl_err.setWordWrap(True)
        self.lbl_err.setStyleSheet(f"color: {ERR};")
        self.lbl_err.hide()
        v.addWidget(self.lbl_err)

        # Rack d'effets (ordre = ordre de traitement)
        rack = QHBoxLayout()
        rack.setSpacing(12)
        self.modules = [SepModule(), FilterModule(), PitchModule(), ReverbModule(), GainModule()]
        for m in self.modules:
            m.changed.connect(self.schedule)
            m.setMinimumHeight(250)
            rack.addWidget(m, 1)
        v.addLayout(rack)

        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_stop.clicked.connect(self.stop)
        self.ab.changed.connect(self.on_ab)
        self.sp_start.valueChanged.connect(lambda _: self.exc_debounce.start())
        self.sp_dur.valueChanged.connect(lambda _: self.exc_debounce.start())

    def set_status(self, text, kind="ok"):
        col = {"ok": OK, "busy": BUSY, "err": ERR}[kind]
        self.lbl_status.setText(f"<span style='color:{col}'>●</span>&nbsp; {html.escape(text)}")

    def set_error(self, text):
        self.lbl_err.setText(text)
        self.lbl_err.setVisible(bool(text))

    # ----- Drag & drop -----
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_files([u.toLocalFile() for u in e.mimeData().urls()])

    # ----- Threads -----
    def run(self, fn, args, on_done, on_fail=None):
        w = Worker(fn, *args)
        self.workers.add(w)
        w.done.connect(on_done)
        w.failed.connect(on_fail or self.fail)
        w.finished.connect(lambda: self.workers.discard(w))
        w.start()

    def fail(self, msg):
        self.set_status("Erreur", "err")
        QMessageBox.warning(self, "Erreur", msg)

    # ----- Bibliothèque / fichiers -----
    def fill_library(self):
        m = self.lib_menu
        m.clear()
        for p in self.paths:
            a = QAction(("●  " if p == self.path else "     ") + os.path.basename(p), m)
            a.triggered.connect(lambda _=False, p=p: self.select(p))
            m.addAction(a)
        if self.paths:
            m.addSeparator()
        m.addAction("Ouvrir un fichier...", self.open_dialog)
        m.addAction("Télécharger depuis une URL...", self.add_url)

    def open_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Ouvrir", "", "Audio (*.mp3 *.wav *.flac *.ogg *.m4a)")
        self.add_files(paths)

    def add_url(self):
        url, ok = QInputDialog.getText(self, "Télécharger", "URL (contenu dont tu as les droits) :")
        if ok and url.strip():
            self.set_status("Téléchargement...", "busy")
            self.run(dsp.download, (url.strip(),), lambda p: self.add_files([p]))

    def add_files(self, paths):
        last = None
        for p in paths:
            p = os.path.abspath(p)
            if os.path.isfile(p):
                if p not in self.paths:
                    self.paths.append(p)
                last = p
        if last:
            self.select(last)

    def select(self, path):
        if path == self.path:
            return
        self.want = path
        self.load_gen += 1
        self.stop()
        self.set_status("Ouverture...", "busy")
        self.run(open_job, (path, 0, DEFAULT_DUR), self.on_opened)

    def on_opened(self, r):
        if r["path"] != self.want:
            return
        self.path, self.total, self.start = r["path"], r["total"], r["start"]
        x, total = r["x"], r["total"]
        for sp in (self.sp_start, self.sp_dur):
            sp.blockSignals(True)
        self.sp_start.setRange(0, max(int(total) - 1, 0))
        self.sp_start.setValue(r["start"])
        self.sp_dur.setRange(1, max(int(total), 1))
        self.sp_dur.setValue(max(1, int(round(x.shape[1] / self.sr))))
        for sp in (self.sp_start, self.sp_dur):
            sp.blockSignals(False)
        self.title.setText(
            f"<span style='font-size:15px; font-weight:700'>{html.escape(os.path.basename(self.path))}</span><br>"
            f"<span style='color:{MUTED}; font-size:11px'>{fmt_t(total)}  ·  {self.sr / 1000:g} kHz  ·  stéréo</span>")
        self.set_status("Prêt", "ok")
        self.apply_excerpt(x, r["spec"], r["start"])

    # ----- Extrait -----
    def refresh_excerpt(self):
        if not self.path:
            return
        self.load_gen += 1
        g = self.load_gen
        self.set_status("Lecture de l'extrait...", "busy")
        self.run(excerpt_job, (self.path, self.sp_start.value(), self.sp_dur.value(), self.total),
                 lambda r, g=g: self.on_excerpt(r, g))

    def on_excerpt(self, r, g):
        if g != self.load_gen or r["path"] != self.path:
            return
        self.set_status("Prêt", "ok")
        self.apply_excerpt(r["x"], r["spec"], r["start"])

    def apply_excerpt(self, x, spec, start):
        if x.shape[1] < 4096:
            self.set_status("Extrait trop court", "err")
            return
        self.gen += 1
        self.start = start
        self.orig, self.res = x, None
        self.specs = {"orig": spec, "res": None}
        self.dur_buf = x.shape[1] / self.sr
        self.lbl_range.setText(f"EXTRAIT  {fmt_t(start)}  →  {fmt_t(start + self.dur_buf)}")
        self.commit("orig", x)
        self.cursor_t = 0.0
        self.refresh_visuals()
        self.set_cursor(0.0)
        self.load_player()
        self.run_chain()

    def current(self):
        if self.view == "res" and self.res is not None:
            return "res", self.res
        return "orig", self.orig

    def commit(self, key, buf):
        self.n += 1
        path = os.path.join(self.tmp, f"{key}_{self.n}.wav")
        dsp.save_wav(path, buf, self.sr)
        self.wavs[key] = path

    # ----- Chaîne d'effets -----
    def collect_specs(self):
        return [m.spec() for m in self.modules]

    def schedule(self):
        if self.orig is not None:
            self.debounce.start()

    def run_chain(self):
        if self.orig is None:
            return
        if self.chain_busy:
            self.chain_pending = True
            return
        self.chain_busy = True
        self.set_status("Traitement...", "busy")
        gen = self.gen
        self.run(chain_job, (self.orig, self.sr, self.collect_specs()),
                 lambda r, g=gen: self.on_chain(r, g), self.on_chain_fail)

    def on_chain(self, r, gen):
        y, info, spec = r
        self.chain_busy = False
        if gen == self.gen:
            self.res = y
            self.specs["res"] = spec
            self.commit("res", y)
            if self.view == "res":
                self.refresh_visuals()
                self.load_player()
            slow = " · ".join(f"{n} {t:.1f} s" for n, t in info["times"] if t >= 0.05)
            self.set_status("Prêt" + (f" — {slow}" if slow else ""), "ok")
            self.set_error("\n".join("⚠ " + w for w in info["warns"]))
        if self.chain_pending:
            self.chain_pending = False
            self.run_chain()

    def on_chain_fail(self, msg):
        self.chain_busy = self.chain_pending = False
        self.set_status("Erreur", "err")
        self.set_error("⚠ " + msg)

    # ----- Affichage -----
    def refresh_visuals(self):
        key, b = self.current()
        if b is None:
            return
        other = self.orig if key == "res" else self.res
        self.wave.set_data(fine_env(b), fine_env(other) if other is not None else None)
        if self.specs[key] is None:
            self.specs[key] = compute_spec(b, self.sr)
        self.specv.set_image(self.specs[key])
        self.update_analysis()

    def update_analysis(self):
        key, b = self.current()
        if b is None:
            return
        N = 4096
        i = int(self.cursor_t * self.sr)
        seg = b[:, i:i + N]
        if seg.shape[1] < N:
            seg = np.pad(seg, ((0, 0), (0, N - seg.shape[1])))
        mag = np.abs(np.fft.rfft(seg.mean(axis=0) * np.hanning(N))) / (N / 4)
        db = 20 * np.log10(mag + 1e-6)
        self.bars.set_vals(np.array([db[lo:hi].max() for lo, hi in zip(self.b_lo, self.b_hi)]))
        f = np.fft.rfftfreq(N, 1 / self.sr)
        k = int(np.argmax(np.where((f >= 30) & (f <= 18000), db, -1e9)))
        if db[k] > -90 and 0 < k < len(db) - 1:
            a, c0, c = db[k - 1], db[k], db[k + 1]
            den = a - 2 * c0 + c
            d = 0.5 * (a - c) / den if den else 0.0
            fp = (k + d) * self.sr / N
            self.lbl_peak.setText(f"{fp:.1f} Hz  ·  {note_name(fp)}  ·  {db[k]:.1f} dB")
        else:
            self.lbl_peak.setText("—")
        L, R = seg[0], seg[1]
        sl, sr_ = float(np.sqrt(np.mean(L ** 2))), float(np.sqrt(np.mean(R ** 2)))
        corr = float(np.corrcoef(L, R)[0, 1]) if L.std() > 1e-9 and R.std() > 1e-9 else 0.0
        bal = 20 * np.log10((sr_ + 1e-9) / (sl + 1e-9))
        self.lbl_corr.setText(f"corr {corr:+.2f}  ·  bal {bal:+.1f} dB")
        self.hm.set_levels([20 * np.log10(sl + 1e-9), 20 * np.log10(sr_ + 1e-9)],
                           [20 * np.log10(np.max(np.abs(L)) + 1e-9), 20 * np.log10(np.max(np.abs(R)) + 1e-9)])

    # ----- Curseur / lecture -----
    def on_seek(self, frac):
        if self.orig is not None:
            self.set_cursor(frac * self.dur_buf, seek=True)

    def set_cursor(self, t, seek=False):
        t = float(np.clip(t, 0, self.dur_buf))
        self.cursor_t = t
        prog = t / self.dur_buf if self.dur_buf else 0.0
        self.wave.set_progress(prog)
        self.specv.set_progress(prog)
        self.lbl_time.setText(f"{fmt_t(t)} <span style='color:{MUTED}; font-size:11px'>/ {fmt_t(self.dur_buf)}</span>")
        if seek and not self.player.source().isEmpty():
            self.player.setPosition(int(t * 1000))
        self.update_analysis()

    def load_player(self):
        key, _ = self.current()
        path = self.wavs.get(key)
        if not path:
            return
        was = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        self.resume = (self.cursor_t, was)
        self.player.setSource(QUrl.fromLocalFile(path))

    def on_media_status(self, st):
        if st == QMediaPlayer.MediaStatus.LoadedMedia and self.resume:
            t, was = self.resume
            self.resume = None
            self.player.setPosition(int(t * 1000))
            if was:
                self.player.play()

    def on_ab(self, i):
        self.view = "orig" if i == 0 else "res"
        self.refresh_visuals()
        self.load_player()

    def toggle_play(self):
        if self.orig is None:
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            return
        if self.cursor_t >= self.dur_buf - 0.05:
            self.set_cursor(0.0)
        self.player.setPosition(int(self.cursor_t * 1000))
        self.player.play()

    def stop(self):
        self.player.stop()
        if self.orig is not None:
            self.set_cursor(0.0)

    def on_state(self, st):
        playing = st == QMediaPlayer.PlaybackState.PlayingState
        self.btn_play.playing = playing
        self.btn_play.update()
        self.timer.start() if playing else self.timer.stop()

    def tick(self):
        self.set_cursor(self.player.position() / 1000)

    # ----- Export -----
    def export_excerpt(self):
        buf = self.res if self.res is not None else self.orig
        if buf is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exporter l'extrait", "extrait.wav", "WAV (*.wav)")
        if path:
            dsp.save_wav(path, buf, self.sr)
            self.set_status(f"Exporté : {os.path.basename(path)}", "ok")

    def export_full(self):
        if not self.path:
            return
        out, _ = QFileDialog.getSaveFileName(self, "Exporter le fichier", "resultat.wav", "WAV (*.wav)")
        if not out:
            return
        specs, src = self.collect_specs(), self.path

        def job():
            y, sr = dsp.load_audio(src)  # décodage complet, uniquement ici
            dsp.save_wav(out, process_chain(y, sr, specs, use_cache=False)[0], sr)
            return out

        self.set_status("Traitement du fichier entier... (peut être long)", "busy")
        self.run(job, (), lambda p: self.set_status(f"Exporté : {os.path.basename(p)}", "ok"))

    def closeEvent(self, e):
        self.player.stop()
        super().closeEvent(e)


if __name__ == "__main__":
    sys.excepthook = lambda *a: traceback.print_exception(*a)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(QSS)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())