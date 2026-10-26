"""
Persistent Identity Manager
Turns per-frame track observations into stable person UIDs that survive exits, re-entries and restarts.

How a UID is decided:
1. Tentative phase - a new track shows "Identifying..." until it has a few good observations
   (or its maximum tentative window is reached).
2. The track's evidence (quality-weighted appearance descriptor, Lab colour consensus of the upper and
   lower garment, garment type, accessory hit/check statistics) is scored against every gallery
   identity that is not held by another visible track:
     appearance  - centered cosine vs. per-identity prototypes (max) and EMA mean, mapped to 0..1 by an
                   online calibration (impostor pairs = concurrently visible tracks, genuine pairs = the
                   same track over time); blended with raw cosine while the population is still small
     colours     - CIEDE2000 similarity, with vetoes for confident strong contradictions
     garment     - pants vs. bare-legs veto when both sides are confident
     accessories - identity-stable accessories only (both present = agreement, present vs. clearly
                   absent = disagreement); transient items are ignored for scoring
   Cue weights are renormalised over the cues available for that pair.
3. Ready tentative tracks are resolved jointly (Hungarian) with exclusivity; a match needs a score
   threshold and a margin over the runner-up, otherwise a new UID is minted from the DB sequence.
4. Confirmed tracks keep refining their identity (diverse prototypes, EMA mean, attribute profile),
   are checked for drift (tracker ID switches), and identities freshly created by a track are
   late-merged into an older identity when the accumulated evidence strongly matches it.
"""

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - scipy is a declared dependency
    linear_sum_assignment = None
    SCIPY_AVAILABLE = False

from forensic_app.core.catalog import (
    ACCESSORY_LABELS,
    IDENTITY_STABLE_ACCESSORIES,
    LOWER_GARMENT_TYPES,
    normalize_accessory,
)

logger = logging.getLogger(__name__)

# Canonical accessory order (= catalog / UI order) and vector layout used for hit/check statistics
ACCESSORY_ORDER: Tuple[str, ...] = tuple(ACCESSORY_LABELS.keys())
_ACC_INDEX: Dict[str, int] = {name: i for i, name in enumerate(ACCESSORY_ORDER)}
_N_ACC = len(ACCESSORY_ORDER)
_STABLE_MASK = np.array([n in IDENTITY_STABLE_ACCESSORIES for n in ACCESSORY_ORDER], dtype=bool)

TYPE_PANTS, TYPE_BARE, TYPE_OCCLUDED = LOWER_GARMENT_TYPES
_TYPE_CODE = {TYPE_PANTS: 0, TYPE_BARE: 1, TYPE_OCCLUDED: 2}
_CODE_TYPE = {v: k for k, v in _TYPE_CODE.items()}

TENTATIVE_LABEL = "Identifying..."
_UNKNOWN_NAMES = {"", "unknown", "not detected", "none", "any"}
_NEG = -1.0e6  # "impossible" score used inside assignment matrices


# =====================================================================================  configuration
@dataclass
class IdentityConfig:
    """Every threshold of the identity manager (scores and similarities are 0..1 unless stated)."""

    # --- tentative phase
    min_good_observations: int = 5        # good observations before the first match/new decision
    max_tentative_observations: int = 15  # forced decision (match or new uid) after this many observations
    min_usable_for_new: int = 3           # a forced new uid needs at least this many usable observations
    good_quality: float = 0.35            # quality from which an observation counts as "good"
    min_quality: float = 0.10             # observations below this are ignored as evidence
    clear_new_score: float = 0.40         # first decision: best candidate below this -> new uid at once
    quick_min_good: int = 2               # fast re-acquire (tracker fragmentation): good observations ...
    quick_recent_frames: int = 90         # ... of an identity seen within this many frames ...
    quick_threshold: float = 0.78         # ... scoring at least this ...
    quick_margin: float = 0.20            # ... with this margin over the runner-up

    # --- track buffers
    track_buffer: int = 30                # recent embeddings used for the track descriptor
    color_buffer: int = 40                # recent colour / accessory samples kept per track

    # --- appearance calibration (online, with priors)
    raw_impostor_prior: float = 0.75      # expected raw cosine of two different people (CNN features share a
                                          # large common component, and a fixed camera shares its background)
    raw_genuine_prior: float = 0.92       # expected raw cosine of the same person
    centered_impostor_prior: float = 0.05  # same, after subtracting the population mean
    centered_genuine_prior: float = 0.60
    calib_prior_weight: float = 20.0      # pseudo-count of the priors
    calib_min_rate: float = 0.02          # slowest EMA rate once many samples were seen
    calib_max_shift: float = 0.20         # learned means stay within prior +- this (bounded updates)
    calib_genuine_max_up: float = 0.03    # genuine level may rise at most this much above its prior:
                                          # same-track pairs are easier than cross-sighting pairs
    calib_min_gap: float = 0.15           # enforced separation (genuine - impostor)
    calib_spread_k: float = 0.5           # genuine level used = genuine mean - k*std
    calib_impostor_k: float = 1.0         # score is 0 up to impostor mean + k*std ...
    calib_impostor_full_k: float = 5.0    # ... and saturates at impostor mean + this*std (or the genuine level)
    calib_multi_k: float = 0.35           # open-set correction: floor rises by k*sqrt(2 ln N) impostor stds
                                          # for N candidate identities (best-of-N impostors score higher)
    raw_impostor_std_prior: float = 0.05  # prior spread of raw impostor similarities
    centered_impostor_std_prior: float = 0.08
    genuine_min_gap: int = 10             # frames between the two windows of a genuine pair
    calib_pair_interval: int = 10         # min frames between two samples of one track / track pair
    calib_gallery_interval: int = 30      # confirmed track vs. other stored identities = impostor samples ...
    calib_gallery_samples: int = 4        # ... this many identities per check (works without concurrency)
    calib_seed_pairs: int = 20            # stored identity pairs used to seed impostor statistics at load
    center_full_identities: int = 8       # population size at which centering is fully trusted
    center_refresh_frames: int = 10       # recompute the population mean at most this often
    proto_weight: float = 0.6             # appearance sim = w * max(prototypes) + (1 - w) * sim(mean)

    # --- colours / garment type
    color_scale: float = 22.0             # CIEDE2000 distance -> similarity scale (~0.37 at this distance)
    color_veto_delta_e: float = 35.0      # CIEDE2000 above this is a contradiction ...
    color_veto_confidence: float = 0.45   # ... when both colour consensuses are at least this confident
    color_veto_min_samples: int = 3       # ... and built from at least this many samples
    min_color_confidence: float = 0.15    # per-frame colour samples below this confidence are ignored
    garment_type_veto_confidence: float = 0.60  # pants vs. bare legs veto needs this type confidence

    # --- accessories
    accessory_min_confidence: float = 0.20  # per-frame detection confidence counted as a hit
    accessory_min_hits: int = 2           # confirmed when hits >= this ...
    accessory_min_ratio: float = 0.30     # ... and hits / checks >= this
    accessory_absent_min_checks: int = 8  # strongly absent when checked at least this often ...
    accessory_absent_max_ratio: float = 0.05  # ... with a hit ratio below this

    # --- fusion
    w_appearance: float = 0.50
    w_upper: float = 0.22
    w_lower: float = 0.13
    w_accessories: float = 0.15
    match_threshold: float = 0.62         # fused score needed to re-use an existing uid
    match_margin: float = 0.05            # ... with this margin over the runner-up
    forced_margin_bonus: float = 0.08     # forced decision: margin waived when score >= threshold + this
    duplicate_pair_score: float = 0.80    # margin waived when the runner-up looks like a duplicate of the best
    no_appearance_threshold: float = 0.80  # stricter threshold when the appearance cue is unavailable
    no_appearance_min_cues: int = 2       # ... and at least this many attribute cues must be available

    # --- late merge of provisional identities
    merge_threshold: float = 0.70
    merge_margin: float = 0.08
    merge_checkpoints: Tuple[int, ...] = (15, 30, 60, 120)  # observations since the uid was minted

    # --- drift protection (tracker identity switches)
    drift_check_every: int = 5            # observations between two drift checks
    drift_window: int = 10                # recent observations compared with the assigned identity
    drift_threshold: float = 0.35         # recent-window score below this is a contradiction
    drift_patience: int = 3               # consecutive contradictions before the track is detached

    # --- gallery maintenance
    gallery_min_quality: float = 0.45     # only observations at least this good update the gallery
    gallery_update_every: int = 3         # observations between two gallery updates of a track
    gallery_consistency: float = 0.20     # single-observation appearance score needed to update
    proto_novelty_centered: float = 0.85  # add a prototype when its max centered sim is below this
    proto_novelty_raw: float = 0.95       # same in raw cosine space (used while centering is weak)
    max_prototypes: int = 24
    proto_min_interval: int = 10          # frames between two prototype changes of one identity
    mean_ema: float = 0.05                # slowest EMA rate of an identity's mean embedding
    color_ema: float = 0.10               # EMA rate of an identity's colour profile per update

    # --- lifecycle
    exclusivity_frames: int = 3           # a uid held by a track seen within this many frames is blocked
    track_timeout_frames: int = 150       # unseen tracks are ended automatically after this many frames
    status_hold_frames: int = 45          # "new"/"reacquired" status is shown this long, then "confirmed"
    flush_every_frames: int = 300         # dirty profiles are persisted at least this often
    best_crop_margin: float = 0.02        # quality improvement needed to flag a new best crop
    load_crop_quality_decay: float = 0.85  # persisted best-crop quality decays per session (fresh crops win)


# =====================================================================================  public records
@dataclass
class TrackObservation:
    """Everything the pipeline measured for one track in one frame."""
    track_id: int
    frame_idx: int
    bbox: Tuple[int, int, int, int]
    embedding: np.ndarray                 # (512,) raw ReID embedding (zeros = unavailable)
    upper: Dict[str, Any]                 # garment dict {"color","hex","lab","confidence"}
    lower: Dict[str, Any]                 # garment dict incl. "type"
    accessories: Dict[str, float]         # canonical name -> confidence detected THIS frame
    accessory_checked: bool = True        # False when accessory detection was throttled this frame
    quality: float = 1.0                  # 0..1 observation quality
    posture: str = "Standing"
    biometrics: Dict[str, float] = field(default_factory=dict)
    crop: Optional[np.ndarray] = None     # BGR crop (for best-thumbnail selection)


@dataclass
class IdentityAssignment:
    """Identity decision for one track in the current frame."""
    track_id: int
    person_uid: Optional[str]             # None while tentative
    status: str                           # "tentative" | "new" | "reacquired" | "confirmed"
    confidence: float                     # 0..1
    display_label: str                    # "UID-0007" or "Identifying..."
    upper: Dict[str, Any]                 # consensus {"color","hex","lab","confidence"}
    lower: Dict[str, Any]                 # consensus {"color","hex","lab","confidence","type"}
    accessories: List[str]                # consensus accessories, catalog order
    posture: str
    merged_from: Optional[str] = None     # provisional uid merged into person_uid this frame
    is_best_crop: bool = False            # this frame's crop is the best seen so far for the identity


# =====================================================================================  pure helpers
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    """L2-normalise; None for zero / non-finite vectors."""
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-8:
        return None
    return v / n


def _as_lab(val: Any) -> Optional[np.ndarray]:
    """Parse a Lab triple (list/tuple/array) or return None."""
    if val is None:
        return None
    try:
        arr = np.asarray(val, dtype=np.float64).ravel()
    except Exception:
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3].copy()


def _known_name(name: Optional[str]) -> bool:
    return bool(name) and str(name).strip().lower() not in _UNKNOWN_NAMES


def _classify_lab(lab: np.ndarray) -> Tuple[str, str]:
    """Colour name + hex of a Lab value (palette of utils.color_naming)."""
    try:
        from forensic_app.utils.color_naming import classify_cielab_color
        name, hx = classify_cielab_color(np.asarray(lab, dtype=np.float32))
        return str(name), str(hx)
    except Exception:
        return "Unknown", "#808080"


def _color_family(name: Optional[str]) -> str:
    try:
        from forensic_app.core.database import color_family_of
        return color_family_of(name)
    except Exception:
        return "unknown"


def color_name_similarity(name_a: Optional[str], name_b: Optional[str]) -> float:
    """Similarity of two colour NAMES (fallback when Lab values are missing)."""
    a, b = (name_a or "").strip().lower(), (name_b or "").strip().lower()
    if a == b:
        return 1.0
    fa, fb = _color_family(a), _color_family(b)
    if fa != "unknown" and fa == fb:
        return 0.7
    return 0.15


def ciede2000_batch(lab: Sequence[float], labs: np.ndarray) -> np.ndarray:
    """Vectorised CIEDE2000 distance between one Lab colour and an (M, 3) array of Lab colours."""
    labs = np.asarray(labs, dtype=np.float64).reshape(-1, 3)
    L1, a1, b1 = (float(v) for v in np.asarray(lab, dtype=np.float64).ravel()[:3])
    L2, a2, b2 = labs[:, 0], labs[:, 1], labs[:, 2]

    c_bar = (math.hypot(a1, b1) + np.hypot(a2, b2)) / 2.0
    c7 = c_bar ** 7
    g = 0.5 * (1.0 - np.sqrt(c7 / (c7 + 25.0 ** 7)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    d_lp = L2 - L1
    d_cp = c2p - c1p
    prod_zero = (c1p * c2p) == 0.0
    d_hp = h2p - h1p
    d_hp = np.where(d_hp > 180.0, d_hp - 360.0, d_hp)
    d_hp = np.where(d_hp < -180.0, d_hp + 360.0, d_hp)
    d_hp = np.where(prod_zero, 0.0, d_hp)
    d_big_hp = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(d_hp) / 2.0)

    lp_bar = (L1 + L2) / 2.0
    cp_bar = (c1p + c2p) / 2.0
    h_sum = h1p + h2p
    hp_bar = np.where(np.abs(h1p - h2p) <= 180.0, h_sum / 2.0,
                      np.where(h_sum < 360.0, (h_sum + 360.0) / 2.0, (h_sum - 360.0) / 2.0))
    hp_bar = np.where(prod_zero, h_sum, hp_bar)

    t = (1.0
         - 0.17 * np.cos(np.radians(hp_bar - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hp_bar))
         + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0)))
    d_theta = 30.0 * np.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    cp7 = cp_bar ** 7
    r_c = 2.0 * np.sqrt(cp7 / (cp7 + 25.0 ** 7))
    s_l = 1.0 + (0.015 * (lp_bar - 50.0) ** 2) / np.sqrt(20.0 + (lp_bar - 50.0) ** 2)
    s_c = 1.0 + 0.045 * cp_bar
    s_h = 1.0 + 0.015 * cp_bar * t
    r_t = -np.sin(np.radians(2.0 * d_theta)) * r_c
    val = ((d_lp / s_l) ** 2 + (d_cp / s_c) ** 2 + (d_big_hp / s_h) ** 2
           + r_t * (d_cp / s_c) * (d_big_hp / s_h))
    return np.sqrt(np.maximum(val, 0.0))


def calibrated_similarity(sim: Any, impostor: float, genuine: float) -> Any:
    """Map a similarity to 0..1: 0 at/below the impostor level, 1 at/above the genuine level."""
    span = max(genuine - impostor, 1e-6)
    return np.clip((np.asarray(sim, dtype=np.float64) - impostor) / span, 0.0, 1.0)


def accessory_presence(hits: np.ndarray, checks: np.ndarray,
                       config: Optional[IdentityConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(present, strongly_absent) boolean arrays from hit/check statistics (any leading shape)."""
    cfg = config or IdentityConfig()
    hits = np.asarray(hits, dtype=np.float64)
    checks = np.asarray(checks, dtype=np.float64)
    ratio = hits / np.maximum(checks, 1.0)
    present = (hits >= cfg.accessory_min_hits) & (ratio >= cfg.accessory_min_ratio)
    absent = (checks >= cfg.accessory_absent_min_checks) & (ratio < cfg.accessory_absent_max_ratio)
    return present, absent


def accessory_cue(track_hits: np.ndarray, track_checks: np.ndarray,
                  id_hits: np.ndarray, id_checks: np.ndarray,
                  config: Optional[IdentityConfig] = None) -> np.ndarray:
    """
    Accessory agreement of one track vs. M identities (NaN = cue unavailable).
    Only identity-stable accessories count: both present -> agreement; present in one side and
    strongly absent in the other -> disagreement; everything else is uninformative.
    """
    cfg = config or IdentityConfig()
    id_hits = np.atleast_2d(np.asarray(id_hits, dtype=np.float64))
    id_checks = np.atleast_2d(np.asarray(id_checks, dtype=np.float64))
    tp, ta = accessory_presence(track_hits, track_checks, cfg)
    ip, ia = accessory_presence(id_hits, id_checks, cfg)
    agree = (tp[None, :] & ip) & _STABLE_MASK[None, :]
    disagree = ((tp[None, :] & ia) | (ta[None, :] & ip)) & _STABLE_MASK[None, :]
    n_agree = agree.sum(axis=1).astype(np.float64)
    n_total = n_agree + disagree.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_total > 0, n_agree / np.maximum(n_total, 1.0), np.nan)


def color_cue(track_color: Optional[Dict[str, Any]], id_lab: np.ndarray, id_conf: np.ndarray,
              id_n: np.ndarray, id_names: Sequence[Optional[str]],
              config: Optional[IdentityConfig] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Colour agreement of one track colour consensus vs. M identity colours.
    Returns (score [NaN = unavailable], weight multiplier 0..1, veto mask).
    """
    cfg = config or IdentityConfig()
    m = len(id_conf)
    score = np.full(m, np.nan)
    weight = np.zeros(m)
    veto = np.zeros(m, dtype=bool)
    if track_color is None or m == 0:
        return score, weight, veto
    t_lab = track_color.get("lab")
    t_conf = float(track_color.get("conf", 0.0))
    t_n = int(track_color.get("n", 0))
    id_lab = np.asarray(id_lab, dtype=np.float64).reshape(-1, 3)
    id_conf = np.asarray(id_conf, dtype=np.float64)
    id_n = np.asarray(id_n, dtype=np.float64)

    if t_lab is not None:
        has = np.isfinite(id_lab[:, 0])
        if has.any():
            d_e = ciede2000_batch(t_lab, id_lab[has])
            score[has] = np.exp(-((d_e / max(cfg.color_scale, 1e-6)) ** 2))
            weight[has] = np.clip(np.sqrt(max(t_conf, 0.0) * id_conf[has]), 0.3, 1.0)
            veto[has] = ((d_e > cfg.color_veto_delta_e)
                         & (t_conf >= cfg.color_veto_confidence)
                         & (id_conf[has] >= cfg.color_veto_confidence)
                         & (t_n >= cfg.color_veto_min_samples)
                         & (id_n[has] >= cfg.color_veto_min_samples))

    t_name = track_color.get("name")
    if _known_name(t_name):
        for i in np.where(np.isnan(score))[0]:
            if _known_name(id_names[i]):
                score[i] = color_name_similarity(t_name, id_names[i])
                weight[i] = 0.6 * float(np.clip(math.sqrt(max(t_conf, 0.0) * max(id_conf[i], 0.0)), 0.3, 1.0))
    return score, weight, veto


def lower_garment_cue(track_color: Optional[Dict[str, Any]], track_type: int, track_type_conf: float,
                      id_lab: np.ndarray, id_conf: np.ndarray, id_n: np.ndarray,
                      id_names: Sequence[Optional[str]], id_type: np.ndarray, id_type_conf: np.ndarray,
                      config: Optional[IdentityConfig] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lower-garment agreement (type + pants colour) of one track vs. M identities.
    Occluded / unknown lower bodies (webcam upper-body views) make the cue unavailable.
    """
    cfg = config or IdentityConfig()
    id_type = np.asarray(id_type, dtype=np.int64)
    id_type_conf = np.asarray(id_type_conf, dtype=np.float64)
    m = len(id_type)
    score = np.full(m, np.nan)
    weight = np.zeros(m)
    veto = np.zeros(m, dtype=bool)
    if m == 0 or track_type == _TYPE_CODE[TYPE_OCCLUDED]:
        return score, weight, veto
    tc = max(float(track_type_conf), 0.0)
    type_w = np.clip(np.sqrt(tc * np.maximum(id_type_conf, 0.0)), 0.3, 1.0)

    if track_type == _TYPE_CODE[TYPE_PANTS]:
        both = id_type == _TYPE_CODE[TYPE_PANTS]
        if both.any() and track_color is not None:
            s, w, v = color_cue(track_color, np.asarray(id_lab)[both], np.asarray(id_conf)[both],
                                np.asarray(id_n)[both], [id_names[i] for i in np.where(both)[0]], cfg)
            score[both], weight[both], veto[both] = s, w, v
        other = id_type == _TYPE_CODE[TYPE_BARE]
    else:  # bare legs / shorts
        both = id_type == _TYPE_CODE[TYPE_BARE]
        score[both] = 0.85
        weight[both] = type_w[both]
        other = id_type == _TYPE_CODE[TYPE_PANTS]

    if other.any():
        confident = (tc >= cfg.garment_type_veto_confidence) & (id_type_conf >= cfg.garment_type_veto_confidence)
        veto |= other & confident
        soft = other & ~confident
        score[soft] = 0.25
        weight[soft] = 0.5 * type_w[soft]
    return score, weight, veto


def fuse_cues(s_app: np.ndarray, s_upper: np.ndarray, w_upper: np.ndarray, s_lower: np.ndarray,
              w_lower: np.ndarray, s_acc: np.ndarray,
              config: Optional[IdentityConfig] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted fusion of the cue scores (NaN = unavailable), weights renormalised per candidate.
    Returns (fused 0..1, appearance_available mask, number of available cues).
    """
    cfg = config or IdentityConfig()
    s_app = np.atleast_1d(np.asarray(s_app, dtype=np.float64))
    m = s_app.shape[0]
    num = np.zeros(m)
    den = np.zeros(m)
    n_cues = np.zeros(m, dtype=np.int64)
    cues = (
        (s_app, np.full(m, cfg.w_appearance)),
        (np.atleast_1d(np.asarray(s_upper, dtype=np.float64)), cfg.w_upper * np.atleast_1d(np.asarray(w_upper, dtype=np.float64))),
        (np.atleast_1d(np.asarray(s_lower, dtype=np.float64)), cfg.w_lower * np.atleast_1d(np.asarray(w_lower, dtype=np.float64))),
        (np.atleast_1d(np.asarray(s_acc, dtype=np.float64)), np.full(m, cfg.w_accessories)),
    )
    for s, w in cues:
        ok = np.isfinite(s) & (w > 0)
        num[ok] += w[ok] * s[ok]
        den[ok] += w[ok]
        n_cues[ok] += 1
    fused = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
    return fused, np.isfinite(s_app), n_cues


def identity_score(s_app: Optional[float], s_upper: Optional[float] = None, s_lower: Optional[float] = None,
                   s_acc: Optional[float] = None, upper_weight: float = 1.0, lower_weight: float = 1.0,
                   veto: bool = False, config: Optional[IdentityConfig] = None) -> float:
    """Scalar fused identity score (None = cue unavailable); 0 when a hard veto applies."""
    if veto:
        return 0.0

    def _v(x: Optional[float]) -> np.ndarray:
        return np.array([np.nan if x is None else float(x)])

    fused, _, _ = fuse_cues(_v(s_app), _v(s_upper), np.array([upper_weight]), _v(s_lower),
                            np.array([lower_weight]), _v(s_acc), config)
    return float(fused[0])


def estimate_observation_quality(bbox: Sequence[float], frame_shape: Sequence[int],
                                 keypoints_crop: Optional[np.ndarray] = None) -> float:
    """
    0..1 quality of one person observation for identity evidence:
    resolution (saturating at ~160 px height), truncation at the frame border (people entering or
    leaving), implausible aspect ratios (merged / partial boxes) and, when available, keypoint
    visibility of the head, shoulders and hips.
    """
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    except Exception:
        return 0.0
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    if w < 8 or h < 16:
        return 0.0

    size_q = math.sqrt(float(np.clip((h - 32.0) / (160.0 - 32.0), 0.05, 1.0)))

    margin = 2.0
    trunc = 1.0
    if x1 <= margin:
        trunc *= 0.7
    if x2 >= frame_w - margin:
        trunc *= 0.7
    if y1 <= margin:
        trunc *= 0.9
    if y2 >= frame_h - margin:
        trunc *= 0.85  # lower body cut off (common for webcams) - mild

    ratio = h / max(w, 1.0)
    if ratio < 0.9:
        aspect = float(np.clip(ratio / 0.9, 0.4, 1.0))
    elif ratio > 4.5:
        aspect = float(np.clip(4.5 / ratio, 0.5, 1.0))
    else:
        aspect = 1.0

    kp_q = 1.0
    if keypoints_crop is not None:
        try:
            kp = np.asarray(keypoints_crop, dtype=np.float64)
            if kp.ndim == 2 and kp.shape[0] >= 13:
                conf = kp[:, 2] if kp.shape[1] >= 3 else (np.abs(kp[:, :2]).sum(axis=1) > 0).astype(np.float64)
                key_idx = [0, 5, 6, 11, 12]  # nose, shoulders, hips
                visible = float(np.mean(conf[key_idx] > 0.3))
                kp_q = 0.55 + 0.45 * visible
        except Exception:
            kp_q = 1.0

    return float(np.clip(size_q * trunc * aspect * kp_q, 0.0, 1.0))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    idx = int(np.searchsorted(cum, 0.5 * cum[-1]))
    return float(v[min(idx, len(v) - 1)])


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    try:
        ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
        bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    except Exception:
        return 0.0
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = max(min(area_a, area_b), 1e-6)
    return max(inter / max(area_a + area_b - inter, 1e-6), 0.8 * inter / smaller)


def consensus_color(samples: Iterable[Tuple[Optional[np.ndarray], float, Optional[str], Optional[str], float]]
                    ) -> Optional[Dict[str, Any]]:
    """
    Quality x confidence weighted colour consensus of (lab, weight, name, hex, confidence) samples:
    per-channel weighted Lab median, confidence from accumulated weight and spread; name-only mode
    when no Lab value is available.
    """
    samples = [s for s in samples if s is not None and s[1] > 0]
    if not samples:
        return None
    lab_s = [s for s in samples if s[0] is not None]
    if lab_s:
        labs = np.stack([s[0] for s in lab_s]).astype(np.float64)
        w = np.array([s[1] for s in lab_s], dtype=np.float64)
        c = np.array([s[4] for s in lab_s], dtype=np.float64)
        med = np.array([_weighted_median(labs[:, k], w) for k in range(3)])
        total = float(w.sum())
        mean_conf = float((w * c).sum() / max(total, 1e-9))
        spread = _weighted_median(np.linalg.norm(labs - med[None, :], axis=1), w)
        dispersion = float(np.clip(1.0 - (spread - 8.0) / 30.0, 0.3, 1.0))
        conf = (total / (total + 1.5)) * mean_conf * dispersion
        name, hx = _classify_lab(med)
        if not _known_name(name):
            name, hx = lab_s[-1][2] or "Unknown", lab_s[-1][3] or "#808080"
        return {"lab": med, "conf": float(conf), "n": len(lab_s), "name": name, "hex": hx}

    # Name-only consensus (legacy producers without Lab values)
    votes: Dict[str, float] = {}
    hexes: Dict[str, str] = {}
    confs: Dict[str, float] = {}
    for _, w, name, hx, cf in samples:
        if not _known_name(name):
            continue
        votes[name] = votes.get(name, 0.0) + w
        hexes[name] = hx or hexes.get(name) or "#808080"
        confs[name] = confs.get(name, 0.0) + w * cf
    if not votes:
        return None
    best = max(votes, key=votes.get)
    total = sum(votes.values())
    share = votes[best] / max(total, 1e-9)
    conf = (total / (total + 1.5)) * (confs[best] / max(votes[best], 1e-9)) * share
    return {"lab": None, "conf": float(conf), "n": len(samples), "name": best, "hex": hexes[best]}


# =====================================================================================  internals
class _Calibrator:
    """Online impostor / genuine similarity statistics with priors and bounded updates."""

    def __init__(self, impostor_prior: float, genuine_prior: float, config: IdentityConfig,
                 impostor_std_prior: float = 0.05):
        self.cfg = config
        self.imp_prior = float(impostor_prior)
        self.gen_prior = float(genuine_prior)
        self.imp_std_prior = float(impostor_std_prior)
        self.reset()

    def reset(self) -> None:
        self.imp_mean, self.gen_mean = self.imp_prior, self.gen_prior
        self.imp_var = self.imp_std_prior ** 2
        self.gen_var = 0.05 ** 2
        self.n_imp = self.n_gen = 0

    def _step(self, mean: float, var: float, n: int, x: float, prior: float,
              max_up: Optional[float] = None) -> Tuple[float, float]:
        rate = max(1.0 / (n + 1 + self.cfg.calib_prior_weight), self.cfg.calib_min_rate)
        delta = x - mean
        up = self.cfg.calib_max_shift if max_up is None else max_up
        mean = float(np.clip(mean + rate * delta, prior - self.cfg.calib_max_shift, prior + up))
        var = float(np.clip(var + rate * (delta * delta - var), 0.02 ** 2, 0.15 ** 2))
        return mean, var

    def add_impostor(self, sim: float) -> None:
        if np.isfinite(sim):
            self.imp_mean, self.imp_var = self._step(self.imp_mean, self.imp_var, self.n_imp, float(sim), self.imp_prior)
            self.n_imp += 1

    def add_genuine(self, sim: float) -> None:
        if np.isfinite(sim):
            self.gen_mean, self.gen_var = self._step(self.gen_mean, self.gen_var, self.n_gen, float(sim),
                                                     self.gen_prior, self.cfg.calib_genuine_max_up)
            self.n_gen += 1

    def bounds(self, n_candidates: int = 1) -> Tuple[float, float]:
        """(lo, hi): similarity mapped to score 0 and to score 1 when choosing among n_candidates."""
        cfg = self.cfg
        imp_std = math.sqrt(self.imp_var)
        multi = cfg.calib_multi_k * math.sqrt(2.0 * math.log(max(int(n_candidates), 1)))
        lo = self.imp_mean + (cfg.calib_impostor_k + multi) * imp_std
        hi = min(self.gen_mean - cfg.calib_spread_k * math.sqrt(self.gen_var),
                 self.imp_mean + cfg.calib_impostor_full_k * imp_std)
        # Never lower the floor below the impostor tail: widen upwards instead
        hi = max(hi, lo + cfg.calib_min_gap)
        return lo, hi

    def score(self, sim: Any, n_candidates: int = 1) -> Any:
        lo, hi = self.bounds(n_candidates)
        return calibrated_similarity(sim, lo, hi)


class _Profile:
    """In-memory gallery identity: appearance prototypes, EMA mean and attribute profile."""

    def __init__(self, uid: str, model_id: str, dim: int):
        self.uid = uid
        self.model_id = model_id
        self.dim = dim
        self.protos = np.zeros((0, dim), dtype=np.float32)
        self.proto_q = np.zeros(0, dtype=np.float32)    # observation quality of each prototype
        self.mean: Optional[np.ndarray] = None
        self.n_mean = 0
        self.upper: Optional[Dict[str, Any]] = None     # {"lab","conf","n","name","hex"}
        self.lower: Optional[Dict[str, Any]] = None     # pants colour only
        self.lower_type = TYPE_OCCLUDED
        self.lower_type_conf = 0.0
        self.acc_hits = np.zeros(_N_ACC)
        self.acc_checks = np.zeros(_N_ACC)
        self.n_observations = 0
        self.best_crop_quality = 0.0
        self.last_seen_iso: Optional[str] = None
        self.first_seen_iso: Optional[str] = None
        self.last_seen_frame = -10 ** 9
        self.created_by_track: Optional[int] = None    # set when minted by a track this session
        self.sightings = 0                              # tracks assigned to it this session
        self.last_proto_frame = -10 ** 9
        self.dirty = False

    def appearance_ok(self, model_id: str) -> bool:
        return self.model_id == model_id and (self.mean is not None or self.protos.shape[0] > 0)

    def reset_appearance(self, model_id: str) -> None:
        self.model_id = model_id
        self.protos = np.zeros((0, self.dim), dtype=np.float32)
        self.proto_q = np.zeros(0, dtype=np.float32)
        self.mean = None
        self.n_mean = 0

    def n_good_protos(self, min_quality: float) -> int:
        return int((self.proto_q >= min_quality).sum())


class _Track:
    """Per-track evidence buffers and identity state."""

    def __init__(self, track_id: int, frame_idx: int, dim: int, cfg: IdentityConfig):
        self.track_id = track_id
        self.dim = dim
        self.cfg = cfg
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.postures: Deque[str] = deque(maxlen=15)
        self.last_obs: Optional[TrackObservation] = None
        self.uid: Optional[str] = None
        self.status = "tentative"
        self.decision_frame = -10 ** 9
        self.confidence = 0.0
        self.created_uid = False
        self.fresh(keep=0)

    def fresh(self, keep: int = 0) -> None:
        """Reset evidence buffers, optionally keeping the most recent `keep` observations."""
        cfg = self.cfg
        old_emb = list(getattr(self, "emb", []))[-keep:] if keep else []
        old_up = list(getattr(self, "upper_s", []))[-keep:] if keep else []
        old_lo = list(getattr(self, "lower_s", []))[-keep:] if keep else []
        old_acc = list(getattr(self, "acc_hist", []))[-keep:] if keep else []
        self.emb: Deque[Tuple[int, np.ndarray, float]] = deque(old_emb, maxlen=cfg.track_buffer)
        self.upper_s: Deque[Tuple] = deque(old_up, maxlen=cfg.color_buffer)
        self.lower_s: Deque[Tuple] = deque(old_lo, maxlen=cfg.color_buffer)
        self.acc_hist: Deque[np.ndarray] = deque(old_acc, maxlen=cfg.color_buffer)
        self.emb_sum = np.zeros(self.dim, dtype=np.float64)
        self.emb_w = 0.0
        for _, e, q in self.emb:
            self.emb_sum += q * e
            self.emb_w += q
        self.acc_hits = np.sum(old_acc, axis=0) if old_acc else np.zeros(_N_ACC)
        self.acc_checks = np.full(_N_ACC, float(len(old_acc)))
        self.acc_committed_hits = np.zeros(_N_ACC)
        self.acc_committed_checks = np.zeros(_N_ACC)
        self.n_obs = max(len(old_emb), len(old_up), len(old_lo))
        self.n_good = sum(1 for _, _, q in self.emb if q >= cfg.good_quality)
        self.n_usable = len(self.emb)
        self.obs_since_uid = 0
        self.merge_done: Set[int] = set()
        self.drift_strikes = 0
        self.last_drift_obs = 0
        self.last_gallery_obs = -10 ** 9
        self.last_genuine_frame = -10 ** 9
        self.last_gallery_calib_frame = -10 ** 9


class _GalleryCache:
    """Stacked, vectorisation-friendly view of a list of profiles."""

    def __init__(self, profiles: List[_Profile], model_id: str, dim: int):
        self.uids = [p.uid for p in profiles]
        m = len(profiles)
        self.size = m
        self.app_ok = np.array([p.appearance_ok(model_id) for p in profiles], dtype=bool)
        self.means = np.zeros((m, dim), dtype=np.float32)
        self.has_mean = np.zeros(m, dtype=bool)
        proto_blocks, owners = [], []
        self.n_protos = np.zeros(m, dtype=np.int64)
        for i, p in enumerate(profiles):
            if not self.app_ok[i]:
                continue
            if p.mean is not None:
                self.means[i] = p.mean
                self.has_mean[i] = True
            if p.protos.shape[0]:
                proto_blocks.append(p.protos)
                owners.append(np.full(p.protos.shape[0], i, dtype=np.int64))
                self.n_protos[i] = p.protos.shape[0]
        self.protos = np.concatenate(proto_blocks).astype(np.float32) if proto_blocks else np.zeros((0, dim), np.float32)
        self.owner = np.concatenate(owners) if owners else np.zeros(0, dtype=np.int64)
        starts = np.zeros(m, dtype=np.int64)
        if m:
            starts[1:] = np.cumsum(self.n_protos)[:-1]
        self.proto_start = starts

        nan3 = np.full(3, np.nan)
        self.upper_lab = np.stack([(p.upper["lab"] if p.upper and p.upper.get("lab") is not None else nan3) for p in profiles]) if m else np.zeros((0, 3))
        self.upper_conf = np.array([float(p.upper["conf"]) if p.upper else 0.0 for p in profiles])
        self.upper_n = np.array([float(p.upper["n"]) if p.upper else 0.0 for p in profiles])
        self.upper_names = [p.upper["name"] if p.upper else None for p in profiles]
        self.lower_lab = np.stack([(p.lower["lab"] if p.lower and p.lower.get("lab") is not None else nan3) for p in profiles]) if m else np.zeros((0, 3))
        self.lower_conf = np.array([float(p.lower["conf"]) if p.lower else 0.0 for p in profiles])
        self.lower_n = np.array([float(p.lower["n"]) if p.lower else 0.0 for p in profiles])
        self.lower_names = [p.lower["name"] if p.lower else None for p in profiles]
        self.type_code = np.array([_TYPE_CODE.get(p.lower_type, 2) for p in profiles], dtype=np.int64)
        self.type_conf = np.array([float(p.lower_type_conf) for p in profiles])
        self.acc_hits = np.stack([p.acc_hits for p in profiles]) if m else np.zeros((0, _N_ACC))
        self.acc_checks = np.stack([p.acc_checks for p in profiles]) if m else np.zeros((0, _N_ACC))


# =====================================================================================  manager
class IdentityManager:
    """Assigns persistent person UIDs to tracker tracks by fusing appearance and attribute evidence."""

    def __init__(self, db, embedding_model_id: str = "unknown", config: Optional[IdentityConfig] = None,
                 uid_prefix: str = "UID-"):
        self.db = db
        self.model_id = str(embedding_model_id or "unknown")
        self.config = config or IdentityConfig()
        self.uid_prefix = uid_prefix
        self.dim = int(getattr(db, "embedding_dim", 512) or 512)
        self._lock = threading.RLock()
        cfg = self.config
        # Fast re-acquire decides on very few frames: any confident colour / garment contradiction vetoes
        self._quick_config = replace(cfg, color_veto_min_samples=1, color_veto_confidence=0.2,
                                     garment_type_veto_confidence=0.3)
        self._cal_raw = _Calibrator(cfg.raw_impostor_prior, cfg.raw_genuine_prior, cfg, cfg.raw_impostor_std_prior)
        self._cal_cen = _Calibrator(cfg.centered_impostor_prior, cfg.centered_genuine_prior, cfg,
                                    cfg.centered_impostor_std_prior)
        self._init_state()
        self._load_gallery()

    # ------------------------------------------------------------------ lifecycle
    def _init_state(self) -> None:
        self._profiles: Dict[str, _Profile] = {}
        self._tracks: Dict[int, _Track] = {}
        self._events: List[Dict[str, Any]] = []
        self._session_uids: Set[str] = set()
        self._retired_uids: Set[str] = set()
        self._seq_floor = 1
        self._frame = 0
        self._last_flush_frame = 0
        self._imp_pair_last: Dict[Tuple[int, int], int] = {}
        self._rng = np.random.default_rng(12345)
        self._cache: Optional[_GalleryCache] = None
        self._cache_key: Optional[Tuple[int, int]] = None
        self._version = 0
        self._mu: Optional[np.ndarray] = None
        self._mu_wc = 0.0
        self._mu_frame = -10 ** 9
        self._mu_count = -1
        self._cal_raw.reset()
        self._cal_cen.reset()

    def _load_gallery(self) -> None:
        """Load persisted identity profiles (appearance usable only for the same embedding model)."""
        if self.db is None or not hasattr(self.db, "load_identity_profiles"):
            return
        try:
            rows = self.db.load_identity_profiles()
        except Exception as e:
            logger.warning(f"Could not load identity profiles: {e}")
            return
        decay = self.config.load_crop_quality_decay
        for row in rows:
            try:
                prof = self._profile_from_dict(row)
                prof.best_crop_quality *= decay
                self._profiles[prof.uid] = prof
            except Exception as e:
                logger.debug(f"Skipping identity profile {row.get('person_uid')}: {e}")
        self._version += 1
        n_ok = sum(1 for p in self._profiles.values() if p.appearance_ok(self.model_id))
        self._seed_impostors_from_gallery()
        logger.info(f"IdentityManager loaded {len(self._profiles)} identity profiles "
                    f"({n_ok} with '{self.model_id}' appearance)")

    def _seed_impostors_from_gallery(self) -> None:
        """Stored identities are different people: their mean similarities seed the impostor statistics,
        so a new session starts with camera/domain-specific impostor levels instead of bare priors."""
        cfg = self.config
        means = [p.mean for p in self._profiles.values() if p.mean is not None and p.model_id == self.model_id]
        if len(means) < 2:
            return
        mat = np.stack(means).astype(np.float32)
        n = mat.shape[0]
        iu, ju = np.triu_indices(n, 1)
        if iu.size > cfg.calib_seed_pairs:
            pick = self._rng.choice(iu.size, size=cfg.calib_seed_pairs, replace=False)
            iu, ju = iu[pick], ju[pick]
        # Raw space only (centered gallery-vs-gallery similarities are biased negative)
        for v in np.einsum("ij,ij->i", mat[iu], mat[ju]):
            self._cal_raw.add_impostor(float(v))

    def reset(self) -> None:
        """Forget all in-memory state (e.g. after the database was cleared) and reload the gallery."""
        with self._lock:
            self._init_state()
            self._load_gallery()

    @property
    def session_identity_count(self) -> int:
        """Distinct person uids assigned to tracks during this session."""
        with self._lock:
            return len(self._session_uids)

    @property
    def gallery_size(self) -> int:
        """Number of identities currently known (persisted + created this session)."""
        with self._lock:
            return len(self._profiles)

    def get_track_uid(self, track_id: int) -> Optional[str]:
        """UID currently assigned to a track (None while tentative / unknown)."""
        with self._lock:
            tr = self._tracks.get(int(track_id))
            return tr.uid if tr else None

    def get_identity_summary(self, uid: str) -> Optional[Dict[str, Any]]:
        """Attribute profile of an identity (colours, lower type, consensus accessories)."""
        with self._lock:
            p = self._profiles.get(uid)
            if p is None:
                return None
            return {
                "person_uid": uid,
                "upper": self._color_out(p.upper),
                "lower": self._lower_out(p.lower, p.lower_type, p.lower_type_conf, None),
                "accessories": self._present_list(p.acc_hits, p.acc_checks),
                "n_observations": int(p.n_observations),
                "best_crop_quality": float(p.best_crop_quality),
                "prototypes": int(p.protos.shape[0]),
                "embedding_model": p.model_id,
            }

    # ------------------------------------------------------------------ main entry point
    def update(self, observations: List[TrackObservation], frame_idx: int) -> Dict[int, IdentityAssignment]:
        """Ingest this frame's observations and return the identity assignment of every observed track."""
        with self._lock:
            cfg = self.config
            self._frame = int(frame_idx)
            obs_by_track: Dict[int, TrackObservation] = {}
            for o in observations or []:
                try:
                    obs_by_track[int(o.track_id)] = o
                except Exception:
                    continue

            observed: List[_Track] = []
            for tid, o in obs_by_track.items():
                tr = self._tracks.get(tid)
                if tr is None:
                    tr = _Track(tid, self._frame, self.dim, cfg)
                    self._tracks[tid] = tr
                self._ingest(tr, o)
                observed.append(tr)

            self._collect_calibration(observed)

            merged_from: Dict[int, str] = {}
            for tr in observed:
                if tr.uid is None:
                    continue
                tr.obs_since_uid += 1
                self._check_drift(tr)
                if tr.uid is not None and tr.created_uid and tr.obs_since_uid in cfg.merge_checkpoints:
                    src = self._check_late_merge(tr)
                    if src:
                        merged_from[tr.track_id] = src
                if tr.uid is not None:
                    self._maybe_update_gallery(tr, obs_by_track[tr.track_id])

            ready = [tr for tr in observed if tr.uid is None and
                     (tr.n_good >= min(cfg.min_good_observations, cfg.quick_min_good)
                      or tr.n_obs >= cfg.max_tentative_observations)]
            if ready:
                self._decide(ready, observed)

            out: Dict[int, IdentityAssignment] = {}
            for tr in observed:
                out[tr.track_id] = self._assignment_for(tr, obs_by_track[tr.track_id], merged_from.get(tr.track_id))

            # Auto-timeout of tracks the tracker stopped reporting
            stale = [tid for tid, tr in self._tracks.items()
                     if self._frame - tr.last_frame > cfg.track_timeout_frames]
            if stale:
                self._end_tracks_locked(stale)

            if self._frame - self._last_flush_frame >= cfg.flush_every_frames:
                self._flush_locked()
            return out

    def end_tracks(self, track_ids: Iterable[int]) -> None:
        """Tracker archived these tracks: commit their evidence, release their uids, persist profiles."""
        with self._lock:
            self._end_tracks_locked(list(track_ids or []))
            self._flush_locked()

    def get_events(self) -> List[Dict[str, Any]]:
        """Drain identity events: {"type": "new"|"reacquired"|"merged", "person_uid", "track_id",
        "confidence", "frame_idx"} (+ "merged_from" for merges)."""
        with self._lock:
            events, self._events = self._events, []
            return events

    def flush(self) -> None:
        """Persist every dirty identity profile."""
        with self._lock:
            self._flush_locked()

    # ------------------------------------------------------------------ ingestion
    def _ingest(self, tr: _Track, o: TrackObservation) -> None:
        cfg = self.config
        q = float(np.clip(o.quality if o.quality is not None else 0.0, 0.0, 1.0))
        tr.n_obs += 1
        tr.last_frame = self._frame
        tr.last_bbox = tuple(o.bbox) if o.bbox is not None else tr.last_bbox
        tr.last_obs = o
        tr.postures.append(o.posture or "Standing")
        usable = q >= cfg.min_quality
        if q >= cfg.good_quality:
            tr.n_good += 1

        if usable:
            tr.n_usable += 1
            e = self._valid_embedding(o.embedding)
            if e is not None:
                tr.emb.append((self._frame, e, q))
                tr.emb_sum += q * e
                tr.emb_w += q

            up = o.upper or {}
            lab = _as_lab(up.get("lab"))
            conf = float(up.get("confidence", 0.6 if lab is not None else 0.4) or 0.0)
            name = up.get("color")
            if conf >= cfg.min_color_confidence and (lab is not None or _known_name(name)):
                tr.upper_s.append((lab, q * conf, name, up.get("hex"), conf))

            lo = o.lower or {}
            ltype = self._lower_type_of(lo)
            l_lab = _as_lab(lo.get("lab")) if ltype == TYPE_PANTS else None
            l_conf = float(lo.get("confidence", 0.6 if l_lab is not None else 0.4) or 0.0)
            tr.lower_s.append((ltype, l_lab, q * l_conf, lo.get("color"), lo.get("hex"), l_conf, q))

        if o.accessory_checked and usable:
            vec = np.zeros(_N_ACC)
            accs = o.accessories or {}
            items = accs.items() if isinstance(accs, dict) else ((a, 1.0) for a in accs)
            for name, c in items:
                idx = _ACC_INDEX.get(normalize_accessory(str(name)))
                try:
                    c = float(c)
                except Exception:
                    c = 1.0
                if idx is not None and c >= cfg.accessory_min_confidence:
                    vec[idx] = 1.0
            tr.acc_hits += vec
            tr.acc_checks += 1.0
            tr.acc_hist.append(vec)

    def _valid_embedding(self, emb: Any) -> Optional[np.ndarray]:
        if emb is None:
            return None
        try:
            arr = np.asarray(emb, dtype=np.float32).ravel()
        except Exception:
            return None
        if arr.size != self.dim:
            return None
        return _unit(arr)

    @staticmethod
    def _lower_type_of(lower: Dict[str, Any]) -> str:
        t = lower.get("type")
        if t in _TYPE_CODE:
            return t
        color = lower.get("color")
        if color in _TYPE_CODE:
            return color
        if _known_name(color):
            low = str(color).lower()
            if "bare" in low or "short" in low:
                return TYPE_BARE
            if "occluded" in low:
                return TYPE_OCCLUDED
            return TYPE_PANTS
        return TYPE_OCCLUDED

    # ------------------------------------------------------------------ evidence
    def _evidence(self, tr: _Track, window: Optional[int] = None, accumulated: bool = False,
                  need_desc: bool = True) -> Dict[str, Any]:
        """Track evidence for scoring: descriptor, colour consensus, garment type, accessory stats."""
        cfg = self.config
        emb = list(tr.emb)[-window:] if window else list(tr.emb)
        if not need_desc:
            desc = None
        elif accumulated and tr.emb_w > 0:
            desc = _unit(tr.emb_sum)
        elif emb:
            mat = np.stack([e for _, e, _ in emb])
            w = np.array([q for _, _, q in emb], dtype=np.float32)
            desc = _unit((w[:, None] * mat).sum(axis=0))
        else:
            desc = None
        ups = list(tr.upper_s)[-window:] if window else list(tr.upper_s)
        los = list(tr.lower_s)[-window:] if window else list(tr.lower_s)
        upper = consensus_color(ups)
        lower = consensus_color((s[1], s[2], s[3], s[4], s[5]) for s in los if s[0] == TYPE_PANTS)
        ltype, ltype_conf = self._type_consensus(los)
        if window:
            hist = list(tr.acc_hist)[-window:]
            hits = np.sum(hist, axis=0) if hist else np.zeros(_N_ACC)
            checks = np.full(_N_ACC, float(len(hist)))
        else:
            hits, checks = tr.acc_hits, tr.acc_checks
        return {
            "desc": desc,
            "upper": upper,
            "lower": lower,
            "type": _TYPE_CODE[ltype],
            "type_name": ltype,
            "type_conf": ltype_conf,
            "acc_hits": hits,
            "acc_checks": checks,
            "n_good": sum(1 for _, _, q in emb if q >= cfg.good_quality),
        }

    @staticmethod
    def _type_consensus(lower_samples: Sequence[Tuple]) -> Tuple[str, float]:
        """Lower garment type vote; occluded votes only dilute (absence of evidence)."""
        w_p = sum(s[6] for s in lower_samples if s[0] == TYPE_PANTS)
        w_b = sum(s[6] for s in lower_samples if s[0] == TYPE_BARE)
        w_o = sum(s[6] for s in lower_samples if s[0] == TYPE_OCCLUDED)
        visible = w_p + w_b
        if visible <= 0 or visible < 0.25 * (visible + w_o):
            return TYPE_OCCLUDED, 0.0
        winner = TYPE_PANTS if w_p >= w_b else TYPE_BARE
        share = max(w_p, w_b) / visible
        conf = share * (visible / (visible + 2.0))
        return winner, float(conf)

    # ------------------------------------------------------------------ appearance space
    def _population_mean(self) -> Tuple[np.ndarray, float]:
        """Population mean of identity means (+ visible tentative tracks) and the centering weight."""
        cfg = self.config
        n_prof = len(self._profiles)
        if (self._mu is not None and self._frame - self._mu_frame < cfg.center_refresh_frames
                and n_prof == self._mu_count):
            return self._mu, self._mu_wc
        vecs = [p.mean for p in self._profiles.values() if p.mean is not None and p.model_id == self.model_id]
        for tr in self._tracks.values():
            if tr.uid is None and self._frame - tr.last_frame <= 30 and len(tr.emb) >= 3:
                d = _unit(tr.emb_sum) if tr.emb_w > 0 else None
                if d is not None:
                    vecs.append(d)
        if vecs:
            mu = np.mean(np.stack(vecs), axis=0).astype(np.float32)
        else:
            mu = np.zeros(self.dim, dtype=np.float32)
        n = len(vecs)
        wc = float(np.clip((n - 2.0) / max(cfg.center_full_identities - 2.0, 1.0), 0.0, 1.0))
        self._mu, self._mu_wc, self._mu_frame, self._mu_count = mu, wc, self._frame, n_prof
        return mu, wc

    @staticmethod
    def _centered_cos(raw_dots: np.ndarray, x_mu: np.ndarray, d_mu: float, mu_mu: float, x_norm2: Any = 1.0) -> np.ndarray:
        """cos(x - mu, d - mu) from raw dots (x.d), x.mu, d.mu, mu.mu (x, d unit unless x_norm2 given)."""
        nx = np.sqrt(np.maximum(x_norm2 - 2.0 * x_mu + mu_mu, 1e-9))
        nd = math.sqrt(max(1.0 - 2.0 * d_mu + mu_mu, 1e-9))
        return (raw_dots - x_mu - d_mu + mu_mu) / (nx * nd)

    def _appearance_scores(self, desc: Optional[np.ndarray], cache: _GalleryCache,
                           n_candidates: Optional[int] = None) -> np.ndarray:
        """Calibrated appearance score of a descriptor vs. every cached identity (NaN = unavailable)."""
        cfg = self.config
        m = cache.size
        out = np.full(m, np.nan)
        if desc is None or m == 0 or not cache.app_ok.any():
            return out
        mu, wc = self._population_mean()
        mu_mu = float(mu @ mu)
        d_mu = float(desc @ mu)

        mean_raw = cache.means @ desc
        mean_cen = self._centered_cos(mean_raw, cache.means @ mu, d_mu, mu_mu)
        max_raw = np.full(m, -np.inf)
        max_cen = np.full(m, -np.inf)
        if cache.protos.shape[0]:
            p_raw = cache.protos @ desc
            p_cen = self._centered_cos(p_raw, cache.protos @ mu, d_mu, mu_mu)
            has_p = cache.n_protos > 0
            starts = cache.proto_start[has_p]
            max_raw[has_p] = np.maximum.reduceat(p_raw, starts)
            max_cen[has_p] = np.maximum.reduceat(p_cen, starts)
        has_p = cache.n_protos > 0
        pw = cfg.proto_weight
        both = has_p & cache.has_mean
        sim_raw = np.where(both, pw * max_raw + (1 - pw) * mean_raw, np.where(has_p, max_raw, mean_raw))
        sim_cen = np.where(both, pw * max_cen + (1 - pw) * mean_cen, np.where(has_p, max_cen, mean_cen))
        n_cand = int(cache.app_ok.sum()) if n_candidates is None else int(n_candidates)
        score = (1.0 - wc) * self._cal_raw.score(sim_raw, n_cand)
        if wc > 0:
            score = score + wc * self._cal_cen.score(sim_cen, n_cand)
        out[cache.app_ok] = score[cache.app_ok]
        return out

    def _score(self, ev: Dict[str, Any], cache: _GalleryCache,
               quick: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fused score of one track's evidence vs. every cached identity.
        Returns (adjusted score [-inf = impossible], fused 0..1, veto mask). The adjusted score is
        shifted so that every candidate is compared with `match_threshold` (stricter threshold for
        candidates without the appearance cue). `quick` uses hair-trigger vetoes (few samples).
        """
        cfg = self._quick_config if quick else self.config
        m = cache.size
        if m == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool)
        s_u, w_u, veto_u = color_cue(ev["upper"], cache.upper_lab, cache.upper_conf, cache.upper_n,
                                     cache.upper_names, cfg)
        s_l, w_l, veto_l = lower_garment_cue(ev["lower"], ev["type"], ev["type_conf"], cache.lower_lab,
                                             cache.lower_conf, cache.lower_n, cache.lower_names,
                                             cache.type_code, cache.type_conf, cfg)
        s_a = accessory_cue(ev["acc_hits"], ev["acc_checks"], cache.acc_hits, cache.acc_checks, cfg)
        veto = veto_u | veto_l
        # Open-set correction counts only identities that compete on appearance (attribute-compatible)
        plausible = cache.app_ok & ~veto
        for s_cue in (s_u, s_l, s_a):
            plausible &= ~(np.isfinite(s_cue) & (s_cue < 0.5))
        s_app = self._appearance_scores(ev["desc"], cache, n_candidates=max(int(plausible.sum()), 1))
        fused, has_app, n_cues = fuse_cues(s_app, s_u, w_u, s_l, w_l, s_a, cfg)
        fused = np.where(veto, 0.0, fused)
        thr = np.where(has_app, cfg.match_threshold, cfg.no_appearance_threshold)
        adj = fused - thr + cfg.match_threshold
        impossible = veto | (~has_app & (n_cues < cfg.no_appearance_min_cues)) | (n_cues == 0)
        adj = np.where(impossible, -np.inf, adj)
        return adj, fused, veto

    def _gallery_cache(self) -> _GalleryCache:
        key = (self._version, self._frame)
        if self._cache is None or self._cache_key != key:
            self._cache = _GalleryCache(list(self._profiles.values()), self.model_id, self.dim)
            self._cache_key = key
        return self._cache

    def _touch(self, prof: _Profile) -> None:
        prof.dirty = True
        self._version += 1

    # ------------------------------------------------------------------ calibration
    def _collect_calibration(self, observed: List[_Track]) -> None:
        """Genuine pairs: same track >= genuine_min_gap frames apart. Impostors: concurrent distinct tracks."""
        cfg = self.config
        if not observed:
            return
        mu, wc = self._population_mean()
        mu_mu = float(mu @ mu)

        def _cen(a: np.ndarray, b: np.ndarray) -> float:
            return float(self._centered_cos(np.array([a @ b]), np.array([a @ mu]), float(b @ mu), mu_mu)[0])

        for tr in observed:
            if tr.drift_strikes or len(tr.emb) < 6 or self._frame - tr.last_genuine_frame < cfg.calib_pair_interval:
                continue
            items = list(tr.emb)
            last_f = items[-1][0]
            old = [e for f, e, _ in items if f <= last_f - cfg.genuine_min_gap][-3:]
            if len(old) < 3:
                continue
            a = _unit(np.mean([e for _, e, _ in items[-3:]], axis=0))
            b = _unit(np.mean(old, axis=0))
            if a is None or b is None:
                continue
            self._cal_raw.add_genuine(float(a @ b))
            if wc >= 0.5:
                self._cal_cen.add_genuine(_cen(a, b))
            tr.last_genuine_frame = self._frame

        # Impostors from the gallery: a confirmed track is a different person from every other identity
        others = None
        for tr in observed:
            if tr.uid is None or tr.drift_strikes or len(tr.emb) < 5 or tr.obs_since_uid < cfg.drift_window:
                continue
            if self._frame - tr.last_gallery_calib_frame < cfg.calib_gallery_interval:
                continue
            tr.last_gallery_calib_frame = self._frame
            if others is None:
                held_now = {t.uid for t in observed if t.uid}
                others = [p for p in self._profiles.values()
                          if p.mean is not None and p.model_id == self.model_id and p.uid not in held_now]
            if not others:
                break
            d = _unit(np.mean([e for _, e, _ in list(tr.emb)[-5:]], axis=0))
            if d is None:
                continue
            pick = self._rng.choice(len(others), size=min(cfg.calib_gallery_samples, len(others)), replace=False)
            for i in pick:
                # Raw space only: centered similarities between gallery members are biased negative
                # (they sum to zero around their own mean) and would make new people look too familiar
                self._cal_raw.add_impostor(float(d @ others[int(i)].mean))

        n_pairs = 0
        cands = [tr for tr in observed if len(tr.emb) >= 3]
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                ta, tb = cands[i], cands[j]
                key = (min(ta.track_id, tb.track_id), max(ta.track_id, tb.track_id))
                if self._frame - self._imp_pair_last.get(key, -10 ** 9) < cfg.calib_pair_interval:
                    continue
                if ta.uid is not None and ta.uid == tb.uid:
                    continue
                if _bbox_iou(ta.last_bbox, tb.last_bbox) > 0.3:
                    continue  # possibly a duplicate detection of one person
                a = _unit(np.mean([e for _, e, _ in list(ta.emb)[-5:]], axis=0))
                b = _unit(np.mean([e for _, e, _ in list(tb.emb)[-5:]], axis=0))
                if a is None or b is None:
                    continue
                self._cal_raw.add_impostor(float(a @ b))
                if wc >= 0.5:
                    self._cal_cen.add_impostor(_cen(a, b))
                self._imp_pair_last[key] = self._frame
                n_pairs += 1
                if n_pairs >= 10:
                    return

    # ------------------------------------------------------------------ decisions
    def _held_uids(self, exclude_tracks: Set[int]) -> Dict[str, _Track]:
        """uids held by tracks observed within the exclusivity window (excluding given tracks)."""
        held = {}
        for tid, tr in self._tracks.items():
            if tid in exclude_tracks or tr.uid is None:
                continue
            if self._frame - tr.last_frame <= self.config.exclusivity_frames:
                held[tr.uid] = tr
        return held

    def _decide(self, ready: List[_Track], observed: List[_Track]) -> None:
        """Jointly resolve ready tentative tracks: match an existing identity or mint a new uid."""
        cfg = self.config
        ready_ids = {tr.track_id for tr in ready}
        held = self._held_uids(ready_ids)
        cache = self._gallery_cache()
        m = cache.size
        t = len(ready)
        scores = np.full((t, m), -np.inf)
        best_any = np.full(t, -np.inf)
        # A uid held by another visible track is never a candidate (unless that track is drifting)
        blocked = np.array([uid in held and held[uid].drift_strikes == 0 for uid in cache.uids], dtype=bool)
        evs = []
        for r, tr in enumerate(ready):
            ev = self._evidence(tr)
            evs.append(ev)
            if m == 0:
                continue
            quick = tr.n_good < cfg.min_good_observations and tr.n_obs < cfg.max_tentative_observations
            adj, _, _ = self._score(ev, cache, quick=quick)
            adj[blocked] = -np.inf
            scores[r] = adj
            best_any[r] = adj.max()

        assigned_col = np.full(t, -1, dtype=np.int64)
        if m > 0 and np.isfinite(scores).any():
            cost = -np.where(np.isfinite(scores), scores, _NEG)
            if SCIPY_AVAILABLE:
                rows, cols = linear_sum_assignment(cost)
            else:  # greedy fallback
                rows, cols, used = [], [], set()
                for r in np.argsort(-best_any):
                    order = np.argsort(-scores[r])
                    for c in order:
                        if c not in used and np.isfinite(scores[r, c]):
                            rows.append(r)
                            cols.append(c)
                            used.add(c)
                            break
            for r, c in zip(rows, cols):
                if np.isfinite(scores[r, c]):
                    assigned_col[r] = c

        for r, tr in enumerate(ready):
            forced = tr.n_obs >= cfg.max_tentative_observations
            full = forced or tr.n_good >= cfg.min_good_observations
            c = assigned_col[r]
            if c >= 0:
                sc = float(scores[r, c])
                others = np.delete(scores[r], c)
                runner = float(others.max()) if others.size else -np.inf
                if not full:
                    # Fast path: an identity lost moments ago, matched unambiguously
                    prof = self._profiles.get(cache.uids[c])
                    recent = prof is not None and self._frame - prof.last_seen_frame <= cfg.quick_recent_frames
                    if recent and sc >= cfg.quick_threshold and sc - runner >= cfg.quick_margin:
                        self._assign(tr, cache.uids[c], "reacquired", float(np.clip(sc, 0.0, 1.0)))
                    continue
                margin_ok = (sc - runner) >= cfg.match_margin
                if sc >= cfg.match_threshold and not margin_ok and np.isfinite(runner):
                    # Ambiguity between two identities of (apparently) the same person is harmless
                    j = int(np.argmax(np.where(np.arange(m) == c, -np.inf, scores[r])))
                    margin_ok = self._looks_duplicate(cache.uids[c], cache.uids[j])
                if sc >= cfg.match_threshold and (margin_ok or (forced and sc >= cfg.match_threshold + cfg.forced_margin_bonus)):
                    self._assign(tr, cache.uids[c], "reacquired", float(np.clip(sc, 0.0, 1.0)))
                    continue
            if not full:
                continue
            best = float(best_any[r])
            if best < cfg.clear_new_score or forced:
                if forced and tr.n_usable < cfg.min_usable_for_new:
                    continue  # not enough usable evidence yet - keep identifying
                self._create_identity(tr, evs[r], best)

    def _looks_duplicate(self, uid_a: str, uid_b: str) -> bool:
        """True when two identities agree in appearance (verification score) and are not colour-vetoed."""
        cfg = self.config
        pa, pb = self._profiles.get(uid_a), self._profiles.get(uid_b)
        if pa is None or pb is None or pa.mean is None or pb.mean is None:
            return False
        if not (pa.appearance_ok(self.model_id) and pb.appearance_ok(self.model_id)):
            return False
        s = self._appearance_scores(pa.mean, _GalleryCache([pb], self.model_id, self.dim), n_candidates=1)
        if not (s.size and np.isfinite(s[0]) and s[0] >= cfg.duplicate_pair_score):
            return False
        if pa.upper and pb.upper and pa.upper.get("lab") is not None and pb.upper.get("lab") is not None:
            if float(ciede2000_batch(pa.upper["lab"], np.asarray(pb.upper["lab"]).reshape(1, 3))[0]) > cfg.color_veto_delta_e:
                return False
        return True

    def _assign(self, tr: _Track, uid: str, status: str, score: float) -> None:
        """Give `uid` to track `tr` (any stale holder loses it) and record the event."""
        for other in self._tracks.values():
            if other is not tr and other.uid == uid:
                logger.info(f"Track #{other.track_id} lost {uid} to track #{tr.track_id}")
                self._detach(other, keep=0)
        prof = self._profiles[uid]
        tr.uid = uid
        tr.status = status
        tr.decision_frame = self._frame
        tr.confidence = float(score)
        tr.created_uid = status == "new"
        tr.obs_since_uid = 0
        tr.merge_done = set()
        tr.drift_strikes = 0
        tr.last_drift_obs = 0
        tr.last_gallery_obs = -10 ** 9
        tr.acc_committed_hits = np.zeros(_N_ACC)
        tr.acc_committed_checks = np.zeros(_N_ACC)
        prof.sightings += 1
        prof.last_seen_frame = self._frame
        prof.last_seen_iso = _now_iso()
        self._session_uids.add(uid)
        self._absorb_track(tr, prof, initial=True)
        self._events.append({"type": status, "person_uid": uid, "track_id": tr.track_id,
                             "confidence": float(score), "frame_idx": self._frame})
        logger.info(f"Track #{tr.track_id} -> {uid} ({status}, score {score:.2f})")

    def _new_uid(self) -> str:
        seq = 1
        if self.db is not None and hasattr(self.db, "next_person_sequence"):
            try:
                seq = int(self.db.next_person_sequence())
            except Exception as e:
                logger.debug(f"next_person_sequence failed: {e}")
        seq = max(seq, self._seq_floor)
        while True:
            uid = f"{self.uid_prefix}{seq:04d}"
            if uid not in self._profiles and uid not in self._retired_uids:
                break
            seq += 1
        self._seq_floor = seq + 1
        return uid

    def _create_identity(self, tr: _Track, ev: Dict[str, Any], best_other: float) -> None:
        uid = self._new_uid()
        prof = _Profile(uid, self.model_id, self.dim)
        prof.created_by_track = tr.track_id
        prof.first_seen_iso = _now_iso()
        self._profiles[uid] = prof
        self._touch(prof)
        conf = 1.0 if not np.isfinite(best_other) else float(np.clip(1.0 - max(best_other, 0.0), 0.5, 1.0))
        self._assign(tr, uid, "new", conf)

    def _detach(self, tr: _Track, keep: int = 0) -> None:
        """Release the track's uid and go back to the tentative phase with a fresh evidence buffer."""
        if tr.uid is not None:
            self._commit_accessories(tr, self._profiles.get(tr.uid))
        tr.uid = None
        tr.status = "tentative"
        tr.created_uid = False
        tr.confidence = 0.0
        tr.fresh(keep=keep)

    # ------------------------------------------------------------------ gallery maintenance
    def _absorb_track(self, tr: _Track, prof: _Profile, initial: bool = False) -> None:
        """Seed / refresh an identity from the track buffer (prototypes, colours, accessories)."""
        cfg = self.config
        if prof.model_id != self.model_id:
            # Profile from another embedding model: appearance restarts in the current space
            prof.reset_appearance(self.model_id)
        if initial and tr.emb:
            items = sorted(tr.emb, key=lambda it: -it[2])
            qmax = items[0][2]
            pool = [(e, q) for _, e, q in items if q >= min(cfg.gallery_min_quality, qmax)]
            if prof.mean is None or prof.protos.shape[0] == 0:
                desc = self._evidence(tr)["desc"]
                if desc is not None and prof.mean is None:
                    prof.mean = desc
                    # Poor seeds get a small pseudo-count so good observations take over quickly
                    prof.n_mean = max(1, sum(1 for _, _, q in tr.emb if q >= cfg.gallery_min_quality))
                for e, q in self._diverse_subset(pool, 4):
                    self._add_prototype(prof, e, q, force=True)
            elif pool:
                self._add_prototype(prof, pool[0][0], pool[0][1], force=False)
        self._update_attributes(tr, prof)
        self._commit_accessories(tr, prof)
        if initial:
            prof.n_observations += tr.n_usable
        self._touch(prof)

    @staticmethod
    def _diverse_subset(pool: List[Tuple[np.ndarray, float]], k: int) -> List[Tuple[np.ndarray, float]]:
        """Greedy farthest-point subset of (embedding, quality) pairs (first element = best quality)."""
        if not pool:
            return []
        chosen = [pool[0]]
        mat = np.stack([e for e, _ in pool])
        best_sim = mat @ pool[0][0]
        while len(chosen) < min(k, len(pool)):
            idx = int(np.argmin(best_sim))
            if best_sim[idx] > 0.995:
                break
            chosen.append(pool[idx])
            best_sim = np.maximum(best_sim, mat @ pool[idx][0])
        return chosen

    def _proto_sims(self, e: np.ndarray, protos: np.ndarray) -> np.ndarray:
        """Blended (raw / centered) similarity of e to prototypes, for novelty and redundancy."""
        mu, wc = self._population_mean()
        raw = protos @ e
        if wc <= 0:
            return raw
        mu_mu = float(mu @ mu)
        cen = self._centered_cos(raw, protos @ mu, float(e @ mu), mu_mu)
        return (1 - wc) * raw + wc * cen

    def _novelty_threshold(self) -> float:
        _, wc = self._population_mean()
        cfg = self.config
        return (1 - wc) * cfg.proto_novelty_raw + wc * cfg.proto_novelty_centered

    def _add_prototype(self, prof: _Profile, e: np.ndarray, q: float, force: bool = False) -> bool:
        """Add a diverse prototype (or replace a poor / redundant one). Returns True when changed."""
        cfg = self.config
        e = e.reshape(1, -1).astype(np.float32)
        if prof.protos.shape[0] == 0:
            prof.protos, prof.proto_q = e, np.array([q], dtype=np.float32)
            prof.last_proto_frame = self._frame
            return True
        # Prototypes seeded from poor observations are replaced first by good ones
        poor = prof.proto_q < cfg.gallery_min_quality
        if q >= cfg.gallery_min_quality and poor.any():
            idx = int(np.argmin(prof.proto_q))
            prof.protos[idx], prof.proto_q[idx] = e[0], q
            prof.last_proto_frame = self._frame
            return True
        if not force and self._frame - prof.last_proto_frame < cfg.proto_min_interval:
            return False
        sims = self._proto_sims(e[0], prof.protos)
        if float(sims.max()) >= self._novelty_threshold():
            return False
        if prof.protos.shape[0] < cfg.max_prototypes:
            prof.protos = np.vstack([prof.protos, e]).astype(np.float32)
            prof.proto_q = np.append(prof.proto_q, np.float32(q))
        else:
            # Replace the most redundant prototype when the newcomer is less redundant
            gram = prof.protos @ prof.protos.T
            np.fill_diagonal(gram, -1.0)
            redundancy = gram.max(axis=1)
            worst = int(np.argmax(redundancy))
            if redundancy[worst] <= float((prof.protos @ e[0]).max()):
                return False
            prof.protos[worst], prof.proto_q[worst] = e[0], q
        prof.last_proto_frame = self._frame
        return True

    def _prune_prototypes(self, prof: _Profile) -> None:
        """Drop the most redundant prototypes until the cap is respected."""
        cap = self.config.max_prototypes
        while prof.protos.shape[0] > cap:
            gram = prof.protos @ prof.protos.T
            np.fill_diagonal(gram, -1.0)
            worst = int(np.argmax(gram.max(axis=1)))
            prof.protos = np.delete(prof.protos, worst, axis=0)
            prof.proto_q = np.delete(prof.proto_q, worst)

    def _is_mature(self, prof: _Profile) -> bool:
        """Enough good-quality appearance evidence to judge consistency / drift against."""
        return prof.n_good_protos(self.config.gallery_min_quality) >= 3 and prof.n_mean >= 5

    def _single_appearance(self, e: np.ndarray, prof: _Profile) -> float:
        cache = _GalleryCache([prof], self.model_id, self.dim)
        s = self._appearance_scores(e, cache)
        return float(s[0]) if s.size and np.isfinite(s[0]) else float("nan")

    def _maybe_update_gallery(self, tr: _Track, o: TrackObservation) -> None:
        """Refine the track's identity from a good, consistent observation."""
        cfg = self.config
        prof = self._profiles.get(tr.uid)
        if prof is None:
            return
        prof.last_seen_frame = self._frame
        if tr.n_obs - tr.last_gallery_obs < cfg.gallery_update_every:
            return
        q = float(o.quality or 0.0)
        if q < cfg.gallery_min_quality or tr.drift_strikes > 0:
            return
        tr.last_gallery_obs = tr.n_obs
        prof.last_seen_iso = _now_iso()
        e = self._valid_embedding(o.embedding)
        if e is not None:
            if prof.model_id != self.model_id:
                prof.reset_appearance(self.model_id)
            consistent = True
            if self._is_mature(prof):
                s = self._single_appearance(e, prof)
                consistent = not np.isfinite(s) or s >= cfg.gallery_consistency
            if not consistent:
                return  # inconsistent appearance: do not let it leak into the gallery
            rate = max(cfg.mean_ema, 1.0 / (prof.n_mean + 1.0))
            base = prof.mean if prof.mean is not None else e
            prof.mean = _unit((1.0 - rate) * base + rate * e)
            prof.n_mean += 1
            self._add_prototype(prof, e, q)
        self._update_attributes(tr, prof)
        self._commit_accessories(tr, prof)
        prof.n_observations += cfg.gallery_update_every
        self._touch(prof)

    def _update_attributes(self, tr: _Track, prof: _Profile) -> None:
        """Blend the track's colour / garment-type consensus into the identity profile."""
        cfg = self.config
        ev = self._evidence(tr, need_desc=False)
        r = cfg.color_ema
        for key in ("upper", "lower"):
            tc = ev[key]
            if tc is None or (key == "lower" and ev["type_name"] != TYPE_PANTS):
                continue  # pants colour only when the track's lower garment really is pants
            pc = getattr(prof, key)
            if pc is None or (pc.get("lab") is None and (tc.get("lab") is not None or _known_name(tc.get("name")))):
                setattr(prof, key, dict(tc))
                continue
            if tc.get("lab") is None or pc.get("lab") is None:
                continue  # keep the measured (Lab) profile over a name-only update
            lab = (1.0 - r) * np.asarray(pc["lab"], dtype=np.float64) + r * np.asarray(tc["lab"], dtype=np.float64)
            name, hx = _classify_lab(lab)
            if not _known_name(name):
                name, hx = tc["name"], tc["hex"]
            setattr(prof, key, {"lab": lab, "conf": float((1 - r) * pc["conf"] + r * tc["conf"]),
                                "n": int(max(pc["n"], tc["n"])), "name": name, "hex": hx})
        t_name, t_conf = ev["type_name"], ev["type_conf"]
        if t_name != TYPE_OCCLUDED:
            if prof.lower_type in (TYPE_OCCLUDED, None) or prof.lower_type == t_name:
                prof.lower_type = t_name
                prof.lower_type_conf = max(prof.lower_type_conf * 0.95, t_conf)
            elif t_conf > prof.lower_type_conf:
                prof.lower_type = t_name
                prof.lower_type_conf = 0.8 * t_conf
            else:
                prof.lower_type_conf *= 0.9

    def _commit_accessories(self, tr: _Track, prof: Optional[_Profile]) -> None:
        """Add the track's accessory statistics gathered since the last commit to the identity."""
        if prof is None:
            return
        d_hits = tr.acc_hits - tr.acc_committed_hits
        d_checks = tr.acc_checks - tr.acc_committed_checks
        if d_checks.max(initial=0.0) <= 0:
            return
        prof.acc_hits = prof.acc_hits + np.maximum(d_hits, 0.0)
        prof.acc_checks = prof.acc_checks + np.maximum(d_checks, 0.0)
        tr.acc_committed_hits = tr.acc_hits.copy()
        tr.acc_committed_checks = tr.acc_checks.copy()
        prof.dirty = True

    # ------------------------------------------------------------------ drift and late merge
    def _check_drift(self, tr: _Track) -> None:
        """Detach a confirmed track whose recent evidence persistently contradicts its identity."""
        cfg = self.config
        if tr.obs_since_uid < cfg.drift_window or tr.obs_since_uid - tr.last_drift_obs < cfg.drift_check_every:
            return
        tr.last_drift_obs = tr.obs_since_uid
        prof = self._profiles.get(tr.uid)
        if prof is None:
            self._detach(tr, keep=0)
            return
        if prof.appearance_ok(self.model_id) and not self._is_mature(prof):
            return  # identity still being built from this track: nothing reliable to drift from
        ev = self._evidence(tr, window=cfg.drift_window)
        if ev["n_good"] < max(3, cfg.drift_window // 2):
            return
        cache = _GalleryCache([prof], self.model_id, self.dim)
        adj, fused, veto = self._score(ev, cache)
        score = float(fused[0]) if fused.size else 0.0
        tr.confidence = float(np.clip(0.7 * tr.confidence + 0.3 * score, 0.0, 1.0))
        if bool(veto[0]) or score < cfg.drift_threshold:
            tr.drift_strikes += 1
        else:
            tr.drift_strikes = 0
        if tr.drift_strikes >= cfg.drift_patience:
            logger.info(f"Track #{tr.track_id} drifted away from {tr.uid} (score {score:.2f}); re-identifying")
            self._detach(tr, keep=cfg.drift_window)

    def _check_late_merge(self, tr: _Track) -> Optional[str]:
        """Merge a provisional identity minted by this track into a strongly matching older identity."""
        cfg = self.config
        tr.merge_done.add(tr.obs_since_uid)
        src = self._profiles.get(tr.uid)
        if src is None or src.created_by_track != tr.track_id or src.sightings > 1:
            return None
        held = self._held_uids({tr.track_id})
        cache = self._gallery_cache()
        if cache.size <= 1:
            return None
        ev = self._evidence(tr, accumulated=True)
        adj, _, _ = self._score(ev, cache)
        for c, uid in enumerate(cache.uids):
            if uid == src.uid or uid in held:
                adj[c] = -np.inf
            else:
                other = self._profiles[uid]
                # Never merge into another identity minted this session by a still-tracked track
                if other.created_by_track is not None and other.created_by_track in self._tracks:
                    adj[c] = -np.inf
        if not np.isfinite(adj).any():
            return None
        order = np.argsort(-adj)
        best = float(adj[order[0]])
        runner = float(adj[order[1]]) if order.size > 1 and np.isfinite(adj[order[1]]) else -np.inf
        if best >= cfg.merge_threshold and best - runner >= cfg.merge_margin:
            dst_uid = cache.uids[order[0]]
            self._merge(src.uid, dst_uid, tr, best)
            return src.uid
        return None

    def _merge(self, src_uid: str, dst_uid: str, tr: _Track, score: float) -> None:
        """Fold provisional identity `src_uid` into `dst_uid` (gallery + DB) and relabel the track."""
        for other in self._tracks.values():
            if other is not tr and other.uid == dst_uid:
                self._detach(other, keep=0)  # a stale holder never keeps a uid a live track takes over
        src = self._profiles.pop(src_uid)
        dst = self._profiles[dst_uid]
        if self.db is not None and hasattr(self.db, "merge_identities"):
            try:
                self.db.merge_identities(src_uid, dst_uid)
            except Exception as e:
                logger.warning(f"DB merge {src_uid} -> {dst_uid} failed: {e}")
        # Appearance
        if src.model_id == self.model_id:
            if dst.model_id != self.model_id:
                dst.reset_appearance(self.model_id)
                dst.protos, dst.proto_q = src.protos.copy(), src.proto_q.copy()
                dst.mean, dst.n_mean = src.mean, src.n_mean
            else:
                if src.protos.shape[0]:
                    dst.protos = np.vstack([dst.protos, src.protos]).astype(np.float32)
                    dst.proto_q = np.concatenate([dst.proto_q, src.proto_q]).astype(np.float32)
                    self._prune_prototypes(dst)
                if src.mean is not None:
                    if dst.mean is None:
                        dst.mean, dst.n_mean = src.mean, src.n_mean
                    else:
                        wd, ws = max(dst.n_mean, 1), max(src.n_mean, 1)
                        dst.mean = _unit(wd * dst.mean + ws * src.mean)
                        dst.n_mean = wd + ws
        # Attributes: the fresher sighting (src) refreshes colours where the dst profile is weaker
        for key in ("upper", "lower"):
            s_c, d_c = getattr(src, key), getattr(dst, key)
            if s_c is not None and (d_c is None or s_c.get("n", 0) >= d_c.get("n", 0)):
                setattr(dst, key, dict(s_c))
        if src.lower_type != TYPE_OCCLUDED and src.lower_type_conf >= dst.lower_type_conf:
            dst.lower_type, dst.lower_type_conf = src.lower_type, src.lower_type_conf
        dst.acc_hits = dst.acc_hits + src.acc_hits
        dst.acc_checks = dst.acc_checks + src.acc_checks
        dst.n_observations += src.n_observations
        dst.best_crop_quality = max(dst.best_crop_quality, src.best_crop_quality)
        dst.last_seen_iso = _now_iso()
        dst.last_seen_frame = self._frame
        dst.sightings += 1
        self._retired_uids.add(src_uid)
        self._session_uids.discard(src_uid)
        self._session_uids.add(dst_uid)
        tr.uid = dst_uid
        tr.status = "reacquired"
        tr.decision_frame = self._frame
        tr.created_uid = False
        tr.confidence = float(np.clip(score, 0.0, 1.0))
        tr.drift_strikes = 0
        self._touch(dst)
        self._save_profile(dst)
        self._events.append({"type": "merged", "person_uid": dst_uid, "merged_from": src_uid,
                             "track_id": tr.track_id, "confidence": float(score), "frame_idx": self._frame})
        logger.info(f"Late merge: {src_uid} -> {dst_uid} (track #{tr.track_id}, score {score:.2f})")

    # ------------------------------------------------------------------ outputs
    def _present_list(self, hits: np.ndarray, checks: np.ndarray, mask: Optional[np.ndarray] = None) -> List[str]:
        present, _ = accessory_presence(hits, checks, self.config)
        if mask is not None:
            present = present & mask
        return [ACCESSORY_ORDER[i] for i in np.where(present)[0]]

    @staticmethod
    def _color_out(c: Optional[Dict[str, Any]], fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if c is not None:
            lab = c.get("lab")
            return {"color": c.get("name") or "Unknown", "hex": c.get("hex") or "#808080",
                    "lab": [float(v) for v in lab] if lab is not None else None,
                    "confidence": float(c.get("conf", 0.0))}
        if fallback:
            out = {"color": fallback.get("color", "Unknown"), "hex": fallback.get("hex", "#808080"),
                   "lab": fallback.get("lab"), "confidence": float(fallback.get("confidence", 0.0) or 0.0)}
            return out
        return {"color": "Unknown", "hex": "#808080", "lab": None, "confidence": 0.0}

    def _lower_out(self, c: Optional[Dict[str, Any]], ltype: str, tconf: float,
                   fallback: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Lower-garment output dict; "color" equals the type string when no pants colour is measurable."""
        if ltype == TYPE_PANTS:
            if c is not None:
                out = self._color_out(c)
            elif fallback and self._lower_type_of(fallback) == TYPE_PANTS:
                out = self._color_out(None, fallback)
            else:
                out = {"color": TYPE_PANTS, "hex": "#404040", "lab": None, "confidence": float(tconf)}
            out["type"] = TYPE_PANTS
            return out
        if ltype == TYPE_BARE:
            hx = fallback.get("hex") if fallback and self._lower_type_of(fallback) == TYPE_BARE else None
            return {"color": TYPE_BARE, "hex": hx or "#D4B996", "lab": None, "confidence": float(tconf),
                    "type": TYPE_BARE}
        return {"color": TYPE_OCCLUDED, "hex": "#404040", "lab": None, "confidence": float(tconf),
                "type": TYPE_OCCLUDED}

    def _assignment_for(self, tr: _Track, o: TrackObservation, merged_from: Optional[str]) -> IdentityAssignment:
        cfg = self.config
        ev = self._evidence(tr, need_desc=False)
        prof = self._profiles.get(tr.uid) if tr.uid else None
        upper_c = ev["upper"] if ev["upper"] is not None else (prof.upper if prof else None)
        upper = self._color_out(upper_c, o.upper if upper_c is None else None)
        lower = self._lower_out(ev["lower"], ev["type_name"], ev["type_conf"], o.lower)

        accessories = self._present_list(tr.acc_hits, tr.acc_checks)
        if prof is not None:
            _, t_absent = accessory_presence(tr.acc_hits, tr.acc_checks, cfg)
            id_present, _ = accessory_presence(prof.acc_hits, prof.acc_checks, cfg)
            extra = id_present & _STABLE_MASK & ~t_absent
            names = set(accessories) | {ACCESSORY_ORDER[i] for i in np.where(extra)[0]}
            accessories = [a for a in ACCESSORY_ORDER if a in names]

        posture = "Standing"
        if tr.postures:
            counts: Dict[str, int] = {}
            for p in tr.postures:
                counts[p] = counts.get(p, 0) + 1
            posture = max(counts, key=counts.get)

        is_best = False
        if prof is not None and tr.drift_strikes == 0:
            q = float(o.quality or 0.0)
            if q >= cfg.min_quality and q > prof.best_crop_quality + cfg.best_crop_margin:
                prof.best_crop_quality = q
                prof.dirty = True
                is_best = True

        if tr.uid is None:
            status = "tentative"
            progress = max(tr.n_good / max(cfg.min_good_observations, 1),
                           tr.n_obs / max(cfg.max_tentative_observations, 1))
            confidence = float(np.clip(progress, 0.0, 0.99))
            label = TENTATIVE_LABEL
        else:
            if self._frame - tr.decision_frame <= cfg.status_hold_frames and tr.status in ("new", "reacquired"):
                status = tr.status
            else:
                status = "confirmed"
                tr.status = "confirmed"
            confidence = float(np.clip(tr.confidence, 0.0, 1.0))
            label = tr.uid
        return IdentityAssignment(track_id=tr.track_id, person_uid=tr.uid, status=status, confidence=confidence,
                                  display_label=label, upper=upper, lower=lower, accessories=accessories,
                                  posture=posture, merged_from=merged_from, is_best_crop=is_best)

    # ------------------------------------------------------------------ end / persistence
    def _end_tracks_locked(self, track_ids: List[int]) -> None:
        for tid in track_ids:
            try:
                tr = self._tracks.pop(int(tid), None)
            except Exception:
                continue
            if tr is None:
                continue
            if tr.uid is not None:
                # Last chance for a provisional identity: merge check with the complete track evidence
                if tr.created_uid and tr.obs_since_uid >= self.config.drift_window                         and tr.obs_since_uid not in tr.merge_done:
                    self._check_late_merge(tr)
                prof = self._profiles.get(tr.uid)
                if prof is not None:
                    self._update_attributes(tr, prof)
                    self._commit_accessories(tr, prof)
                    prof.last_seen_frame = tr.last_frame
                    self._touch(prof)
            for key in [k for k in self._imp_pair_last if tr.track_id in k]:
                self._imp_pair_last.pop(key, None)

    def _flush_locked(self) -> None:
        self._last_flush_frame = self._frame
        for prof in list(self._profiles.values()):
            if prof.dirty:
                self._save_profile(prof)

    def _save_profile(self, prof: _Profile) -> None:
        if self.db is None or not hasattr(self.db, "save_identity_profile"):
            prof.dirty = False
            return
        try:
            if self.db.save_identity_profile(self._profile_to_dict(prof)):
                prof.dirty = False
        except Exception as e:
            logger.warning(f"Could not persist identity profile {prof.uid}: {e}")

    def _profile_to_dict(self, p: _Profile) -> Dict[str, Any]:
        def _c(c: Optional[Dict[str, Any]], key: str) -> Any:
            if c is None:
                return None
            if key == "lab":
                return [float(v) for v in c["lab"]] if c.get("lab") is not None else None
            return c.get(key)

        return {
            "person_uid": p.uid,
            "embedding_model": p.model_id,
            "mean_embedding": p.mean,
            "prototypes": p.protos if p.protos.shape[0] else None,
            "upper_color": _c(p.upper, "name"),
            "upper_hex": _c(p.upper, "hex"),
            "upper_lab": _c(p.upper, "lab"),
            "lower_color": _c(p.lower, "name") if p.lower_type == TYPE_PANTS else p.lower_type,
            "lower_hex": _c(p.lower, "hex"),
            "lower_lab": _c(p.lower, "lab"),
            "lower_type": p.lower_type,
            "accessory_stats": {name: {"hits": float(p.acc_hits[i]), "checks": float(p.acc_checks[i])}
                                for i, name in enumerate(ACCESSORY_ORDER) if p.acc_checks[i] > 0},
            "n_observations": int(p.n_observations),
            "best_crop_quality": float(p.best_crop_quality),
            "last_seen": p.last_seen_iso or _now_iso(),
            "extra": {
                "upper_conf": float(p.upper["conf"]) if p.upper else 0.0,
                "upper_n": int(p.upper["n"]) if p.upper else 0,
                "lower_conf": float(p.lower["conf"]) if p.lower else 0.0,
                "lower_n": int(p.lower["n"]) if p.lower else 0,
                "lower_type_conf": float(p.lower_type_conf),
                "n_mean": int(p.n_mean),
                "proto_quality": [round(float(v), 4) for v in p.proto_q],
                "first_seen": p.first_seen_iso,
            },
        }

    def _profile_from_dict(self, d: Dict[str, Any]) -> _Profile:
        p = _Profile(str(d["person_uid"]), str(d.get("embedding_model") or "unknown"), self.dim)
        extra = d.get("extra") or {}
        mean = d.get("mean_embedding")
        if mean is not None:
            p.mean = _unit(np.asarray(mean, dtype=np.float32))
        protos = d.get("prototypes")
        if protos is not None:
            arr = np.asarray(protos, dtype=np.float32).reshape(-1, self.dim)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            keep = norms[:, 0] > 1e-8
            p.protos = (arr[keep] / norms[keep]).astype(np.float32)
            pq = np.asarray(extra.get("proto_quality") or [], dtype=np.float32)
            p.proto_q = pq[keep] if pq.shape[0] == arr.shape[0] else np.full(p.protos.shape[0], 0.6, np.float32)
        p.n_mean = int(extra.get("n_mean", max(p.protos.shape[0], 1 if p.mean is not None else 0)))
        u_lab = _as_lab(d.get("upper_lab"))
        if u_lab is not None or _known_name(d.get("upper_color")):
            p.upper = {"lab": u_lab, "conf": float(extra.get("upper_conf", 0.5)), "n": int(extra.get("upper_n", 3)),
                       "name": d.get("upper_color") or "Unknown", "hex": d.get("upper_hex") or "#808080"}
        ltype = d.get("lower_type") if d.get("lower_type") in _TYPE_CODE else TYPE_OCCLUDED
        p.lower_type = ltype
        p.lower_type_conf = float(extra.get("lower_type_conf", 0.5 if ltype != TYPE_OCCLUDED else 0.0))
        l_lab = _as_lab(d.get("lower_lab"))
        if ltype == TYPE_PANTS and (l_lab is not None or _known_name(d.get("lower_color"))):
            p.lower = {"lab": l_lab, "conf": float(extra.get("lower_conf", 0.5)), "n": int(extra.get("lower_n", 3)),
                       "name": d.get("lower_color") or "Unknown", "hex": d.get("lower_hex") or "#404040"}
        stats = d.get("accessory_stats") or {}
        for name, st in stats.items():
            idx = _ACC_INDEX.get(normalize_accessory(name))
            if idx is None or not isinstance(st, dict):
                continue
            p.acc_hits[idx] = float(st.get("hits", 0.0))
            p.acc_checks[idx] = float(st.get("checks", 0.0))
        # Keep long histories bounded so recent sightings still matter
        peak = float(p.acc_checks.max(initial=0.0))
        if peak > 200.0:
            scale = 200.0 / peak
            p.acc_hits *= scale
            p.acc_checks *= scale
        p.n_observations = int(d.get("n_observations") or 0)
        p.best_crop_quality = float(d.get("best_crop_quality") or 0.0)
        p.last_seen_iso = d.get("last_seen")
        p.first_seen_iso = extra.get("first_seen")
        return p
