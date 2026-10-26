"""
Test suite for the persistent IdentityManager (forensic_app.core.identity)
Verifies, on a realistic synthetic world (ImageNet-like embeddings with a large shared component,
lighting jitter on clothing colours, noisy accessory detections):
1. A person who exits and re-enters with a new track id keeps the same UID ("reacquired").
2. Different people (different clothes, or identical clothes) get distinct UIDs.
3. Concurrent tracks never share a UID; persistence across sessions; legacy UID sequence continuation.
4. Late merge of a provisional identity, accessory consensus, shirt-colour veto, drift detachment.
"""

import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forensic_app.core.database import ForensicDatabase
from forensic_app.core.signature import ForensicSignature
from forensic_app.core.identity import (
    IdentityAssignment,
    IdentityConfig,
    IdentityManager,
    TrackObservation,
    accessory_cue,
    ciede2000_batch,
    estimate_observation_quality,
    identity_score,
)
from forensic_app.utils.color_naming import classify_cielab_color, delta_e_ciede2000

DIM = 512
MODEL_ID = "synthetic-test-v1"

# Reference clothing colours (CIELAB)
NAVY = (18.0, 8.0, -32.0)
RED = (46.0, 62.0, 42.0)
WHITE = (93.0, 0.0, 2.0)
BLACK = (10.0, 0.0, 0.0)
BEIGE = (82.0, 2.0, 16.0)
DENIM = (42.0, -2.0, -24.0)
GREEN = (48.0, -42.0, 30.0)
GREY = (55.0, 0.0, 0.0)
YELLOW = (82.0, 4.0, 72.0)
PURPLE = (35.0, 40.0, -38.0)
OLIVE = (40.0, -16.0, 30.0)
KHAKI = (68.0, 2.0, 26.0)


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


class SynthPerson:
    """Ground-truth person: identity vector + clothing + accessories."""

    def __init__(self, vec, upper_lab, lower_lab, lower_type, accessories):
        self.vec = vec
        self.upper_lab = np.asarray(upper_lab, dtype=np.float64)
        self.lower_lab = None if lower_lab is None else np.asarray(lower_lab, dtype=np.float64)
        self.lower_type = lower_type
        self.accessories = set(accessories)


class SyntheticWorld:
    """
    Embedding model:  e = normalize(0.8*common + 0.55*person + 0.2*sighting + noise*frame_noise)
    -> raw cosine between DIFFERENT people is ~0.6 (like ImageNet features), so the identity manager
    must rely on calibration / centering and attribute cues rather than raw cosine.
    """

    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.common = _unit(self.rng.standard_normal(DIM))

    def person(self, upper, lower=None, lower_type="Pants/Jeans", accessories=()) -> SynthPerson:
        return SynthPerson(_unit(self.rng.standard_normal(DIM)), upper, lower, lower_type, accessories)

    def sighting(self) -> Dict[str, np.ndarray]:
        """Per-sighting pose / lighting conditions."""
        return {
            "offset": _unit(self.rng.standard_normal(DIM)),
            "light": np.array([self.rng.uniform(-4, 4), self.rng.uniform(-3, 3), self.rng.uniform(-3, 3)]),
        }

    def embedding(self, person: SynthPerson, sighting, noise: float = 0.25) -> np.ndarray:
        e = (0.8 * self.common + 0.55 * person.vec + 0.2 * sighting["offset"]
             + noise * _unit(self.rng.standard_normal(DIM)))
        return _unit(e)

    def _garment(self, lab, light, confidence=0.75) -> Dict:
        jitter = self.rng.normal(0.0, 2.0, 3)
        val = np.asarray(lab) + light + jitter
        val[0] = float(np.clip(val[0], 0.0, 100.0))
        name, hx = classify_cielab_color(val.astype(np.float32))
        return {"color": name, "hex": hx, "lab": [float(v) for v in val], "confidence": confidence}

    def observation(self, person: SynthPerson, track_id: int, frame_idx: int, sighting,
                    quality: float = 0.85, noise: float = 0.25, checked: Optional[bool] = None,
                    detect_prob: float = 0.6, false_pos_prob: float = 0.03,
                    upper_override=None, embedding_override=None, bbox=None) -> TrackObservation:
        upper = self._garment(upper_override if upper_override is not None else person.upper_lab, sighting["light"])
        if person.lower_type == "Pants/Jeans" and person.lower_lab is not None:
            lower = self._garment(person.lower_lab, sighting["light"])
            lower["type"] = "Pants/Jeans"
        else:
            lower = {"color": person.lower_type, "hex": "#404040", "lab": None, "confidence": 0.7,
                     "type": person.lower_type}
        if checked is None:
            checked = (frame_idx + track_id) % 3 == 0
        accessories: Dict[str, float] = {}
        if checked:
            for acc in person.accessories:
                if self.rng.random() < detect_prob:
                    accessories[acc] = float(self.rng.uniform(0.4, 0.9))
            if self.rng.random() < false_pos_prob:
                fp = str(self.rng.choice(["bottle", "handbag", "cell phone", "umbrella"]))
                accessories.setdefault(fp, float(self.rng.uniform(0.25, 0.5)))
        emb = embedding_override if embedding_override is not None else self.embedding(person, sighting, noise)
        if bbox is None:
            x = 40 + 120 * (track_id % 5)
            bbox = (x, 60, x + 80, 300)
        return TrackObservation(track_id=track_id, frame_idx=frame_idx, bbox=bbox, embedding=emb,
                                upper=upper, lower=lower, accessories=accessories,
                                accessory_checked=bool(checked), quality=quality, posture="Standing")


def run_sighting(mgr: IdentityManager, world: SyntheticWorld, person: SynthPerson, track_id: int,
                 start: int, n_frames: int, end: bool = True, **kw) -> List[IdentityAssignment]:
    sighting = world.sighting()
    out = []
    for k in range(n_frames):
        f = start + k
        res = mgr.update([world.observation(person, track_id, f, sighting, **kw)], f)
        out.append(res[track_id])
    if end:
        mgr.end_tracks([track_id])
    return out


def run_concurrent(mgr: IdentityManager, world: SyntheticWorld, people: Sequence[Tuple[SynthPerson, int]],
                   start: int, n_frames: int, end: bool = True) -> Dict[int, List[IdentityAssignment]]:
    sightings = {tid: world.sighting() for _, tid in people}
    out: Dict[int, List[IdentityAssignment]] = {tid: [] for _, tid in people}
    for k in range(n_frames):
        f = start + k
        obs = [world.observation(p, tid, f, sightings[tid]) for p, tid in people]
        res = mgr.update(obs, f)
        for _, tid in people:
            out[tid].append(res[tid])
        uids = [res[tid].person_uid for _, tid in people if res[tid].person_uid]
        assert len(uids) == len(set(uids)), f"uid shared by concurrent tracks at frame {f}: {uids}"
    if end:
        mgr.end_tracks([tid for _, tid in people])
    return out


def final_uid(assignments: List[IdentityAssignment]) -> Optional[str]:
    return assignments[-1].person_uid if assignments else None


class TestIdentityManager(unittest.TestCase):
    """End-to-end identity behaviour on synthetic tracks."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="identity_test_")
        self.db_path = os.path.join(self.tmpdir, "identity.db")
        self.db = ForensicDatabase(db_path=self.db_path)
        self.world = SyntheticWorld(seed=11)
        self.mgr = IdentityManager(self.db, embedding_model_id=MODEL_ID)

    def tearDown(self):
        try:
            self.db.close()
        finally:
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------ (a) re-entry
    def test_reentry_keeps_same_uid(self):
        p1 = self.world.person(NAVY, BEIGE, accessories=["backpack"])
        first = run_sighting(self.mgr, self.world, p1, track_id=1, start=0, n_frames=60)
        self.assertEqual(first[0].status, "tentative")
        self.assertEqual(first[0].display_label, "Identifying...")
        uid = final_uid(first)
        self.assertEqual(uid, "UID-0001")
        events = self.mgr.get_events()
        self.assertEqual([e["type"] for e in events], ["new"])

        second = run_sighting(self.mgr, self.world, p1, track_id=7, start=200, n_frames=40)
        self.assertEqual(final_uid(second), uid, "Re-entering person must get the SAME uid")
        statuses = {a.status for a in second if a.person_uid}
        self.assertIn("reacquired", statuses)
        events = self.mgr.get_events()
        self.assertTrue(any(e["type"] == "reacquired" and e["person_uid"] == uid and e["track_id"] == 7
                            for e in events), events)
        self.assertEqual(self.mgr.session_identity_count, 1)
        # Decision is taken quickly (tentative phase only a handful of frames)
        first_uid_frame = next(i for i, a in enumerate(second) if a.person_uid)
        self.assertLessEqual(first_uid_frame, 15)

    # ------------------------------------------------------------------ (b) different clothes
    def test_distinct_people_get_distinct_uids(self):
        p1 = self.world.person(NAVY, BEIGE)
        p2 = self.world.person(RED, None, lower_type="Bare Legs / Shorts")
        a = run_sighting(self.mgr, self.world, p1, 1, 0, 40)
        b = run_sighting(self.mgr, self.world, p2, 2, 100, 40)
        self.assertNotEqual(final_uid(a), final_uid(b))
        self.assertEqual(self.mgr.session_identity_count, 2)
        # Both come back (in reverse order) and are recognised
        b2 = run_sighting(self.mgr, self.world, p2, 3, 200, 40)
        a2 = run_sighting(self.mgr, self.world, p1, 4, 300, 40)
        self.assertEqual(final_uid(a2), final_uid(a))
        self.assertEqual(final_uid(b2), final_uid(b))

    # ------------------------------------------------------------------ (c) same clothes
    def test_same_clothes_different_people(self):
        p1 = self.world.person(BLACK, DENIM)
        p2 = self.world.person(BLACK, DENIM)  # identical outfit, different person
        a = run_sighting(self.mgr, self.world, p1, 1, 0, 40)
        b = run_sighting(self.mgr, self.world, p2, 2, 100, 40)
        self.assertIsNotNone(final_uid(b))
        self.assertNotEqual(final_uid(a), final_uid(b), "Appearance must separate identical outfits")
        a2 = run_sighting(self.mgr, self.world, p1, 3, 200, 40)
        self.assertEqual(final_uid(a2), final_uid(a))

    # ------------------------------------------------------------------ (d) exclusivity
    def test_concurrent_tracks_never_share_uid(self):
        p1 = self.world.person(GREEN, KHAKI)
        first = run_sighting(self.mgr, self.world, p1, 1, 0, 40)
        uid = final_uid(first)
        # Two simultaneous tracks of the same-looking person (e.g. a duplicate detection / twins)
        out = run_concurrent(self.mgr, self.world, [(p1, 10), (p1, 11)], 100, 40)
        u10, u11 = final_uid(out[10]), final_uid(out[11])
        self.assertIn(uid, (u10, u11), "One of the tracks should re-acquire the known identity")
        self.assertNotEqual(u10, u11)

    # ------------------------------------------------------------------ (e) persistence
    def test_persistence_across_sessions(self):
        p1 = self.world.person(YELLOW, BLACK, accessories=["cap"])
        uid = final_uid(run_sighting(self.mgr, self.world, p1, 1, 0, 50))
        self.mgr.flush()
        self.db.close()

        # New process: new database handle + new identity manager on the same file
        self.db = ForensicDatabase(db_path=self.db_path)
        profiles = self.db.load_identity_profiles()
        self.assertEqual([p["person_uid"] for p in profiles], [uid])
        self.assertEqual(profiles[0]["embedding_model"], MODEL_ID)
        self.assertIsNotNone(profiles[0]["prototypes"])

        mgr2 = IdentityManager(self.db, embedding_model_id=MODEL_ID)
        again = run_sighting(mgr2, self.world, p1, 1, 0, 40)
        self.assertEqual(final_uid(again), uid, "Previous session's identity must be re-acquired")
        p2 = self.world.person(PURPLE, GREY)
        other = run_sighting(mgr2, self.world, p2, 2, 100, 40)
        self.assertEqual(final_uid(other), "UID-0002", "Sequence must continue after persisted identities")

    def test_legacy_uid_sequence_continues(self):
        legacy = ForensicSignature(person_uid="UID-CAM_01-0036", camera_id="CAM_01",
                                   upper_garment={"color": "Brown", "hex": "#65371E"},
                                   lower_garment={"color": "None / Occluded", "hex": "#404040"},
                                   embedding=[0.05] * DIM)
        self.db.save_signature(legacy)
        self.assertEqual(self.db.next_person_sequence(), 37)
        mgr = IdentityManager(self.db, embedding_model_id=MODEL_ID)
        p = self.world.person(WHITE, NAVY)
        self.assertEqual(final_uid(run_sighting(mgr, self.world, p, 1, 0, 30)), "UID-0037")
        q = self.world.person(RED, GREY)
        self.assertEqual(final_uid(run_sighting(mgr, self.world, q, 2, 100, 30, end=False)), "UID-0038")

    # ------------------------------------------------------------------ (f) late merge
    def test_late_merge_of_provisional_identity(self):
        p1 = self.world.person(RED, DENIM)
        uid = final_uid(run_sighting(self.mgr, self.world, p1, 1, 0, 50))
        self.mgr.get_events()

        # Re-entry: first 15 frames are heavily occluded / blurred (poor embedding, no colours)
        sighting = self.world.sighting()
        rng = np.random.default_rng(5)
        out = []
        for k in range(15):
            f = 100 + k
            junk = _unit(0.4 * self.world.common + 1.0 * _unit(rng.standard_normal(DIM)))
            obs = self.world.observation(p1, 9, f, sighting, quality=0.2, embedding_override=junk)
            obs.upper = {"color": "Unknown", "hex": "#808080", "lab": None, "confidence": 0.0}
            obs.lower = {"color": "None / Occluded", "hex": "#404040", "lab": None, "confidence": 0.0,
                         "type": "None / Occluded"}
            out.append(self.mgr.update([obs], f)[9])
        provisional = out[-1].person_uid
        self.assertIsNotNone(provisional, "Forced decision after the tentative window")
        self.assertNotEqual(provisional, uid, "Junk evidence must not be matched immediately")

        merged = None
        for k in range(15, 80):
            f = 100 + k
            a = self.mgr.update([self.world.observation(p1, 9, f, sighting)], f)[9]
            out.append(a)
            if a.merged_from:
                merged = a
        self.assertIsNotNone(merged, "Provisional identity must be merged into the original")
        self.assertEqual(merged.merged_from, provisional)
        self.assertEqual(merged.person_uid, uid)
        self.assertEqual(final_uid(out), uid)
        events = self.mgr.get_events()
        self.assertTrue(any(e["type"] == "merged" and e["merged_from"] == provisional and e["person_uid"] == uid
                            for e in events), events)
        self.assertNotIn(provisional, [p["person_uid"] for p in self.db.load_identity_profiles()])
        self.assertEqual(self.mgr.session_identity_count, 1)

    # ------------------------------------------------------------------ (g) accessories
    def test_accessory_consensus(self):
        p1 = self.world.person(NAVY, BEIGE, accessories=["backpack"])
        out = run_sighting(self.mgr, self.world, p1, 1, 0, 60, end=False, false_pos_prob=0.0)
        # Inject a single-frame false positive handbag
        sighting = self.world.sighting()
        obs = self.world.observation(p1, 1, 60, sighting, checked=True, false_pos_prob=0.0)
        obs.accessories["handbag"] = 0.9
        res = self.mgr.update([obs], 60)[1]
        self.assertIn("backpack", res.accessories)
        self.assertNotIn("handbag", res.accessories)
        self.mgr.end_tracks([1])
        summary = self.mgr.get_identity_summary(final_uid(out))
        self.assertIn("backpack", summary["accessories"])
        self.assertNotIn("handbag", summary["accessories"])

        # Re-entry: the identity's backpack is shown even before it is re-detected in this track
        again = run_sighting(self.mgr, self.world, p1, 2, 200, 12, end=False, detect_prob=0.0)
        self.assertEqual(again[-1].person_uid, final_uid(out))
        self.assertIn("backpack", again[-1].accessories)
        # Persisted with the profile
        self.mgr.flush()
        prof = [p for p in self.db.load_identity_profiles() if p["person_uid"] == final_uid(out)][0]
        self.assertGreaterEqual(prof["accessory_stats"]["backpack"]["hits"], 2)

    # ------------------------------------------------------------------ (h) colour veto
    def test_shirt_color_veto(self):
        p1 = self.world.person(WHITE, BLACK)
        uid = final_uid(run_sighting(self.mgr, self.world, p1, 1, 0, 50))
        # Same appearance vector, but now wearing a clearly different (red) shirt
        red = run_sighting(self.mgr, self.world, p1, 2, 100, 40, upper_override=RED)
        self.assertIsNotNone(final_uid(red))
        self.assertNotEqual(final_uid(red), uid, "Clearly different shirt colour must veto the match")

    # ------------------------------------------------------------------ (i) drift
    def test_drift_detaches_swapped_track(self):
        pa = self.world.person(NAVY, BEIGE)
        pb = self.world.person(YELLOW, GREEN, lower_type="Pants/Jeans")
        ua = final_uid(run_sighting(self.mgr, self.world, pa, 1, 0, 40))
        ub = final_uid(run_sighting(self.mgr, self.world, pb, 2, 100, 40))
        self.assertNotEqual(ua, ub)
        # Track 5 follows A for 40 frames, then the tracker silently switches to B
        sa, sb = self.world.sighting(), self.world.sighting()
        out = []
        for k in range(100):
            f = 200 + k
            person, s = (pa, sa) if k < 40 else (pb, sb)
            out.append(self.mgr.update([self.world.observation(person, 5, f, s)], f)[5])
        self.assertEqual(out[39].person_uid, ua)
        self.assertEqual(final_uid(out), ub, "Swapped track must be detached and re-identified as B")
        swap_frames = [i for i, a in enumerate(out) if i >= 40 and a.person_uid == ub]
        self.assertTrue(swap_frames and swap_frames[0] < 40 + 45, "Drift should be detected reasonably fast")

    # ------------------------------------------------------------------ crowd
    def test_crowd_reidentification(self):
        outfits = [(NAVY, BEIGE), (RED, DENIM), (WHITE, BLACK), (BLACK, DENIM), (GREEN, KHAKI),
                   (GREY, BLACK), (BLACK, DENIM), (YELLOW, NAVY), (PURPLE, GREY), (OLIVE, KHAKI)]
        people = [self.world.person(u, l) for u, l in outfits]
        uids = {}
        tid = 1
        # People walk by two at a time (concurrent tracks give impostor calibration pairs)
        for i in range(0, len(people), 2):
            out = run_concurrent(self.mgr, self.world, [(people[i], tid), (people[i + 1], tid + 1)], 100 * i, 40)
            uids[i], uids[i + 1] = final_uid(out[tid]), final_uid(out[tid + 1])
            tid += 2
        self.assertEqual(len(set(uids.values())), len(people), f"Every person needs a distinct uid: {uids}")
        # Everyone comes back alone, in a shuffled order
        order = [7, 2, 9, 0, 5, 3, 8, 1, 6, 4]
        for j, i in enumerate(order):
            again = run_sighting(self.mgr, self.world, people[i], 100 + j, 2000 + 100 * j, 40)
            self.assertEqual(final_uid(again), uids[i], f"Person {i} must be re-identified")
        self.assertEqual(self.mgr.session_identity_count, len(people))

    # ------------------------------------------------------------------ tracker fragmentation
    def test_fragmented_tracks_reacquire_quickly(self):
        pa = self.world.person(NAVY, BEIGE, accessories=["backpack"])
        pb = self.world.person(RED, DENIM)
        sa, sb = self.world.sighting(), self.world.sighting()
        uids_a, uids_b = set(), set()
        tid_a, tid_b = 1, 2
        frame = 0
        labels_after_break = []
        for segment in range(6):
            # The tracker loses both people every 25 frames and re-creates tracks 2 frames later
            for k in range(25):
                obs = [self.world.observation(pa, tid_a, frame, sa), self.world.observation(pb, tid_b, frame, sb)]
                res = self.mgr.update(obs, frame)
                if segment > 0 and k < 4:
                    labels_after_break.append(res[tid_a].person_uid)
                for tid, bucket in ((tid_a, uids_a), (tid_b, uids_b)):
                    if res[tid].person_uid:
                        bucket.add(res[tid].person_uid)
                frame += 1
            self.mgr.end_tracks([tid_a, tid_b])
            frame += 2
            tid_a, tid_b = tid_a + 10, tid_b + 10
        self.assertEqual(len(uids_a), 1, uids_a)
        self.assertEqual(len(uids_b), 1, uids_b)
        self.assertNotEqual(uids_a, uids_b)
        # After a break the uid is back within ~3 frames (fast path, no long "Identifying...")
        self.assertGreaterEqual(sum(1 for u in labels_after_break if u), len(labels_after_break) // 2)

    # ------------------------------------------------------------------ misc
    def test_reset_and_zero_embeddings(self):
        p1 = self.world.person(NAVY, BEIGE)
        zero = np.zeros(DIM, dtype=np.float32)
        out = run_sighting(self.mgr, self.world, p1, 1, 0, 30, embedding_override=zero)
        self.assertIsNotNone(final_uid(out), "Attribute-only evidence still yields an identity")
        self.db.clear_database()
        self.mgr.reset()
        self.assertEqual(self.mgr.session_identity_count, 0)
        self.assertEqual(self.mgr.gallery_size, 0)
        out2 = run_sighting(self.mgr, self.world, p1, 2, 100, 30)
        self.assertEqual(final_uid(out2), "UID-0001")

    def test_model_mismatch_uses_attributes_only(self):
        p1 = self.world.person(RED, DENIM, accessories=["backpack"])
        uid = final_uid(run_sighting(self.mgr, self.world, p1, 1, 0, 60))
        self.mgr.flush()
        mgr2 = IdentityManager(self.db, embedding_model_id="another-model-v9")
        again = run_sighting(mgr2, self.world, p1, 1, 0, 40)
        # Same clothes + backpack: attribute-only match under the stricter threshold
        self.assertEqual(final_uid(again), uid)
        mgr2.flush()
        prof = [p for p in self.db.load_identity_profiles() if p["person_uid"] == uid][0]
        self.assertEqual(prof["embedding_model"], "another-model-v9", "Profile adopts the new embedding space")


class TestDatabaseIdentityExtensions(unittest.TestCase):
    """identity_profiles table, merge_identities, uid sequence and attribute query semantics."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="identity_db_test_")
        self.db = ForensicDatabase(db_path=os.path.join(self.tmpdir, "records.db"))
        self.rng = np.random.default_rng(0)

    def tearDown(self):
        try:
            self.db.close()
        finally:
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _sig(self, uid, upper="Navy Blue", lower="Beige", accessories=(), crop_path=None, ts="2026-01-01T10:00:00Z"):
        emb = _unit(self.rng.standard_normal(DIM))
        return ForensicSignature(person_uid=uid, camera_id="CAM_01", timestamp=ts,
                                 upper_garment={"color": upper, "hex": "#0F173E"},
                                 lower_garment={"color": lower, "hex": "#E0D4BB"},
                                 accessories=list(accessories), embedding=emb.tolist(), crop_path=crop_path)

    def test_profile_roundtrip(self):
        protos = np.stack([_unit(self.rng.standard_normal(DIM)) for _ in range(5)])
        profile = {
            "person_uid": "UID-0004", "embedding_model": MODEL_ID, "mean_embedding": protos.mean(axis=0),
            "prototypes": protos, "upper_color": "Navy Blue", "upper_hex": "#0F173E", "upper_lab": [18.0, 8.0, -32.0],
            "lower_color": "Beige", "lower_hex": "#E0D4BB", "lower_lab": [82.0, 2.0, 16.0], "lower_type": "Pants/Jeans",
            "accessory_stats": {"backpack": {"hits": 5.0, "checks": 9.0}}, "n_observations": 42,
            "best_crop_quality": 0.8, "extra": {"upper_conf": 0.6},
        }
        self.assertTrue(self.db.save_identity_profile(profile))
        profile["n_observations"] = 43
        self.assertTrue(self.db.save_identity_profile(profile))  # upsert
        rows = self.db.load_identity_profiles()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        np.testing.assert_allclose(row["prototypes"], protos, atol=1e-6)
        self.assertEqual(row["proto_count"], 5)
        self.assertEqual(row["n_observations"], 43)
        self.assertEqual(row["upper_lab"], [18.0, 8.0, -32.0])
        self.assertEqual(row["accessory_stats"]["backpack"]["hits"], 5.0)
        self.assertEqual(row["extra"]["upper_conf"], 0.6)
        self.assertEqual(self.db.next_person_sequence(), 5)
        self.assertTrue(self.db.delete_identity_profile("UID-0004"))
        self.assertEqual(self.db.load_identity_profiles(), [])

    def test_next_person_sequence_mixed_formats(self):
        self.assertEqual(self.db.next_person_sequence(), 1)
        self.db.save_signature(self._sig("UID-CAM_01-0036"))
        self.db.save_identity_profile({"person_uid": "UID-0012", "embedding_model": MODEL_ID})
        self.assertEqual(self.db.next_person_sequence(), 37)

    def test_merge_identities(self):
        crops = Path(self.tmpdir) / "crops"
        crops.mkdir(parents=True, exist_ok=True)
        src_crop = crops / "UID-0002.jpg"
        src_crop.write_bytes(b"jpegdata")
        self.db.save_signature(self._sig("UID-0001", ts="2026-01-01T10:00:00Z"))
        self.db.save_signature(self._sig("UID-0001", ts="2026-01-01T10:00:05Z"))
        self.db.save_signature(self._sig("UID-0002", upper="Red", accessories=["backpack"], crop_path=str(src_crop),
                                         ts="2026-01-01T11:00:00Z"))
        for uid in ("UID-0001", "UID-0002"):
            self.db.save_identity_profile({"person_uid": uid, "embedding_model": MODEL_ID})

        self.assertTrue(self.db.merge_identities("UID-0002", "UID-0001"))
        stats = self.db.get_stats()
        self.assertEqual(stats["total_persons"], 1)
        self.assertEqual(stats["total_signatures"], 3)
        rows = self.db.conn.execute("SELECT DISTINCT person_uid FROM signatures").fetchall()
        self.assertEqual([r[0] for r in rows], ["UID-0001"])
        person = self.db.conn.execute("SELECT * FROM persons").fetchone()
        self.assertEqual(person["upper_color"], "Red", "Freshest sighting refreshes the person summary")
        self.assertNotIn("UID-0002", set(self.db.index_to_uid.values()))
        self.assertNotIn("UID-0002", self.db.person_mean_embeddings)
        self.assertNotIn("UID-0002", self.db.uid_to_sig)
        self.assertFalse(src_crop.exists())
        self.assertTrue((crops / "UID-0001.jpg").exists(), "Crop moved to the surviving uid")
        paths = {r[0] for r in self.db.conn.execute("SELECT crop_path FROM signatures WHERE crop_path IS NOT NULL")}
        self.assertEqual(paths, {str(crops / "UID-0001.jpg")})
        self.assertEqual([p["person_uid"] for p in self.db.load_identity_profiles()], ["UID-0001"])
        # Vector search now resolves to the surviving uid only
        uid, _ = self.db.match_person(self.db.uid_to_sig["UID-0001"].get_embedding_numpy(), "Red", "Beige")
        self.assertEqual(uid, "UID-0001")
        self.assertFalse(self.db.merge_identities("UID-0001", "UID-0001"))

    def test_merge_into_identity_without_person_row(self):
        self.db.save_signature(self._sig("UID-0009"))
        self.assertTrue(self.db.merge_identities("UID-0009", "UID-0003"))
        uids = [r[0] for r in self.db.conn.execute("SELECT person_uid FROM persons")]
        self.assertEqual(uids, ["UID-0003"])
        self.assertEqual(self.db.get_stats()["total_signatures"], 1)

    def test_query_accessories_all_semantics_and_tolerant_colors(self):
        self.db.save_signature(self._sig("UID-0001", upper="Navy Blue", accessories=["backpack", "cap"]))
        self.db.save_signature(self._sig("UID-0002", upper="Red", accessories=["Backpack"]))
        self.db.save_signature(self._sig("UID-0003", upper="White", lower="Bare Legs / Shorts", accessories=["handbag"]))

        def uids(**kw):
            return sorted(r["person_uid"] for r in self.db.query_by_attributes(**kw))

        self.assertEqual(uids(accessories=["backpack", "cap"]), ["UID-0001"])
        self.assertEqual(uids(accessories=["backpack"]), ["UID-0001", "UID-0002"])
        self.assertEqual(uids(accessories=["hat"]), ["UID-0001"], "Synonyms normalise to the catalog name")
        self.assertEqual(uids(upper_color="Blue"), ["UID-0001"], "'Blue' must match 'Navy Blue'")
        self.assertEqual(uids(upper_color="Any", lower_color="None / Bare Legs"), ["UID-0003"])
        self.assertEqual(uids(upper_color="Red", accessories=["backpack"]), ["UID-0002"])

    def test_log_event_and_clear_wipe_profiles(self):
        self.db.log_event("TEST", "hello")
        self.db.save_identity_profile({"person_uid": "UID-0001", "embedding_model": MODEL_ID})
        self.assertTrue(self.db.clear_database())
        self.assertEqual(self.db.load_identity_profiles(), [])
        self.assertEqual(self.db.next_person_sequence(), 1)


class TestIdentityScoringHelpers(unittest.TestCase):
    """Pure scoring helpers."""

    def test_ciede2000_batch_matches_reference(self):
        rng = np.random.default_rng(3)
        labs = np.c_[rng.uniform(0, 100, 200), rng.uniform(-80, 80, 200), rng.uniform(-80, 80, 200)]
        for ref in labs[:10]:
            batch = ciede2000_batch(ref, labs)
            scalar = np.array([delta_e_ciede2000(ref, lab) for lab in labs])
            np.testing.assert_allclose(batch, scalar, atol=1e-9)

    def test_identity_score_renormalises_and_vetoes(self):
        cfg = IdentityConfig()
        self.assertAlmostEqual(identity_score(0.9, None, None, None, config=cfg), 0.9, places=6)
        both = identity_score(1.0, 0.0, None, None, config=cfg)
        expected = cfg.w_appearance / (cfg.w_appearance + cfg.w_upper)
        self.assertAlmostEqual(both, expected, places=6)
        self.assertEqual(identity_score(1.0, 1.0, 1.0, 1.0, veto=True, config=cfg), 0.0)

    def test_accessory_cue_semantics(self):
        cfg = IdentityConfig()
        k = 12
        idx_bp = 0  # backpack (catalog order)
        t_hits, t_checks = np.zeros(k), np.full(k, 12.0)
        t_hits[idx_bp] = 6
        id_hits = np.zeros((3, k))
        id_checks = np.full((3, k), 12.0)
        id_hits[0, idx_bp] = 5          # identity 0 also has a backpack -> agreement
        id_checks[2] = 2.0              # identity 2: barely checked -> uninformative
        s = accessory_cue(t_hits, t_checks, id_hits, id_checks, cfg)
        self.assertAlmostEqual(s[0], 1.0)
        self.assertAlmostEqual(s[1], 0.0)   # strongly absent in identity 1 -> disagreement
        self.assertTrue(np.isnan(s[2]))
        # Transient items (bottle) never count
        t2 = np.zeros(k)
        t2[9] = 10
        s2 = accessory_cue(t2, t_checks, np.zeros((1, k)), np.full((1, k), 12.0), cfg)
        self.assertTrue(np.isnan(s2[0]))

    def test_observation_quality(self):
        shape = (720, 1280, 3)
        big = estimate_observation_quality((500, 100, 600, 400), shape)
        small = estimate_observation_quality((500, 100, 520, 150), shape)
        cut = estimate_observation_quality((0, 100, 100, 400), shape)
        self.assertGreater(big, 0.9)
        self.assertLess(small, big)
        self.assertLess(cut, big)
        self.assertEqual(estimate_observation_quality((0, 0, 3, 3), shape), 0.0)
        kp = np.zeros((17, 3))
        kp[[0, 5, 6], 2] = 0.9  # webcam: head + shoulders visible, hips not
        webcam = estimate_observation_quality((300, 80, 900, 720), shape, keypoints_crop=kp)
        self.assertGreaterEqual(webcam, IdentityConfig().gallery_min_quality)


if __name__ == "__main__":
    unittest.main()
