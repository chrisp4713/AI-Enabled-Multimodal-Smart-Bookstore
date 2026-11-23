"""
Face login: enrollment and identification.            UC-01, UC-02

How it works
------------
1. A face DETECTOR finds the face in a camera frame (YOLO face weights if
   models/yolov8n-face.pt exists, otherwise OpenCV's built-in Haar cascade).
2. An EMBEDDER (FaceNet / InceptionResnetV1) turns the face crop into a
   512-number vector. Photos of the same person give vectors that point in
   nearly the same direction.
3. enroll() averages the vectors from several good photos into one "template"
   per user and saves it in biometrics/ (kept out of Git, NFR-07).
4. identify() embeds the login photo(s), compares them with every saved
   template using cosine similarity, and accepts the best match only if its
   score is above CONFIDENCE_THRESHOLD.

New users never need retraining: enrolling is just "compute a vector and
save it".

Public functions (for Alyssa's screens and the rest of the team)
----------------------------------------------------------------
    warm_up()                       load models once at app start
    capture_frames(count)           grab frames from the webcam
    check_frame_quality(frame)      (ok, reason) - for live feedback
    enroll(username, frames)        -> EnrollResult
    identify(frames)                -> IdentifyResult
    is_enrolled(username), delete_enrollment(username)
    lockout_remaining(), reset_lockout()
"""
from __future__ import annotations

import math
import os
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Configuration - tune these after testing with real teammates.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIOMETRICS_DIR = PROJECT_ROOT / "biometrics"                      # gitignored
YOLO_FACE_WEIGHTS = PROJECT_ROOT / "models" / "yolov8n-face.pt"   # gitignored

CONFIDENCE_THRESHOLD = 0.65   # cosine similarity needed to accept a login
MAX_FAILED_ATTEMPTS = 3       # failed recognitions before lockout (UC-02)
LOCKOUT_SECONDS = 60          # how long the lockout lasts

MIN_GOOD_SAMPLES = 3          # usable photos needed to enroll (FR-01.3)
MIN_FACE_PIXELS = 80          # face box width below this = "too far away"
MIN_BRIGHTNESS = 60           # mean gray level (0-255) below this = too dark
MAX_BRIGHTNESS = 210          # above this = washed out
MIN_SHARPNESS = 60.0          # variance of Laplacian below this = blurry

FACE_SIZE = 160               # face crops are resized to 160x160 for FaceNet


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------
class CameraError(Exception):
    """Camera missing, busy, or not delivering frames (NFR-10)."""


@dataclass
class EnrollResult:
    ok: bool
    message: str                # safe to show directly to the user
    samples_used: int = 0


@dataclass
class IdentifyResult:
    ok: bool
    username: str | None        # set on success; also set for suspended accounts
    confidence: float           # best similarity score, 0..1 (higher = closer)
    message: str                # safe to show directly to the user
    locked_out: bool = False
    seconds: float = 0.0        # how long identify() took (NFR-01: <= 3 s)


# --------------------------------------------------------------------------
# Model loading (lazy, so importing this file is fast and torch is optional
# until it is actually needed)
# --------------------------------------------------------------------------
_models_lock = threading.RLock()
_yolo = None
_yolo_checked = False
_haar = None
_embedder = None


def _load_yolo():
    """Return the YOLO face model, or None if weights/package are missing."""
    global _yolo, _yolo_checked
    with _models_lock:
        if not _yolo_checked:
            _yolo_checked = True
            if YOLO_FACE_WEIGHTS.exists():
                try:
                    from ultralytics import YOLO
                    _yolo = YOLO(str(YOLO_FACE_WEIGHTS))
                except ImportError:
                    warnings.warn("ultralytics is not installed; using Haar face detector.")
            else:
                warnings.warn(f"{YOLO_FACE_WEIGHTS} not found; using Haar face detector.")
        return _yolo


def _load_haar():
    global _haar
    with _models_lock:
        if _haar is None:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            _haar = cv2.CascadeClassifier(path)
        return _haar


def _load_embedder():
    global _embedder
    with _models_lock:
        if _embedder is None:
            from facenet_pytorch import InceptionResnetV1
            _embedder = InceptionResnetV1(pretrained="vggface2").eval()
        return _embedder


def warm_up() -> None:
    """Load every model now so the first login is fast (NFR-01).
    Call once at app start, ideally from a background thread."""
    _load_yolo()
    _load_haar()
    _load_embedder()


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
def capture_frames(count: int = 5, camera_index: int = 0,
                   delay: float = 0.3, warmup_frames: int = 5) -> list[np.ndarray]:
    """Grab `count` BGR frames from the webcam, `delay` seconds apart.
    Raises CameraError with a user-friendly message if anything goes wrong."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise CameraError("Camera not available. Check that it is connected "
                          "and not being used by another program.")
    try:
        for _ in range(warmup_frames):      # let auto-exposure settle
            cap.read()
        frames = []
        for _ in range(count):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise CameraError("The camera stopped sending images.")
            frames.append(frame)
            time.sleep(delay)
        return frames
    finally:
        cap.release()


# --------------------------------------------------------------------------
# Detection and quality checks
# --------------------------------------------------------------------------
def detect_faces(frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    """Return [(x1, y1, x2, y2, confidence), ...] for faces in a BGR frame."""
    yolo = _load_yolo()
    if yolo is not None:
        boxes = yolo(frame, verbose=False, conf=0.5)[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        return [(int(a), int(b), int(c), int(d), float(s))
                for (a, b, c, d), s in zip(xyxy, conf)]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = _load_haar().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(MIN_FACE_PIXELS // 2, MIN_FACE_PIXELS // 2))
    return [(int(x), int(y), int(x + w), int(y + h), 1.0) for x, y, w, h in rects]


def _crop_face(frame: np.ndarray, box, margin: float = 0.1) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box[:4]
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    return frame[y1:y2, x1:x2]


def _assess_frame(frame):
    """Return (ok, reason, face_box). `reason` is user-friendly when not ok."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return False, "No image from the camera.", None
    faces = detect_faces(frame)
    if len(faces) == 0:
        return False, "No face found - please face the camera.", None
    if len(faces) > 1:
        return False, "More than one face in view - one person at a time.", None
    box = faces[0]
    if box[2] - box[0] < MIN_FACE_PIXELS:
        return False, "Face is too small - move closer to the camera.", None
    crop = cv2.resize(_crop_face(frame, box), (FACE_SIZE, FACE_SIZE))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    if brightness < MIN_BRIGHTNESS:
        return False, "Image is too dark - find better lighting.", None
    if brightness > MAX_BRIGHTNESS:
        return False, "Image is too bright - avoid strong light behind or on you.", None
    if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_SHARPNESS:
        return False, "Image is blurry - hold still and try again.", None
    return True, "", box


def check_frame_quality(frame: np.ndarray) -> tuple[bool, str]:
    """(ok, reason) for one frame - handy for live hints on the camera screen."""
    ok, reason, _ = _assess_frame(frame)
    return ok, reason


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _embed_face(frame: np.ndarray, box) -> np.ndarray:
    """Face crop -> unit-length 512-number vector."""
    import torch
    crop = cv2.resize(_crop_face(frame, box), (FACE_SIZE, FACE_SIZE))
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.5) / 128.0                       # FaceNet's expected scaling
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        emb = _load_embedder()(tensor)[0].numpy()
    return _normalize(emb.astype(np.float32))


# --------------------------------------------------------------------------
# Template storage (one small .npy file per user, written atomically)
# --------------------------------------------------------------------------
def _template_path(username: str) -> Path:
    # quote() makes any username safe to use as a filename
    return BIOMETRICS_DIR / f"{quote(username, safe='')}.npy"


def _save_template(username: str, template: np.ndarray) -> None:
    BIOMETRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = _template_path(username)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, template)
    os.replace(tmp, path)             # atomic: never leaves a half-written file


def _load_templates() -> dict[str, np.ndarray]:
    from urllib.parse import unquote
    if not BIOMETRICS_DIR.exists():
        return {}
    templates = {}
    for path in BIOMETRICS_DIR.glob("*.npy"):
        if path.name.endswith(".tmp.npy"):
            continue
        templates[unquote(path.stem)] = np.load(path)
    return templates


def is_enrolled(username: str) -> bool:
    return bool(username) and _template_path(username).exists()


def delete_enrollment(username: str) -> bool:
    """Remove a user's saved face. Returns True if something was deleted."""
    path = _template_path(username)
    if path.exists():
        path.unlink()
        return True
    return False


# --------------------------------------------------------------------------
# Enrollment (UC-01)
# --------------------------------------------------------------------------
def enroll(username: str, frames: list[np.ndarray]) -> EnrollResult:
    """Register a user's face from several camera frames.

    Unusable frames (no face, too dark, blurry, ...) are skipped. At least
    MIN_GOOD_SAMPLES must pass or enrollment fails with an explanation.
    Enrolling an existing username replaces their old face data.
    """
    if not username or not username.strip():
        return EnrollResult(False, "A username is required.")

    embeddings, reasons = [], []
    for frame in frames:
        ok, reason, box = _assess_frame(frame)
        if not ok:
            reasons.append(reason)
            continue
        embeddings.append(_embed_face(frame, box))

    if len(embeddings) < MIN_GOOD_SAMPLES:
        problem = max(set(reasons), key=reasons.count) if reasons else "not enough photos"
        return EnrollResult(
            False,
            f"Only {len(embeddings)} of {len(frames)} photos were usable "
            f"(need {MIN_GOOD_SAMPLES}). Main problem: {problem}",
            len(embeddings))

    template = _normalize(np.mean(embeddings, axis=0))
    _save_template(username, template)
    return EnrollResult(True, "Face registered successfully.", len(embeddings))


# --------------------------------------------------------------------------
# Login (UC-02) with 3-failure lockout
# --------------------------------------------------------------------------
_state_lock = threading.Lock()
_failed_attempts = 0
_locked_until = 0.0


def lockout_remaining() -> int:
    """Seconds of lockout left (0 = not locked)."""
    with _state_lock:
        return max(0, math.ceil(_locked_until - time.monotonic()))


def reset_lockout() -> None:
    global _failed_attempts, _locked_until
    with _state_lock:
        _failed_attempts = 0
        _locked_until = 0.0


def _register_failure() -> tuple[bool, int]:
    """Count a failed recognition. Returns (now_locked, attempts_left)."""
    global _failed_attempts, _locked_until
    with _state_lock:
        _failed_attempts += 1
        if _failed_attempts >= MAX_FAILED_ATTEMPTS:
            _locked_until = time.monotonic() + LOCKOUT_SECONDS
            _failed_attempts = 0
            return True, 0
        return False, MAX_FAILED_ATTEMPTS - _failed_attempts


def _default_is_active():
    """Use Sahadat's core.accounts.is_active once it exists; until then
    treat every account as active so this module can be tested alone."""
    try:
        from core.accounts import is_active
        return is_active
    except ImportError:
        return lambda username: True


def identify(frames, threshold: float | None = None, is_active=None) -> IdentifyResult:
    """Work out who is in front of the camera.

    frames:    one frame or a list of frames (2-3 frames is a good balance)
    threshold: overrides CONFIDENCE_THRESHOLD
    is_active: function(username) -> bool; defaults to core.accounts.is_active

    Only a real mismatch counts toward the 3-failure lockout. Bad photos
    (no face, too dark) and suspended accounts just ask the user to retry.
    """
    start = time.perf_counter()
    result = _identify(frames, threshold, is_active)
    result.seconds = round(time.perf_counter() - start, 2)
    return result


def _identify(frames, threshold, is_active) -> IdentifyResult:
    if isinstance(frames, np.ndarray):
        frames = [frames]
    threshold = CONFIDENCE_THRESHOLD if threshold is None else threshold

    wait = lockout_remaining()
    if wait > 0:
        return IdentifyResult(False, None, 0.0,
                              f"Too many failed attempts. Try again in {wait} seconds.",
                              locked_out=True)

    templates = _load_templates()
    if not templates:
        return IdentifyResult(False, None, 0.0, "No faces are registered yet.")

    embeddings, reasons = [], []
    for frame in frames:
        ok, reason, box = _assess_frame(frame)
        if ok:
            embeddings.append(_embed_face(frame, box))
        else:
            reasons.append(reason)
    if not embeddings:
        problem = max(set(reasons), key=reasons.count) if reasons else "No image."
        return IdentifyResult(False, None, 0.0, problem)

    probe = _normalize(np.mean(embeddings, axis=0))
    scores = {user: float(np.dot(probe, tmpl)) for user, tmpl in templates.items()}
    best_user = max(scores, key=scores.get)
    best = scores[best_user]

    if best < threshold:
        locked, left = _register_failure()
        if locked:
            return IdentifyResult(
                False, None, best,
                f"Face not recognized. Login is locked for {LOCKOUT_SECONDS} seconds.",
                locked_out=True)
        return IdentifyResult(False, None, best,
                              f"Face not recognized. {left} attempt(s) left.")

    if not (is_active or _default_is_active())(best_user):
        return IdentifyResult(False, best_user, best,
                              "This account is suspended. Please contact an administrator.")

    reset_lockout()
    return IdentifyResult(True, best_user, best, f"Welcome back, {best_user}!")


# --------------------------------------------------------------------------
# Quick manual test from the project root:
#   python -m ai.biometrics enroll alice
#   python -m ai.biometrics identify
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    usage = "usage: python -m ai.biometrics enroll <username> | identify"
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "enroll":
            print("Look at the camera...")
            print(enroll(sys.argv[2], capture_frames(5)))
        elif len(sys.argv) == 2 and sys.argv[1] == "identify":
            print("Look at the camera...")
            print(identify(capture_frames(3)))
        else:
            print(usage)
    except CameraError as err:
        print(err)