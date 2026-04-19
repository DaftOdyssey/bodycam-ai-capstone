from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import queue
import signal
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import cv2
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import zmq

# Force TensorFlow to run on CPU only, freeing GPU resources for the slow brain (PyTorch)
# and ensuring compatibility on systems without an NVIDIA card.
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
)
logger = logging.getLogger("edge_pipeline")

# Platform detection (used by config defaults and helpers below)
_PLATFORM = platform.system()


@dataclass(slots=True)
class EdgeConfig:
    # Identity / transport
    camera_id: str = "cachyos_demo_bodycam"
    store_id: str = "demo_store_001"
    control_address: str = "tcp://127.0.0.1:5555"
    data_address: str = "tcp://127.0.0.1:5556"
    server_status_poll_seconds: float = 1.0

    # Video ingest
    # Set video_device_index to None to auto-detect the first available webcam.
    # Set it to an integer (0, 1, 2, …) to pin a specific device index.
    video_device_index: Optional[int] = None
    video_device_max_probe: int = 10

    # Set video_backend to None to auto-select based on OS.
    video_backend: Optional[int] = None
    frame_width: int = 1920
    frame_height: int = 1080
    fps: int = 30
    buffer_seconds: int = 20
    jpeg_quality: int = 85
    video_warmup_seconds: float = 5.0

    # Audio ingest / YAMNet
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_chunk_seconds: float = 1.0
    audio_dtype: str = "float32"
    audio_latency: str = "high"
    audio_queue_maxsize: int = 8
    # Set to None to auto-select the best default input device for the OS.
    audio_input_device: Optional[str] = None
    audio_min_rms_for_inference: float = 0.01

    # Ambient noise gating
    ambient_rms_window_seconds: float = 30.0
    ambient_rms_spike_multiplier: float = 1.5

    # Triggering / label tuning
    debug_top_k: int = 5
    monitored_label_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "Shout": 0.10,
            "Yell": 0.10,
            "Screaming": 0.08,
            "Bellow": 0.10,
            "Gunshot, gunfire": 0.05,
            "Glass": 0.15,
        }
    )

    # Incident lifecycle
    auto_clear_after_seconds: Optional[float] = None
    post_clear_tail_seconds: float = 2.0

    # Trigger parameters
    demo_mode: bool = False
    trigger_cooldown_seconds: float = 10.0

    # Network / buffering
    outbound_queue_maxsize: int = 4096
    compress_live_audio_to_int16: bool = True


@dataclass(slots=True)
class IncidentState:
    active: bool = False
    event_id: Optional[str] = None
    trigger_reason: Optional[str] = None
    started_ts: float = 0.0
    cleared_ts: float = 0.0


class EdgePipeline:
    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self.max_frames = self.config.fps * self.config.buffer_seconds
        self.max_audio_chunks = max(1, math.ceil(self.config.buffer_seconds / self.config.audio_chunk_seconds))

        self.stop_event = threading.Event()
        self.video_ready_event = threading.Event()
        self.buffer_lock = threading.Lock()
        self.incident_lock = threading.Lock()

        self.video_ring_buffer: Deque[tuple[float, bytes]] = deque(maxlen=self.max_frames)
        self.audio_ring_buffer: Deque[tuple[float, np.ndarray]] = deque(maxlen=self.max_audio_chunks)

        self.audio_inference_queue: queue.Queue[tuple[float, np.ndarray]] = queue.Queue(
            maxsize=self.config.audio_queue_maxsize
        )
        self.outbound_queue: queue.Queue[dict] = queue.Queue(maxsize=self.config.outbound_queue_maxsize)

        self.incident_state = IncidentState()
        self.server_clear_event = threading.Event()
        self.last_status_poll_ts = 0.0

        # Trigger tracking
        self.last_trigger_ts = 0.0
        self.consecutive_strong_detections = 0
        self.last_best_label = None

        # Ambient noise tracking
        ambient_maxlen = max(1, int(self.config.ambient_rms_window_seconds / self.config.audio_chunk_seconds))
        self.recent_rms: Deque[float] = deque(maxlen=ambient_maxlen)

        logger.info("Loading YAMNet model from TensorFlow Hub...")
        self.yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        self.yamnet_classes = self._load_class_names()
        self.monitored_targets = self._resolve_monitored_targets()

        self.zmq_context = zmq.Context()
        self.control_socket_lock = threading.Lock()
        self.data_socket = self._init_data_socket()
        self.control_socket = self._init_control_socket()

    # ---------------------------------------------------------------------
    # Platform-aware helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _resolve_video_backend(override: Optional[int] = None) -> int:
        if override is not None:
            return override
        if _PLATFORM == "Windows":
            return cv2.CAP_DSHOW
        if _PLATFORM == "Darwin":
            return cv2.CAP_AVFOUNDATION
        return cv2.CAP_V4L2

    @staticmethod
    def _detect_video_device(
            max_probe: int,
            backend: int,
            frame_width: int,
            frame_height: int,
    ) -> int:
        logger.info(
            "Auto-detecting webcam (probing indices 0-%d, backend=%s)...",
            max_probe - 1,
            backend,
        )
        for idx in range(max_probe):
            cap = cv2.VideoCapture(idx, backend)
            try:
                if not cap.isOpened():
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
                ok, _ = cap.read()
                if ok:
                    logger.info(
                        "Auto-detected webcam at index %d. Pin video_device_index=%d to skip probing next time.",
                        idx,
                        idx,
                    )
                    return idx
            finally:
                cap.release()
        raise RuntimeError(
            f"No working webcam found after probing {max_probe} indices. "
            "Connect a camera or set video_device_index explicitly in EdgeConfig."
        )

    @staticmethod
    def _resolve_audio_device(override: Optional[str]) -> Optional[str]:
        if override is not None:
            return override
        if _PLATFORM == "Linux":
            try:
                devices = sd.query_devices()
                names = [str(device.get("name", "")) for device in devices]
                if any("pulse" in name.lower() for name in names):
                    logger.info("Audio: using PulseAudio/PipeWire device ('pulse').")
                    return "pulse"
            except Exception:
                logger.exception("Audio device auto-detection failed; falling back to OS default device.")
            logger.info("Audio: 'pulse' not found; using OS default input device.")
            return None
        logger.info("Audio: using OS default input device (platform=%s).", _PLATFORM)
        return None

    def _wait_for_outbound_drain(self, timeout_seconds: float) -> None:
        deadline = time.time() + max(0.0, timeout_seconds)
        while time.time() < deadline:
            if self.outbound_queue.empty():
                return
            time.sleep(0.01)

    # ---------------------------------------------------------------------
    # Setup helpers
    # ---------------------------------------------------------------------
    def _load_class_names(self) -> list[str]:
        class_map_path = self.yamnet_model.class_map_path().numpy()
        if isinstance(class_map_path, bytes):
            class_map_path = class_map_path.decode("utf-8")

        class_names: list[str] = []
        with tf.io.gfile.GFile(class_map_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names.append(row["display_name"])

        logger.info("Loaded %d YAMNet class names.", len(class_names))
        return class_names

    def _resolve_monitored_targets(self) -> dict[int, tuple[str, float]]:
        label_to_index = {label: idx for idx, label in enumerate(self.yamnet_classes)}
        monitored_targets: dict[int, tuple[str, float]] = {}
        missing_labels: list[str] = []

        for label, threshold in self.config.monitored_label_thresholds.items():
            idx = label_to_index.get(label)
            if idx is None:
                missing_labels.append(label)
                continue
            monitored_targets[idx] = (label, threshold)

        if monitored_targets:
            pretty_targets = [
                f"{label}>={threshold:.2f}"
                for label, threshold in self.config.monitored_label_thresholds.items()
                if label in label_to_index
            ]
            logger.info("Monitoring YAMNet labels: %s", pretty_targets)
        else:
            logger.warning(
                "No configured YAMNet labels matched the model class map. "
                "Audio triggers will never fire until monitored_label_thresholds is fixed."
            )

        if missing_labels:
            logger.warning("Configured labels not found in YAMNet class map: %s", missing_labels)

        return monitored_targets

    def _init_data_socket(self) -> zmq.Socket:
        socket = self.zmq_context.socket(zmq.PUSH)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.config.data_address)
        logger.info("Data PUSH socket connected to %s", self.config.data_address)
        return socket

    def _init_control_socket(self) -> zmq.Socket:
        socket = self.zmq_context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 1000)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        socket.connect(self.config.control_address)
        logger.info("Control REQ socket connected to %s", self.config.control_address)
        return socket

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------
    def _json_bytes(self, payload: dict) -> bytes:
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def _pcm_bytes(self, waveform: np.ndarray) -> tuple[bytes, str]:
        if self.config.compress_live_audio_to_int16:
            clipped = np.clip(waveform, -1.0, 1.0)
            pcm = (clipped * 32767.0).astype(np.int16)
            return pcm.tobytes(), "int16"
        return waveform.astype(np.float32).tobytes(), "float32"

    def _enqueue_packet(self, packet: dict, *, drop_if_full: bool = False) -> bool:
        try:
            self.outbound_queue.put_nowait(packet)
            return True
        except queue.Full:
            if not drop_if_full:
                logger.warning(
                    "Outbound queue full while enqueueing critical packet type=%s event_id=%s",
                    packet.get("kind"),
                    packet.get("metadata", {}).get("event_id"),
                )
                return False

            logger.warning(
                "Outbound queue full; dropping non-critical packet type=%s event_id=%s",
                packet.get("kind"),
                packet.get("metadata", {}).get("event_id"),
            )
            return False

    def _reset_control_socket(self) -> None:
        try:
            self.control_socket.close(0)
        except Exception:
            logger.exception("Failed to close stale control socket during reset.")
        self.control_socket = self._init_control_socket()

    def _send_control_request(self, payload: dict) -> dict:
        with self.control_socket_lock:
            try:
                self.control_socket.send(self._json_bytes(payload))
                response_bytes = self.control_socket.recv()
                response = json.loads(response_bytes.decode("utf-8"))
                return response
            except Exception:
                logger.exception("Control request failed for payload type=%s", payload.get("type"))
                try:
                    self._reset_control_socket()
                except Exception:
                    logger.exception("Control socket reset failed.")
                return {"ack": False, "clear": False}

    def _make_base_metadata(self) -> dict:
        return {
            "camera_id": self.config.camera_id,
            "store_id": self.config.store_id,
        }

    # ---------------------------------------------------------------------
    # Incident state machine
    # ---------------------------------------------------------------------
    def _start_incident(self, trigger_reason: str, trigger_ts: float) -> None:
        with self.incident_lock:
            if self.incident_state.active:
                logger.info("Incident already active; suppressing duplicate start: %s", trigger_reason)
                return

            event_id = str(uuid.uuid4())
            self.incident_state = IncidentState(
                active=True,
                event_id=event_id,
                trigger_reason=trigger_reason,
                started_ts=trigger_ts,
                cleared_ts=0.0,
            )
            self.server_clear_event.clear()
            self.last_status_poll_ts = 0.0

        with self.buffer_lock:
            buffered_video = list(self.video_ring_buffer)
            buffered_audio = list(self.audio_ring_buffer)

        logger.info(
            "Incident START event_id=%s reason=%s | pre-roll video_frames=%d audio_chunks=%d",
            event_id,
            trigger_reason,
            len(buffered_video),
            len(buffered_audio),
        )

        start_payload = {
            "type": "incident_start",
            **self._make_base_metadata(),
            "event_id": event_id,
            "trigger_ts": trigger_ts,
            "trigger_reason": trigger_reason,
            "pre_roll_seconds": self.config.buffer_seconds,
            "audio_sample_rate": self.config.audio_sample_rate,
            "audio_channels": self.config.audio_channels,
            "frame_width": self.config.frame_width,
            "frame_height": self.config.frame_height,
            "capture_fps": self.config.fps,
        }

        response = self._send_control_request(start_payload)
        logger.info("Server response to incident_start: %s", response)

        # Tell the server an incident has begun before the pre-roll packets arrive.
        self._enqueue_packet(
            {
                "kind": "event",
                "metadata": {
                    **self._make_base_metadata(),
                    "event_id": event_id,
                    "packet_type": "incident_start",
                    "trigger_ts": trigger_ts,
                    "trigger_reason": trigger_reason,
                    "created_ts": time.time(),
                },
                "payload": b"",
            },
            drop_if_full=False,
        )

        # Send buffered audio first so the server can reconstruct an A/V clip.
        for chunk_idx, (chunk_ts, waveform) in enumerate(buffered_audio):
            pcm_bytes, encoded_dtype = self._pcm_bytes(waveform)
            self._enqueue_packet(
                {
                    "kind": "audio",
                    "metadata": {
                        **self._make_base_metadata(),
                        "event_id": event_id,
                        "packet_type": "pre_roll_audio",
                        "chunk_idx": chunk_idx,
                        "chunk_ts": chunk_ts,
                        "sample_rate": self.config.audio_sample_rate,
                        "channels": self.config.audio_channels,
                        "dtype": encoded_dtype,
                    },
                    "payload": pcm_bytes,
                },
                drop_if_full=False,
            )

        for frame_idx, (frame_ts, jpg_bytes) in enumerate(buffered_video):
            self._enqueue_packet(
                {
                    "kind": "video",
                    "metadata": {
                        **self._make_base_metadata(),
                        "event_id": event_id,
                        "packet_type": "pre_roll_video",
                        "frame_idx": frame_idx,
                        "frame_ts": frame_ts,
                    },
                    "payload": bytes(jpg_bytes),
                },
                drop_if_full=False,
            )

        if response.get("clear"):
            logger.info("Server immediately marked incident as cleared: event_id=%s", event_id)
            self.server_clear_event.set()

    def _finalize_incident(self, reason: str) -> None:
        with self.incident_lock:
            if not self.incident_state.active or self.incident_state.event_id is None:
                return
            event_id = self.incident_state.event_id
            trigger_reason = self.incident_state.trigger_reason
            started_ts = self.incident_state.started_ts
            self.incident_state.active = False
            self.incident_state.cleared_ts = time.time()
            cleared_ts = self.incident_state.cleared_ts

        logger.info("Incident END event_id=%s reason=%s", event_id, reason)

        # Put an explicit data-plane end marker on the wire first. The processing node
        # now waits for a quiet period before finalizing, which reduces the risk of
        # assembling media before trailing packets have arrived.
        self._enqueue_packet(
            {
                "kind": "event",
                "metadata": {
                    **self._make_base_metadata(),
                    "event_id": event_id,
                    "packet_type": "incident_end",
                    "ended_ts": cleared_ts,
                    "end_reason": reason,
                },
                "payload": b"",
            },
            drop_if_full=False,
        )
        self._wait_for_outbound_drain(timeout_seconds=1.0)

        end_payload = {
            "type": "incident_end",
            **self._make_base_metadata(),
            "event_id": event_id,
            "started_ts": started_ts,
            "ended_ts": cleared_ts,
            "end_reason": reason,
            "trigger_reason": trigger_reason,
        }
        response = self._send_control_request(end_payload)
        logger.info("Server response to incident_end: %s", response)

        with self.incident_lock:
            self.incident_state = IncidentState()
            self.server_clear_event.clear()

    def _poll_server_status_if_needed(self) -> None:
        with self.incident_lock:
            if not self.incident_state.active or self.incident_state.event_id is None:
                return
            now = time.time()
            if (now - self.last_status_poll_ts) < self.config.server_status_poll_seconds:
                return
            event_id = self.incident_state.event_id
            started_ts = self.incident_state.started_ts
            trigger_reason = self.incident_state.trigger_reason
            self.last_status_poll_ts = now

        payload = {
            "type": "incident_status",
            **self._make_base_metadata(),
            "event_id": event_id,
            "started_ts": started_ts,
            "trigger_reason": trigger_reason,
        }
        response = self._send_control_request(payload)
        logger.info("Server response to incident_status event_id=%s: %s", event_id, response)

        if response.get("clear"):
            self.server_clear_event.set()

    def _incident_active(self) -> bool:
        with self.incident_lock:
            return self.incident_state.active and self.incident_state.event_id is not None

    def _current_event_id(self) -> Optional[str]:
        with self.incident_lock:
            return self.incident_state.event_id

    # ---------------------------------------------------------------------
    # Audio callback and workers
    # ---------------------------------------------------------------------
    def _audio_callback(self, indata: np.ndarray, _frames: int, _time_info, status) -> None:
        if status:
            logger.warning("Audio callback status: %s", status)

        if self.stop_event.is_set():
            return

        chunk_ts = time.time()
        chunk = np.asarray(indata, dtype=np.float32).reshape(-1).copy()

        with self.buffer_lock:
            self.audio_ring_buffer.append((chunk_ts, chunk))

        try:
            self.audio_inference_queue.put_nowait((chunk_ts, chunk))
        except queue.Full:
            try:
                _ = self.audio_inference_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.audio_inference_queue.put_nowait((chunk_ts, chunk))
            except queue.Full:
                logger.warning("Audio inference queue saturated; dropping chunk.")

        if self._incident_active():
            event_id = self._current_event_id()
            if event_id is not None:
                pcm_bytes, encoded_dtype = self._pcm_bytes(chunk)
                self._enqueue_packet(
                    {
                        "kind": "audio",
                        "metadata": {
                            **self._make_base_metadata(),
                            "event_id": event_id,
                            "packet_type": "live_audio",
                            "chunk_ts": chunk_ts,
                            "sample_rate": self.config.audio_sample_rate,
                            "channels": self.config.audio_channels,
                            "dtype": encoded_dtype,
                        },
                        "payload": pcm_bytes,
                    },
                    drop_if_full=True,
                )

    def _audio_worker_loop(self) -> None:
        logger.info("Audio worker thread started.")

        while not self.stop_event.is_set():
            try:
                chunk_ts, waveform = self.audio_inference_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            if waveform.size == 0:
                continue

            rms = float(np.sqrt(np.mean(np.square(waveform))))

            # Track ambient noise before gating
            self.recent_rms.append(rms)
            ambient_noise_floor = float(np.mean(self.recent_rms)) if self.recent_rms else 0.0

            if rms < self.config.audio_min_rms_for_inference:
                # Reset consecutive count if it gets too quiet
                self.consecutive_strong_detections = 0
                self.last_best_label = None
                continue

            try:
                scores, _, _ = self.yamnet_model(waveform)
            except Exception:
                logger.exception("YAMNet inference failed for the audio chunk.")
                continue

            mean_scores = tf.reduce_mean(scores, axis=0).numpy()

            top_k = max(1, self.config.debug_top_k)
            top_indices = np.argsort(mean_scores)[-top_k:][::-1]

            if not self.monitored_targets:
                continue

            best_idx = max(self.monitored_targets, key=lambda idx: float(mean_scores[idx]))
            best_label, best_threshold = self.monitored_targets[best_idx]
            best_score = float(mean_scores[best_idx])

            decision_text = "NO trigger"
            triggered = False
            best_score_str = f"{best_label}={best_score:.2f}"

            top_summary_short = f"{self.yamnet_classes[top_indices[0]]}={float(mean_scores[top_indices[0]]):.2f}"

            # Smart trigger strategy
            if best_score >= best_threshold:
                if best_label == self.last_best_label:
                    self.consecutive_strong_detections += 1
                else:
                    self.consecutive_strong_detections = 1
                self.last_best_label = best_label

                # 2 consecutive OR (score 1.5x threshold + RMS spike above ambient)
                is_consecutive = self.consecutive_strong_detections >= 2
                rms_spike = max(
                    self.config.audio_min_rms_for_inference * 2.0,
                    ambient_noise_floor * self.config.ambient_rms_spike_multiplier
                )
                is_strong = (best_score >= best_threshold * 1.5) and (rms >= rms_spike)

                in_cooldown = (chunk_ts - self.last_trigger_ts) < self.config.trigger_cooldown_seconds

                if self._incident_active():
                    decision_text = "ALREADY ACTIVE"
                elif in_cooldown:
                    decision_text = "IN COOLDOWN"
                elif is_consecutive or is_strong:
                    triggered = True
                    decision_text = "TRIGGERED"
                else:
                    decision_text = "WAITING FOR CONSECUTIVE"
            else:
                self.consecutive_strong_detections = 0
                self.last_best_label = None
                decision_text = "NO trigger"

            # 🎤 Audio chunk | top: Yell=0.87 | monitored: Shout=0.41 (RMS=0.015, Ambient=0.010, SpikeReq=0.020) → NO trigger
            logger.info("🎤 Audio chunk | top: %s | monitored: %s (RMS=%.4f, Ambient=%.4f, SpikeReq=%.4f) → %s",
                        top_summary_short, best_score_str, rms, ambient_noise_floor,
                        max(self.config.audio_min_rms_for_inference * 2.0,
                            ambient_noise_floor * self.config.ambient_rms_spike_multiplier) if best_score >= best_threshold else 0.0,
                        decision_text)

            if triggered:
                self.last_trigger_ts = chunk_ts
                trigger_reason = (
                    f"{best_label} (confidence={best_score:.2f}, threshold={best_threshold:.2f}, ts={chunk_ts:.6f})"
                )
                self._start_incident(trigger_reason=trigger_reason, trigger_ts=chunk_ts)

                if self.config.demo_mode:
                    event_id = self._current_event_id() or "unknown"
                    print(f"\n🔥 INCIDENT TRIGGERED! {best_label} ({best_score:.2f}) | event_id={event_id}\n")

    # ---------------------------------------------------------------------
    # Video capture
    # ---------------------------------------------------------------------
    def _video_loop(self) -> None:
        backend = self._resolve_video_backend(self.config.video_backend)

        if self.config.video_device_index is None:
            try:
                device_index = self._detect_video_device(
                    max_probe=self.config.video_device_max_probe,
                    backend=backend,
                    frame_width=self.config.frame_width,
                    frame_height=self.config.frame_height,
                )
            except RuntimeError as exc:
                logger.error("Camera auto-detection failed: %s", exc)
                self.stop_event.set()
                return
        else:
            device_index = self.config.video_device_index
            logger.info(
                "Using pinned video device index %d (backend=%s).",
                device_index,
                backend,
            )

        logger.info(
            "Forcing MJPEG mode and setting resolution to %dx%d.",
            self.config.frame_width,
            self.config.frame_height,
        )

        def _open_cap() -> Optional[cv2.VideoCapture]:
            cap = cv2.VideoCapture(device_index, backend)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            if not cap.isOpened():
                cap.release()
                return None
            return cap

        cap = _open_cap()
        if cap is None:
            logger.error(
                "Could not open video device index %d (backend=%s).",
                device_index,
                backend,
            )
            self.stop_event.set()
            return

        logger.info(
            "[%s] Video ingestion active. Buffering up to %ds (%d frames max).",
            self.config.camera_id,
            self.config.buffer_seconds,
            self.max_frames,
        )
        self.video_ready_event.set()

        consecutive_failures = 0
        max_consecutive_failures = 10

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    logger.warning(
                        "Video capture returned no frame (failure %d/%d).",
                        consecutive_failures,
                        max_consecutive_failures,
                    )
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(
                            "Too many consecutive read failures; attempting reconnect to device index %d.",
                            device_index,
                        )
                        cap.release()
                        time.sleep(1.0)
                        cap = _open_cap()
                        if cap is None:
                            logger.error("Reconnect failed for device index %d.", device_index)
                            self.stop_event.set()
                            return
                        consecutive_failures = 0
                        logger.info("Reconnected to device index %d.", device_index)
                    else:
                        time.sleep(0.05)
                    continue

                consecutive_failures = 0
                ok, jpg_buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality],
                )
                if not ok:
                    logger.warning("Failed to JPEG-encode frame; skipping.")
                    continue

                frame_ts = time.time()
                jpg_bytes = jpg_buffer.tobytes()

                with self.buffer_lock:
                    self.video_ring_buffer.append((frame_ts, jpg_bytes))

                if self._incident_active():
                    event_id = self._current_event_id()
                    if event_id is not None:
                        self._enqueue_packet(
                            {
                                "kind": "video",
                                "metadata": {
                                    **self._make_base_metadata(),
                                    "event_id": event_id,
                                    "packet_type": "live_video",
                                    "frame_ts": frame_ts,
                                },
                                "payload": jpg_bytes,
                            },
                            drop_if_full=True,
                        )
        except Exception:
            logger.exception("Unhandled exception in video loop.")
            self.stop_event.set()
        finally:
            cap.release()
            logger.info("Video capture released.")

    # ---------------------------------------------------------------------
    # Network sender and control polling
    # ---------------------------------------------------------------------
    def _outbound_sender_loop(self) -> None:
        logger.info("Outbound sender thread started.")

        while not self.stop_event.is_set() or not self.outbound_queue.empty():
            try:
                packet = self.outbound_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            kind = packet["kind"]
            metadata = packet["metadata"]
            payload = packet["payload"]

            try:
                self.data_socket.send_multipart(
                    [
                        kind.encode("utf-8"),
                        self._json_bytes(metadata),
                        payload,
                    ]
                )
            except Exception:
                logger.exception(
                    "Failed to send outbound packet type=%s event_id=%s",
                    kind,
                    metadata.get("event_id"),
                )

    def _control_poll_loop(self) -> None:
        logger.info("Control poll thread started.")

        while not self.stop_event.is_set():
            if self._incident_active():
                self._poll_server_status_if_needed()

                if self.config.auto_clear_after_seconds is not None:
                    with self.incident_lock:
                        age = time.time() - self.incident_state.started_ts if self.incident_state.active else 0.0
                    if age >= self.config.auto_clear_after_seconds:
                        logger.info(
                            "auto_clear_after_seconds reached (%.2fs); treating as cleared for demo.",
                            age,
                        )
                        self.server_clear_event.set()

                if self.server_clear_event.is_set():
                    if self.config.post_clear_tail_seconds > 0:
                        logger.info(
                            "Server clear observed. Sending %.2fs tail before ending incident.",
                            self.config.post_clear_tail_seconds,
                        )
                        time.sleep(self.config.post_clear_tail_seconds)
                    self._finalize_incident(reason="server_clear")
            time.sleep(0.10)

    # ---------------------------------------------------------------------
    # Audio stream and lifecycle
    # ---------------------------------------------------------------------
    def _audio_stream_loop(self) -> None:
        blocksize = int(self.config.audio_sample_rate * self.config.audio_chunk_seconds)
        audio_device = self._resolve_audio_device(self.config.audio_input_device)
        logger.info(
            "[%s] Audio trigger active at %d Hz, %d channel(s), %.2fs chunks, device=%s.",
            self.config.camera_id,
            self.config.audio_sample_rate,
            self.config.audio_channels,
            self.config.audio_chunk_seconds,
            audio_device if audio_device is not None else "<OS default>",
        )

        try:
            with sd.InputStream(
                    samplerate=self.config.audio_sample_rate,
                    channels=self.config.audio_channels,
                    dtype=self.config.audio_dtype,
                    callback=self._audio_callback,
                    blocksize=blocksize,
                    latency=self.config.audio_latency,
                    device=audio_device,
            ):
                while not self.stop_event.is_set():
                    time.sleep(0.25)
        except Exception:
            logger.exception("Audio input stream failed.")
            self.stop_event.set()

    def _install_signal_handlers(self) -> None:
        def _handler(signum, _frame) -> None:
            logger.info("Received signal %s. Beginning shutdown.", signum)
            self.stop_event.set()

        candidates = [signal.SIGINT]
        if hasattr(signal, "SIGTERM"):
            candidates.append(signal.SIGTERM)

        for sig in candidates:
            try:
                signal.signal(sig, _handler)
            except Exception:
                logger.warning("Could not register handler for signal %s.", sig)

    def shutdown(self) -> None:
        self.stop_event.set()

        if self._incident_active():
            self._finalize_incident(reason="shutdown")

        try:
            self.data_socket.close(0)
        except Exception:
            logger.exception("Failed to close data socket cleanly.")

        try:
            self.control_socket.close(0)
        except Exception:
            logger.exception("Failed to close control socket cleanly.")

        try:
            self.zmq_context.term()
        except Exception:
            logger.exception("Failed to terminate ZMQ context cleanly.")

        logger.info("Shutdown complete.")

    def run(self) -> None:
        self._install_signal_handlers()

        video_thread = threading.Thread(target=self._video_loop, name="video-loop", daemon=True)
        audio_worker_thread = threading.Thread(target=self._audio_worker_loop, name="audio-worker", daemon=True)
        outbound_sender_thread = threading.Thread(
            target=self._outbound_sender_loop,
            name="outbound-sender",
            daemon=True,
        )
        control_poll_thread = threading.Thread(
            target=self._control_poll_loop,
            name="control-poll",
            daemon=True,
        )

        video_thread.start()
        audio_worker_thread.start()
        outbound_sender_thread.start()
        control_poll_thread.start()

        logger.info("Waiting for video capture to warm up...")
        self.video_ready_event.wait(timeout=max(3.0, self.config.video_warmup_seconds + 1.0))
        logger.info(
            "Warming video buffer for %.1f seconds before arming audio trigger...",
            self.config.video_warmup_seconds,
        )
        time.sleep(self.config.video_warmup_seconds)

        try:
            self._audio_stream_loop()
        finally:
            self.shutdown()
            video_thread.join(timeout=2.0)
            audio_worker_thread.join(timeout=2.0)
            outbound_sender_thread.join(timeout=2.0)
            control_poll_thread.join(timeout=2.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Pipeline")
    parser.add_argument("--demo-mode", action="store_true", help="Enable demo mode with large prints")
    parser.add_argument("--video-device", type=int, default=None, help="Pin video device index")
    parser.add_argument("--cooldown", type=float, default=None, help="Cooldown between triggers in seconds")
    parser.add_argument("--rms-gate", type=float, default=None, help="Minimum RMS for inference")
    parser.add_argument("--thresholds", type=str, default=None, help="Comma separated Label=Threshold pairs")
    parser.add_argument("--ambient-window", type=float, default=None, help="Ambient noise window in seconds")
    parser.add_argument("--ambient-spike", type=float, default=None, help="Ambient noise spike multiplier")
    parser.add_argument("--resolution", type=str, default="1080p", choices=["1080p", "720p"],
                        help="Video resolution (1080p or 720p)")

    args = parser.parse_args()

    config = EdgeConfig()

    # Environment variables fallback
    if os.environ.get("DEMO_MODE", "").lower() in ("1", "true", "yes"):
        config.demo_mode = True
    if "VIDEO_DEVICE" in os.environ:
        config.video_device_index = int(os.environ["VIDEO_DEVICE"])
    if "TRIGGER_COOLDOWN" in os.environ:
        config.trigger_cooldown_seconds = float(os.environ["TRIGGER_COOLDOWN"])
    if "RMS_GATE" in os.environ:
        config.audio_min_rms_for_inference = float(os.environ["RMS_GATE"])
    if "AMBIENT_WINDOW" in os.environ:
        config.ambient_rms_window_seconds = float(os.environ["AMBIENT_WINDOW"])
    if "AMBIENT_SPIKE" in os.environ:
        config.ambient_rms_spike_multiplier = float(os.environ["AMBIENT_SPIKE"])
    if "MONITORED_THRESHOLDS" in os.environ:
        for pair in os.environ["MONITORED_THRESHOLDS"].split(","):
            if "=" in pair:
                label, val = pair.split("=")
                config.monitored_label_thresholds[label.strip()] = float(val.strip())

    # CLI args override env vars
    if args.demo_mode:
        config.demo_mode = True
    if args.video_device is not None:
        config.video_device_index = args.video_device
    if args.cooldown is not None:
        config.trigger_cooldown_seconds = args.cooldown
    if args.rms_gate is not None:
        config.audio_min_rms_for_inference = args.rms_gate
    if args.ambient_window is not None:
        config.ambient_rms_window_seconds = args.ambient_window
    if args.ambient_spike is not None:
        config.ambient_rms_spike_multiplier = args.ambient_spike
    if args.thresholds is not None:
        for pair in args.thresholds.split(","):
            if "=" in pair:
                label, val = pair.split("=")
                config.monitored_label_thresholds[label.strip()] = float(val.strip())

    if args.resolution == "720p":
        config.frame_width = 1280
        config.frame_height = 720
    else:
        config.frame_width = 1920
        config.frame_height = 1080

    pipeline = EdgePipeline(config)
    pipeline.run()