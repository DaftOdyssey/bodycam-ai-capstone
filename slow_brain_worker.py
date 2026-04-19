from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psycopg2

try:
    from faster_whisper import WhisperModel

    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import XCLIPProcessor, XCLIPModel, pipeline

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
)
logger = logging.getLogger("slow_brain")


@dataclass
class SlowBrainConfig:
    # X-CLIP semantic video scoring
    xclip_model_name: str = "microsoft/xclip-base-patch32"
    xclip_num_frames: int = 8

    xclip_candidate_labels: list[str] = field(
        default_factory=lambda: [
            "a calm conversation",
            "normal shopping activity",
            "a person standing still",
            "people arguing",
            "a person shouting angrily",
            "a physical fight",
            "someone threatening another person",
            "a chaotic emergency situation",
        ]
    )

    xclip_label_risk_weights: dict[str, float] = field(
        default_factory=lambda: {
            "a calm conversation": 0.0,
            "normal shopping activity": 0.0,
            "a person standing still": 0.05,
            "people arguing": 0.55,
            "a person shouting angrily": 0.65,
            "a physical fight": 0.95,
            "someone threatening another person": 0.90,
            "a chaotic emergency situation": 0.85,
        }
    )

    # Fusion between old heuristic video score and X-CLIP semantic score
    xclip_fusion_weight: float = 0.65
    heuristic_fusion_weight: float = 0.35

    # Directory scanning
    events_dir: str = "events"
    poll_interval_seconds: float = 2.0

    # Transcription (Faster Whisper)
    whisper_model_size: str = "base.en"
    whisper_device: str = "auto"
    whisper_compute_type: str = "default"

    # Video Scoring (Heuristics for now, X-CLIP placeholder)
    video_motion_threshold: float = 15.0
    video_darkness_threshold: float = 40.0
    video_blur_threshold: float = 100.0

    # Risk Assessment Thresholds
    risk_weights: dict[str, float] = field(
        default_factory=lambda: {
            "audio": 0.4,
            "video": 0.3,
            "transcript": 0.3,
        }
    )

    high_risk_threshold: float = 0.75
    medium_risk_threshold: float = 0.50

    # PostgreSQL / pgvector settings
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "bodycam_db"
    db_user: str = "postgres"
    db_password: str = ""
    enable_pgvector: bool = False
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Concurrency
    max_workers: int = 2

    demo_mode: bool = False


class ModalityAnalyzer:
    """Base class or namespace for different analysis modalities."""

    @staticmethod
    def analyze_audio(event_dir: Path) -> dict[str, Any]:
        """
        Analyze the concatenated audio file for volume peaks, sustained loudness, etc.
        For this capstone, we simulate or do basic feature extraction.
        """
        audio_path = event_dir / "incident_audio.wav"
        if not audio_path.exists():
            return {"status": "missing", "risk_score": 0.0, "details": "No audio file found."}

        try:
            import soundfile as sf
            data, samplerate = sf.read(str(audio_path))

            if len(data) == 0:
                return {"status": "empty", "risk_score": 0.0, "details": "Audio file is empty."}

            # Basic RMS calculation
            rms = float(np.sqrt(np.mean(np.square(data))))

            # Map RMS to a 0.0 - 1.0 risk score (heuristic mapping)
            # Typically RMS of normalized float32 audio maxes at 1.0.
            # Normal speech is around 0.05 - 0.15. Shouting might be 0.3+.
            risk_score = min(1.0, rms * 3.0)

            peak = float(np.max(np.abs(data)))

            return {
                "status": "success",
                "risk_score": risk_score,
                "rms": rms,
                "peak": peak,
                "details": f"Analyzed {len(data) / samplerate:.1f}s of audio."
            }
        except Exception as e:
            logger.exception("Audio analysis failed.")
            return {"status": "error", "risk_score": 0.0, "details": str(e)}

    @staticmethod
    def sample_video_frames_for_xclip(video_path: Path, num_frames: int) -> list[np.ndarray]:
        """
        Sample RGB frames from a video for X-CLIP.

        X-CLIP base was trained with 8 frames, so the default should stay at 8 unless
        you intentionally switch to a different checkpoint.
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return []

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if frame_count <= 0:
                return []

            indices = np.linspace(0, max(0, frame_count - 1), num=num_frames, dtype=np.int64)

            frames: list[np.ndarray] = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame_bgr = cap.read()

                if not ret or frame_bgr is None:
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            if not frames:
                return []

            # Pad if the video had fewer readable frames than requested.
            while len(frames) < num_frames:
                frames.append(frames[-1].copy())

            return frames[:num_frames]

        finally:
            cap.release()

    @staticmethod
    def analyze_video_with_xclip(
            video_path: Path,
            config: SlowBrainConfig,
            xclip_processor: Any | None,
            xclip_model: Any | None,
            xclip_device: str = "cpu",
    ) -> dict[str, Any]:
        """
        Run X-CLIP zero-shot semantic scoring on the incident video.

        This compares the sampled video frames against candidate text labels and
        converts those label probabilities into a semantic video risk score.
        """
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            return {
                "status": "skipped",
                "risk_score": 0.0,
                "details": "torch and/or transformers not installed.",
            }

        if xclip_processor is None or xclip_model is None:
            return {
                "status": "skipped",
                "risk_score": 0.0,
                "details": "X-CLIP model or processor not loaded.",
            }

        if not video_path.exists():
            return {
                "status": "missing",
                "risk_score": 0.0,
                "details": f"Video file not found: {video_path}",
            }

        try:
            frames = ModalityAnalyzer.sample_video_frames_for_xclip(
                video_path=video_path,
                num_frames=config.xclip_num_frames,
            )

            if not frames:
                return {
                    "status": "empty",
                    "risk_score": 0.0,
                    "details": "Could not sample frames for X-CLIP.",
                }

            candidate_labels = list(config.xclip_candidate_labels)

            inputs = xclip_processor(
                text=candidate_labels,
                videos=frames,
                return_tensors="pt",
                padding=True,
            )

            inputs = {
                key: value.to(xclip_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

            xclip_model.eval()

            with torch.inference_mode():
                outputs = xclip_model(**inputs)

            probs_tensor = outputs.logits_per_video.softmax(dim=1)[0]
            probs = probs_tensor.detach().cpu().numpy().tolist()

            label_scores = {
                label: float(score)
                for label, score in zip(candidate_labels, probs)
            }

            risk_score = 0.0

            for label, probability in label_scores.items():
                label_risk_weight = config.xclip_label_risk_weights.get(label, 0.0)
                risk_score += probability * label_risk_weight

            risk_score = float(max(0.0, min(1.0, risk_score)))

            if not label_scores:
                return {
                    "status": "empty",
                    "risk_score": 0.0,
                    "details": "X-CLIP produced no label scores.",
                }

            top_label, top_label_score = max(label_scores.items(), key=lambda item: item[1])

            return {
                "status": "success",
                "risk_score": risk_score,
                "top_label": top_label,
                "top_label_score": top_label_score,
                "label_scores": label_scores,
                "num_frames": len(frames),
                "model": config.xclip_model_name,
            }

        except Exception as e:
            logger.exception("X-CLIP video analysis failed.")
            return {
                "status": "error",
                "risk_score": 0.0,
                "details": str(e),
            }

    @staticmethod
    def analyze_video(
            event_dir: Path,
            config: SlowBrainConfig,
            xclip_processor: Any | None = None,
            xclip_model: Any | None = None,
            xclip_device: str = "cpu",
    ) -> dict[str, Any]:
        """
        Analyze the incident video using:
        1. Basic visual heuristics: motion, darkness, blur
        2. Optional X-CLIP semantic video-text scoring

        The final video risk score is a fusion of both.
        """
        video_path = event_dir / "incident_video.mp4"

        if not video_path.exists():
            return {
                "status": "missing",
                "risk_score": 0.0,
                "details": "No video file found.",
            }

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return {
                "status": "error",
                "risk_score": 0.0,
                "details": "Could not open video.",
            }

        frame_count = 0
        total_motion = 0.0
        dark_frames = 0
        blurry_frames = 0
        prev_gray = None

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                mean_val = float(np.mean(gray))
                if mean_val < config.video_darkness_threshold:
                    dark_frames += 1

                variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if variance < config.video_blur_threshold:
                    blurry_frames += 1

                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    motion = float(np.mean(diff))
                    total_motion += motion

                prev_gray = gray

                # Keep slow brain bounded.
                if frame_count >= 300:
                    break

        finally:
            cap.release()

        if frame_count == 0:
            return {
                "status": "empty",
                "risk_score": 0.0,
                "details": "Video has no frames.",
            }

        avg_motion = total_motion / max(1, frame_count - 1)
        dark_ratio = dark_frames / frame_count
        blur_ratio = blurry_frames / frame_count

        motion_risk = min(1.0, avg_motion / config.video_motion_threshold)
        obscurity_risk = min(1.0, dark_ratio + blur_ratio)

        heuristic_risk = float((motion_risk * 0.7) + (obscurity_risk * 0.3))

        xclip_results = ModalityAnalyzer.analyze_video_with_xclip(
            video_path=video_path,
            config=config,
            xclip_processor=xclip_processor,
            xclip_model=xclip_model,
            xclip_device=xclip_device,
        )

        if xclip_results.get("status") == "success":
            xclip_risk = float(xclip_results.get("risk_score", 0.0))

            final_video_risk = (
                    xclip_risk * config.xclip_fusion_weight
                    + heuristic_risk * config.heuristic_fusion_weight
            )

            model_used = "xclip+heuristics"

        else:
            final_video_risk = heuristic_risk
            model_used = "heuristics_only"

        final_video_risk = float(max(0.0, min(1.0, final_video_risk)))

        return {
            "status": "success",
            "risk_score": final_video_risk,
            "model_used": model_used,
            "heuristic_risk_score": heuristic_risk,
            "xclip": xclip_results,
            "avg_motion": avg_motion,
            "dark_ratio": dark_ratio,
            "blur_ratio": blur_ratio,
            "frames_analyzed": frame_count,
        }

    @staticmethod
    def extract_transcript(event_dir: Path, whisper_model: Any, nlp_classifier=None) -> dict[str, Any]:
        """
        Run Faster Whisper on the audio file.
        """
        if not HAS_WHISPER or whisper_model is None:
            return {"status": "skipped", "risk_score": 0.0, "details": "Whisper not installed or model not loaded."}

        audio_path = event_dir / "incident_audio.wav"
        if not audio_path.exists():
            return {"status": "missing", "risk_score": 0.0, "details": "No audio file for transcription."}

        try:
            segments, info = whisper_model.transcribe(str(audio_path), beam_size=5)

            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)

            full_text = " ".join(text_parts).strip()

            risk_score = 0.0
            model_used = "none"

            if len(full_text.strip()) > 5 and nlp_classifier is not None:
                try:
                    candidate_labels = [
                        "a physical threat",
                        "an argument",
                        "a request for help",
                        "a calm conversation"
                    ]

                    result = nlp_classifier(full_text, candidate_labels, multi_label=False)

                    # Compute risk score by summing probabilities of tension/risk labels
                    probs = dict(zip(result['labels'], result['scores']))
                    risk_score = min(1.0,
                                     probs.get("a physical threat", 0.0) + probs.get("an argument", 0.0) + probs.get(
                                         "a request for help", 0.0))
                    model_used = "zero-shot-classification"
                except Exception as e:
                    logger.warning(f"NLP classification failed, falling back to keywords: {e}")
                    nlp_classifier = None  # trigger fallback

            if nlp_classifier is None and len(full_text.strip()) > 0:
                # Fallback: Simple keyword matching
                high_risk_keywords = ["help", "stop", "gun", "knife", "shoot", "kill", "police", "drop it"]
                lower_text = full_text.lower()
                keyword_hits = sum(1 for kw in high_risk_keywords if kw in lower_text)
                risk_score = min(1.0, keyword_hits * 0.25)
                model_used = "keyword-matching"

            return {
                "status": "success" if full_text else "empty",
                "risk_score": risk_score,
                "text": full_text,
                "language": info.language,
                "language_probability": info.language_probability,
                "model": model_used
            }

        except Exception as e:
            logger.exception("Transcription failed.")
            return {"status": "error", "risk_score": 0.0, "details": str(e)}


class SlowBrainWorker:
    def __init__(self, config: SlowBrainConfig):
        self.config = config
        self.events_dir = Path(config.events_dir)
        self.events_dir.mkdir(parents=True, exist_ok=True)

        self.whisper_model = None
        self.embedding_model = None
        self.xclip_processor = None
        self.xclip_model = None
        self.xclip_device = "cpu"
        self.nlp_classifier = None

        self._init_models()
        self._init_db()

    def _init_models(self):
        logger.info("Initializing ML models...")

        if HAS_WHISPER:
            try:
                logger.info(f"Loading Whisper model '{self.config.whisper_model_size}'...")
                self.whisper_model = WhisperModel(
                    self.config.whisper_model_size,
                    device=self.config.whisper_device,
                    compute_type=self.config.whisper_compute_type
                )
                logger.info("Whisper loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
        else:
            logger.warning("faster_whisper not installed. Transcription disabled.")

        if HAS_TORCH and HAS_TRANSFORMERS:
            try:
                logger.info(f"Loading X-CLIP model '{self.config.xclip_model_name}'...")

                self.xclip_processor = XCLIPProcessor.from_pretrained(
                    self.config.xclip_model_name
                )

                self.xclip_model = XCLIPModel.from_pretrained(
                    self.config.xclip_model_name
                )

                self.xclip_device = "cuda" if torch.cuda.is_available() else "cpu"
                self.xclip_model.to(self.xclip_device)
                self.xclip_model.eval()

                logger.info(f"X-CLIP loaded successfully on {self.xclip_device}.")

            except Exception as e:
                logger.error(f"Failed to load X-CLIP: {e}")
                self.xclip_model = None
                self.xclip_processor = None
                self.xclip_device = "cpu"

            try:
                logger.info("Loading NLP zero-shot classifier 'typeform/distilbert-base-uncased-mnli'...")

                device_idx = 0 if self.xclip_device == "cuda" else -1

                self.nlp_classifier = pipeline(
                    "zero-shot-classification",
                    model="typeform/distilbert-base-uncased-mnli",
                    device=device_idx,
                )

                logger.info("NLP classifier loaded successfully.")

            except Exception as e:
                logger.error(f"Failed to load NLP classifier: {e}")
                self.nlp_classifier = None

        else:
            logger.warning("torch and/or transformers not installed. X-CLIP and NLP classifier disabled.")

        if self.config.enable_pgvector and HAS_SENTENCE_TRANSFORMERS:
            try:
                logger.info(f"Loading embedding model '{self.config.embedding_model_name}'...")
                self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
                logger.info("Embedding model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
        elif self.config.enable_pgvector:
            logger.warning("pgvector enabled but sentence_transformers not installed.")

    def _init_db(self):
        if not self.config.enable_pgvector:
            return

        logger.info("Connecting to PostgreSQL...")
        try:
            self.conn = psycopg2.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                dbname=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password
            )

            with self.conn.cursor() as cur:
                # Ensure pgvector extension exists
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Create incident table
                cur.execute("""
                            CREATE TABLE IF NOT EXISTS incidents
                            (
                                id
                                VARCHAR
                                PRIMARY
                                KEY,
                                camera_id
                                VARCHAR,
                                start_time
                                TIMESTAMP,
                                end_time
                                TIMESTAMP,
                                trigger_reason
                                VARCHAR,
                                final_risk_score
                                FLOAT,
                                risk_level
                                VARCHAR,
                                transcript
                                TEXT,
                                narrative
                                TEXT,
                                embedding
                                vector
                            (
                                384
                            )
                                );
                            """)
            self.conn.commit()
            logger.info("Database initialized successfully.")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self.config.enable_pgvector = False

    def get_pending_events(self) -> list[Path]:
        """Find event directories that have finalized media but no slow_brain_results.json."""
        pending = []
        for event_dir in self.events_dir.iterdir():
            if not event_dir.is_dir():
                continue

            # Check if the incident is closed (harness creates a marker or just wait for concatenations)
            # For simplicity, we assume if concatenated.mp4 exists, it's ready for analysis.
            # In a strict pipeline, we'd look for an 'incident_closed.json' marker.

            has_media = (event_dir / "incident_video.mp4").exists() or (event_dir / "incident_audio.wav").exists()
            has_results = (event_dir / "slow_brain_results.json").exists()
            is_failed = (event_dir / "slow_brain_failed.marker").exists()

            if has_media and not has_results and not is_failed:
                # To prevent race conditions, check file modification time.
                # If modified in the last 5 seconds, it might still be writing.
                mp4_mtime = (event_dir / "incident_video.mp4").stat().st_mtime if (
                            event_dir / "incident_video.mp4").exists() else 0
                wav_mtime = (event_dir / "incident_audio.wav").stat().st_mtime if (
                            event_dir / "incident_audio.wav").exists() else 0

                latest_mtime = max(mp4_mtime, wav_mtime)
                if (time.time() - latest_mtime) > 5.0:
                    pending.append(event_dir)

        return pending

    def process_event(self, event_dir: Path):
        event_id = event_dir.name
        logger.info(f"Processing event: {event_id}")

        try:
            # Load metadata if available
            metadata = {}
            metadata_path = event_dir / "session_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            # 1. Analyze Modalities
            audio_results = ModalityAnalyzer.analyze_audio(event_dir)
            video_results = ModalityAnalyzer.analyze_video(
                event_dir,
                self.config,
                xclip_processor=getattr(self, 'xclip_processor', None),
                xclip_model=getattr(self, 'xclip_model', None),
                xclip_device=getattr(self, 'xclip_device', 'cpu')
            )
            transcript_results = ModalityAnalyzer.extract_transcript(
                event_dir,
                self.whisper_model,
                nlp_classifier=getattr(self, 'nlp_classifier', None)
            )

            # 2. Dynamic Weighting & Risk Fusion
            active_modalities = 0
            total_weight = 0.0
            weighted_score = 0.0

            if audio_results["status"] == "success":
                weighted_score += audio_results["risk_score"] * self.config.risk_weights["audio"]
                total_weight += self.config.risk_weights["audio"]
                active_modalities += 1

            if video_results["status"] == "success":
                weighted_score += video_results["risk_score"] * self.config.risk_weights["video"]
                total_weight += self.config.risk_weights["video"]
                active_modalities += 1

            if transcript_results["status"] in ("success", "empty"):
                weighted_score += transcript_results["risk_score"] * self.config.risk_weights["transcript"]
                total_weight += self.config.risk_weights["transcript"]
                active_modalities += 1

            if total_weight > 0:
                final_risk_score = weighted_score / total_weight
            else:
                final_risk_score = 0.0

            # Determine Risk Level
            if final_risk_score >= self.config.high_risk_threshold:
                risk_level = "HIGH"
            elif final_risk_score >= self.config.medium_risk_threshold:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            # 3. Narrative Generation
            trigger_reason = metadata.get("trigger_reason", "Unknown trigger")
            transcript_text = transcript_results.get("text", "")

            narrative = f"Incident triggered by: {trigger_reason}. "
            narrative += f"Analysis computed a risk score of {final_risk_score:.2f} ({risk_level}). "

            if transcript_text:
                narrative += f"Transcript highlights: '{transcript_text[:200]}...' "
            else:
                narrative += "No speech detected. "

            if video_results["status"] == "success":
                narrative += f"Video showed avg motion {video_results.get('avg_motion', 0.0):.1f}. "

            # 4. Save Results
            results = {
                "event_id": event_id,
                "processed_ts": time.time(),
                "final_risk_score": final_risk_score,
                "risk_level": risk_level,
                "narrative": narrative,
                "modalities": {
                    "audio": audio_results,
                    "video": video_results,
                    "transcript": transcript_results
                },
                "metadata_snapshot": metadata
            }

            results_path = event_dir / "slow_brain_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

            if self.config.demo_mode:
                print("\n" + "=" * 60)
                print(f"🧠 SLOW BRAIN ANALYSIS COMPLETE: {event_id}")
                print(f"   Risk Level:  {risk_level} ({final_risk_score:.2f})")
                print(f"   Narrative:   {narrative}")
                print("=" * 60 + "\n")
            else:
                logger.info(f"Finished {event_id} -> {risk_level} ({final_risk_score:.2f})")

            # 5. Database Integration (pgvector)
            if self.config.enable_pgvector and self.embedding_model:
                self._save_to_db(event_id, metadata, results, transcript_text, narrative)

        except Exception as e:
            logger.exception(f"Failed to process event {event_id}")
            # Mark as failed, so we don't infinitely retry
            (event_dir / "slow_brain_failed.marker").touch()

    def _save_to_db(self, event_id: str, metadata: dict, results: dict, transcript: str, narrative: str):
        try:
            # Generate embedding from narrative
            embedding = self.embedding_model.encode(narrative).tolist()

            from datetime import datetime
            start_ts = metadata.get("trigger_ts", 0)
            end_ts = metadata.get("ended_ts", start_ts)

            start_time = datetime.fromtimestamp(start_ts) if start_ts else None
            end_time = datetime.fromtimestamp(end_ts) if end_ts else None

            with self.conn.cursor() as cur:
                cur.execute("""
                            INSERT INTO incidents
                            (id, camera_id, start_time, end_time, trigger_reason, final_risk_score, risk_level,
                             transcript, narrative, embedding)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO
                            UPDATE SET
                                final_risk_score = EXCLUDED.final_risk_score,
                                risk_level = EXCLUDED.risk_level,
                                narrative = EXCLUDED.narrative,
                                embedding = EXCLUDED.embedding;
                            """, (
                                event_id,
                                metadata.get("camera_id", "unknown"),
                                start_time,
                                end_time,
                                metadata.get("trigger_reason", ""),
                                results["final_risk_score"],
                                results["risk_level"],
                                transcript,
                                narrative,
                                embedding
                            ))
            self.conn.commit()
            logger.info(f"Saved {event_id} to database.")
        except Exception as e:
            logger.error(f"Database insert failed for {event_id}: {e}")
            self.conn.rollback()

    def run_forever(self):
        logger.info(
            f"Slow Brain Worker started. Polling '{self.events_dir}' every {self.config.poll_interval_seconds}s.")

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            while True:
                try:
                    pending_events = self.get_pending_events()

                    for event_dir in pending_events:
                        # Simple lock file to prevent multiple workers picking up the same event if scaled horizontally
                        lock_file = event_dir / "slow_brain.lock"
                        if lock_file.exists():
                            continue

                        lock_file.touch()
                        executor.submit(self._process_and_unlock, event_dir, lock_file)

                except Exception as e:
                    logger.error(f"Error in poll loop: {e}")

                time.sleep(self.config.poll_interval_seconds)

    def _process_and_unlock(self, event_dir: Path, lock_file: Path):
        try:
            self.process_event(event_dir)
        finally:
            if lock_file.exists():
                lock_file.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slow Brain Worker")
    parser.add_argument("--demo-mode", action="store_true", help="Enable rich console output")
    parser.add_argument("--events-dir", type=str, default="events", help="Directory containing finalized events")
    parser.add_argument("--disable-pgvector", action="store_true", help="Disable database integration")

    args = parser.parse_args()

    config = SlowBrainConfig(
        demo_mode=args.demo_mode,
        events_dir=args.events_dir,
        enable_pgvector=not args.disable_pgvector
    )

    # Environment variable overrides
    if os.environ.get("DEMO_MODE", "").lower() in ("1", "true", "yes"):
        config.demo_mode = True

    worker = SlowBrainWorker(config)

    try:
        worker.run_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down Slow Brain Worker.")