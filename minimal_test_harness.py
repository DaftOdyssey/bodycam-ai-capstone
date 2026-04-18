"""
Minimal ZMQ Media Pipeline Test Harness

This script acts as a dummy server for edge_pipeline.py. It listens on ZMQ ports 
for control and data messages, buffers media in memory for the active incident, 
and upon completion (or a quiet period timeout) writes the assembled media 
to disk and passes it to the slow_brain_worker for offline processing.
"""

import json
import logging
import signal
import sys
import select
import time
import wave
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import zmq

from slow_brain_worker import SlowBrainWorker, SlowBrainConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("minimal_test_harness")


class MinimalHarness:
    def __init__(self):
        self.context = zmq.Context()
        
        # 1. Control REP socket (handles starts, status, ends)
        self.control_socket = self.context.socket(zmq.REP)
        self.control_socket.bind("tcp://127.0.0.1:5555")
        
        # 2. Data PULL socket (receives audio/video/events)
        self.data_socket = self.context.socket(zmq.PULL)
        self.data_socket.bind("tcp://127.0.0.1:5556")
        
        # Poller to listen on both sockets non-blockingly
        self.poller = zmq.Poller()
        self.poller.register(self.control_socket, zmq.POLLIN)
        self.poller.register(self.data_socket, zmq.POLLIN)
        
        # Active incident state
        self.active_event_id: Optional[str] = None
        self.audio_chunks: List[tuple[bytes, str]] = []
        self.video_frames: List[bytes] = []
        self.audio_metadata = {}
        self.event_metadata = {}
        
        # Lifecycle / Timeout
        self.last_packet_ts = 0.0
        self.pending_finalization_ts: Optional[float] = None
        self.quiet_period_seconds = 5.0
        self.stop_event = False
        self.manual_clear_requested = False

        # Prepare slow brain worker (using defaults)
        self.slow_brain_config = SlowBrainConfig()
        self.slow_brain = SlowBrainWorker(self.slow_brain_config)

    def _finalize_event(self) -> None:
        """Assemble the in-memory media, write to disk, and trigger slow brain."""
        if not self.active_event_id:
            return
            
        event_id = self.active_event_id
        logger.info(f"Finalizing event: {event_id} | audio chunks: {len(self.audio_chunks)} | video frames: {len(self.video_frames)}")
        
        # Set up output directories
        event_dir = Path(f"./events/{event_id}")
        assembled_dir = event_dir / "assembled"
        assembled_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Write Metadata Files
        camera_id = self.event_metadata.get("camera_id", "demo_camera")
        store_id = self.event_metadata.get("store_id", "demo_store")
        trigger_reason = self.event_metadata.get("trigger_reason", "unknown")
        started_ts = self.event_metadata.get("started_ts") or self.event_metadata.get("trigger_ts") or time.time()
        ended_ts = self.event_metadata.get("ended_ts") or time.time()
        
        session_meta_path = event_dir / "session_metadata.json"
        with open(session_meta_path, "w") as f:
            json.dump({
                "event_id": event_id,
                "camera_id": camera_id,
                "store_id": store_id,
                "trigger_reason": trigger_reason,
                "started_ts": started_ts,
                "ended_ts": ended_ts,
                "trigger_ts": self.event_metadata.get("trigger_ts", started_ts)
            }, f, indent=2)
            
        manifest_path = event_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({
                "event_id": event_id,
                "finalized": True,
                "camera_id": camera_id,
                "store_id": store_id,
                "trigger_reason": trigger_reason,
                "started_ts": started_ts,
                "ended_ts": ended_ts
            }, f, indent=2)
        
        # 2. Assemble Audio -> WAV
        audio_path = assembled_dir / "incident_audio.wav"
        if self.audio_chunks:
            sample_rate = self.audio_metadata.get("sample_rate", 16000)
            channels = self.audio_metadata.get("channels", 1)
            
            with wave.open(str(audio_path), "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2) # Normalize all to 16-bit PCM
                wf.setframerate(sample_rate)
                
                for payload, dtype in self.audio_chunks:
                    if dtype == "float32":
                        float_arr = np.frombuffer(payload, dtype=np.float32)
                        int16_arr = np.clip(float_arr, -1.0, 1.0)
                        int16_arr = (int16_arr * 32767.0).astype(np.int16)
                        wf.writeframes(int16_arr.tobytes())
                    else:
                        wf.writeframes(payload)
                
                audio_frames = wf.getnframes()
                
            audio_duration = audio_frames / sample_rate
            logger.info(f"Wrote assembled audio -> {audio_path} (duration: {audio_duration:.2f}s)")
            
        # 3. Assemble Video -> MP4
        video_path = assembled_dir / "incident_video.mp4"
        if self.video_frames:
            h, w = None, None
            fps = float(self.event_metadata.get("capture_fps", 30.0))
            if self.audio_chunks and 'audio_duration' in locals() and audio_duration > 0:
                fps = len(self.video_frames) / audio_duration
                logger.info(f"Calculated effective video fps: {fps:.2f} to match audio duration.")
                
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = None
            
            for frame_bytes in self.video_frames:
                try:
                    np_arr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                        
                    if out is None:
                        h, w = frame.shape[:2]
                        out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                        
                    # Enforce consistent resolution to avoid video writer errors
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h))
                        
                    out.write(frame)
                except Exception as e:
                    logger.warning(f"Failed to decode/write video frame: {e}")
                    continue
                    
            if out is not None:
                out.release()
                logger.info(f"Wrote assembled video -> {video_path}")
                
        # 3. Call Slow Brain Worker
        logger.info("Triggering slow_brain_worker offline analysis...")
        try:
            self.slow_brain.process_event_dir(event_dir)
            logger.info(f"slow_brain_worker finished. Results at {event_dir}/slow_brain_output.json")
        except Exception as e:
            logger.error(f"slow_brain_worker failed: {e}")
            
        # 4. Reset internal state for the next incident
        self.active_event_id = None
        self.audio_chunks = []
        self.video_frames = []
        self.audio_metadata = {}
        self.event_metadata = {}
        self.pending_finalization_ts = None
        self.manual_clear_requested = False

    def handle_control(self, payload: dict) -> dict:
        """Process requests on the control port (REQ/REP)."""
        msg_type = payload.get("type")
        event_id = payload.get("event_id")
        logger.info(f"Control message: {msg_type} (event_id: {event_id})")
        
        if msg_type == "incident_start":
            if self.active_event_id and self.active_event_id != event_id:
                self._finalize_event()
            self.active_event_id = event_id
            self.event_metadata.update(payload)
            self.last_packet_ts = time.time()
            return {"ack": True, "clear": False}
            
        elif msg_type == "incident_status":
            self.last_packet_ts = time.time()
            do_clear = self.manual_clear_requested
            if do_clear:
                logger.info("Replying to incident_status with clear=True to signal edge_pipeline.")
            return {"ack": True, "clear": do_clear}
            
        elif msg_type == "incident_end":
            self.event_metadata.update(payload)
            if not self.pending_finalization_ts:
                logger.info(f"Received control incident_end. Starting trailing {self.quiet_period_seconds}s quiet period.")
                self.pending_finalization_ts = time.time()
            return {"ack": True, "clear": True}
            
        return {"ack": True}

    def run(self):
        """Main loop: Poll both ZMQ sockets and enforce timeouts."""
        logger.info("Minimal harness started.")
        logger.info("Control REP listening on tcp://127.0.0.1:5555")
        logger.info("Data PULL listening on tcp://127.0.0.1:5556")
        
        while not self.stop_event:
            try:
                events = dict(self.poller.poll(1000))
            except zmq.ZMQError:
                break
            except Exception as e:
                logger.error(f"Poll error: {e}")
                break
                
            now = time.time()
            
            # 1. Handle Control Messages (Priority)
            if self.control_socket in events:
                try:
                    msg_bytes = self.control_socket.recv(flags=zmq.NOBLOCK)
                    payload = json.loads(msg_bytes.decode("utf-8"))
                    response = self.handle_control(payload)
                    self.control_socket.send(json.dumps(response).encode("utf-8"))
                except zmq.Again:
                    pass
                    
            # 2. Handle Data Packets
            if self.data_socket in events:
                try:
                    parts = self.data_socket.recv_multipart(flags=zmq.NOBLOCK)
                    if len(parts) == 3:
                        kind = parts[0].decode("utf-8")
                        metadata = json.loads(parts[1].decode("utf-8"))
                        payload = parts[2]
                        
                        event_id = metadata.get("event_id")
                        if event_id:
                            self.active_event_id = event_id
                            self.last_packet_ts = now
                            
                            if kind == "audio":
                                if not self.audio_metadata:
                                    self.audio_metadata = metadata
                                self.audio_chunks.append((payload, metadata.get("dtype", "int16")))
                            elif kind == "video":
                                self.video_frames.append(payload)
                            elif kind == "event":
                                self.event_metadata.update(metadata)
                                if metadata.get("packet_type") == "incident_end":
                                    if not self.pending_finalization_ts:
                                        logger.info(f"Received data-plane incident_end marker. Starting trailing {self.quiet_period_seconds}s quiet period.")
                                        self.pending_finalization_ts = now
                except zmq.Again:
                    pass

            # 3. Enforce Quiet Period Timeout / Finalization
            if self.active_event_id:
                # If we received an explicit end marker, wait for the short trailing period
                if self.pending_finalization_ts and (now - self.pending_finalization_ts > self.quiet_period_seconds):
                    logger.info("Trailing quiet period elapsed. Finalizing event.")
                    self._finalize_event()
                # Or if the connection completely dropped without an end marker (safety net)
                elif not self.pending_finalization_ts and (now - self.last_packet_ts > 30.0):
                    logger.info("Hard timeout reached (30s without packets). Finalizing event automatically.")
                    self._finalize_event()

            # 4. Check for manual "clear" command from keyboard (non-blocking)
            rlist, _, _ = select.select([sys.stdin], [], [], 0)
            if rlist:
                line = sys.stdin.readline().strip().lower()
                if line in ("clear", "c"):
                    if self.active_event_id:
                        logger.info("Manual CLEAR command received. Will signal edge pipeline on next status poll.")
                        self.manual_clear_requested = True
                    else:
                        logger.info("Manual CLEAR ignored: no active incident.")

    def shutdown(self, *args):
        """Cleanup sockets safely on Ctrl+C."""
        logger.info("Shutting down harness...")
        self.stop_event = True
        self.data_socket.close(linger=0)
        self.control_socket.close(linger=0)
        self.context.term()


if __name__ == "__main__":
    harness = MinimalHarness()
    signal.signal(signal.SIGINT, harness.shutdown)
    signal.signal(signal.SIGTERM, harness.shutdown)
    harness.run()
