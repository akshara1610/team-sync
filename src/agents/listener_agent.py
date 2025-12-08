import asyncio
from typing import List, Optional
import json
from datetime import datetime
from loguru import logger
from livekit import rtc, api
import numpy as np
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from src.config import settings
from src.models.schemas import TranscriptData, SpeakerSegment


class ListenerAgent:
    """
    Listener Agent: Captures real-time audio streams via LiveKit,
    performs transcription using WhisperX/Faster-Whisper,
    and conducts speaker diarization with pyannote.audio.
    """

    def __init__(self):
        self.room: Optional[rtc.Room] = None
        self.meeting_id: Optional[str] = None
        self.segments: List[SpeakerSegment] = []
        self.participants: set = set()

        # Lazy loading - only initialize when needed
        self.whisper_model = None
        self.diarization_pipeline = None

        # Audio buffer
        self.audio_buffer = []
        self.sample_rate = 16000

        logger.info("Listener Agent initialized (models will load on first use)")

    def _ensure_models_loaded(self):
        """Lazy load Whisper and diarization models only when needed."""
        if self.whisper_model is None:
            logger.info("Loading Whisper model...")
            self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

        if self.diarization_pipeline is None and settings.HF_TOKEN:
            logger.info("Loading speaker diarization model...")
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    token=settings.HF_TOKEN
                )
            except Exception as e:
                logger.warning(f"Could not load diarization model: {e}. Speaker diarization will be disabled.")
                self.diarization_pipeline = None

    async def join_meeting(self, room_name: str, meeting_id: str, token: str) -> bool:
        """
        Join a LiveKit meeting room as a virtual participant.

        Args:
            room_name: LiveKit room name
            meeting_id: Unique meeting identifier
            token: LiveKit access token

        Returns:
            bool: Success status
        """
        try:
            self.meeting_id = meeting_id
            self.room = rtc.Room()

            # Set up event handlers
            self.room.on("track_subscribed", self._on_track_subscribed)
            self.room.on("participant_connected", self._on_participant_connected)

            # Connect to room
            await self.room.connect(settings.LIVEKIT_URL, token)
            logger.info(f"Successfully joined meeting: {room_name}")

            return True
        except Exception as e:
            logger.error(f"Failed to join meeting: {e}")
            return False

    async def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        """Handle new participant joining."""
        self.participants.add(participant.identity)
        logger.info(f"Participant joined: {participant.identity}")

    async def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant
    ):
        """Handle audio track subscription."""
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Subscribed to audio track from: {participant.identity}")

            audio_stream = rtc.AudioStream(track)
            asyncio.create_task(self._process_audio_stream(audio_stream, participant.identity))

    async def _process_audio_stream(self, stream: rtc.AudioStream, speaker_id: str):
        """Process incoming audio stream in real-time."""
        try:
            async for frame in stream:
                # Convert audio frame to numpy array
                audio_data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
                self.audio_buffer.extend(audio_data)

                # Process in chunks (e.g., every 5 seconds)
                if len(self.audio_buffer) >= self.sample_rate * 5:
                    await self._transcribe_chunk(speaker_id)

        except Exception as e:
            logger.error(f"Error processing audio stream: {e}")

    async def _transcribe_chunk(self, speaker_id: str):
        """Transcribe audio chunk using Whisper."""
        try:
            audio_chunk = np.array(self.audio_buffer[:self.sample_rate * 5])
            self.audio_buffer = self.audio_buffer[self.sample_rate * 5:]

            # Save temporary audio file for processing
            temp_audio_path = f"/tmp/audio_chunk_{datetime.utcnow().timestamp()}.wav"

            # Transcribe with Whisper
            segments, info = self.whisper_model.transcribe(
                audio_chunk,
                language="en",
                task="transcribe",
                vad_filter=True
            )

            # Process segments
            for segment in segments:
                speaker_segment = SpeakerSegment(
                    speaker=speaker_id,
                    start_time=segment.start,
                    end_time=segment.end,
                    text=segment.text.strip()
                )
                self.segments.append(speaker_segment)
                logger.info(f"[{speaker_id}] {segment.text}")

        except Exception as e:
            logger.error(f"Error transcribing chunk: {e}")

    async def perform_diarization(self, audio_path: str) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on complete audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            List of speaker segments with corrected speaker labels
        """
        try:
            logger.info("Performing speaker diarization...")
            diarization = self.diarization_pipeline(audio_path)

            diarized_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    speaker=speaker,
                    start_time=turn.start,
                    end_time=turn.end,
                    text=""  # Will be filled by matching transcription
                )
                diarized_segments.append(segment)

            logger.info(f"Identified {len(set(s.speaker for s in diarized_segments))} unique speakers")
            return diarized_segments

        except Exception as e:
            logger.error(f"Error in diarization: {e}")
            return []

    def get_transcript(self, meeting_title: str, start_time: datetime) -> TranscriptData:
        """
        Get complete transcript with metadata.

        Args:
            meeting_title: Title of the meeting
            start_time: Meeting start time

        Returns:
            Complete transcript data
        """
        return TranscriptData(
            meeting_id=self.meeting_id,
            meeting_title=meeting_title,
            start_time=start_time,
            end_time=datetime.utcnow(),
            segments=self.segments,
            participants=list(self.participants)
        )

    async def leave_meeting(self):
        """Leave the meeting and cleanup."""
        if self.room:
            await self.room.disconnect()
            logger.info("Left meeting successfully")

    def save_transcript(self, output_path: str):
        """Save transcript to JSON file."""
        try:
            transcript_dict = {
                "meeting_id": self.meeting_id,
                "segments": [
                    {
                        "speaker": seg.speaker,
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "text": seg.text
                    }
                    for seg in self.segments
                ],
                "participants": list(self.participants)
            }

            with open(output_path, 'w') as f:
                json.dump(transcript_dict, f, indent=2)

            logger.info(f"Transcript saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")
