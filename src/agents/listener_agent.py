import asyncio
from typing import List, Optional, Dict
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
            self._ensure_models_loaded()

            if self.diarization_pipeline is None:
                logger.warning("Diarization pipeline not available, falling back to transcription only")
                return await self._transcribe_without_diarization(audio_path)

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

    async def _transcribe_without_diarization(self, audio_path: str) -> List[SpeakerSegment]:
        """
        Transcribe audio without diarization (fallback mode).

        Args:
            audio_path: Path to audio file

        Returns:
            List of segments with generic speaker labels
        """
        try:
            logger.info("Transcribing audio without speaker diarization...")
            self._ensure_models_loaded()

            segments, info = self.whisper_model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                vad_filter=True
            )

            result = []
            for segment in segments:
                speaker_segment = SpeakerSegment(
                    speaker="SPEAKER_00",  # Single speaker fallback
                    start_time=segment.start,
                    end_time=segment.end,
                    text=segment.text.strip()
                )
                result.append(speaker_segment)

            logger.info(f"Transcribed {len(result)} segments without diarization")
            return result

        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return []

    async def transcribe_and_diarize(self, audio_path: str) -> List[SpeakerSegment]:
        """
        Complete pipeline: Transcribe audio with Whisper and perform speaker diarization,
        then merge the results to get speaker-labeled transcripts.

        Args:
            audio_path: Path to audio file

        Returns:
            List of speaker segments with both text and speaker labels
        """
        try:
            logger.info("Starting complete transcription + diarization pipeline...")
            self._ensure_models_loaded()

            # Step 1: Whisper transcription (gets text + timestamps)
            logger.info("Step 1: Transcribing with Whisper...")
            whisper_segments, info = self.whisper_model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                vad_filter=True
            )

            whisper_segments = list(whisper_segments)
            logger.info(f"Whisper transcribed {len(whisper_segments)} segments")

            # Step 2: Speaker diarization (gets speaker labels + timestamps)
            if self.diarization_pipeline is None:
                logger.warning("No diarization pipeline available, using single speaker")
                # Fallback: all segments assigned to SPEAKER_00
                result = []
                for segment in whisper_segments:
                    result.append(SpeakerSegment(
                        speaker="SPEAKER_00",
                        start_time=segment.start,
                        end_time=segment.end,
                        text=segment.text.strip()
                    ))
                return result

            logger.info("Step 2: Performing speaker diarization...")
            diarization = self.diarization_pipeline(audio_path)

            # Convert diarization to list of (start, end, speaker) tuples
            diar_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                diar_segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })

            logger.info(f"Diarization identified {len(set(d['speaker'] for d in diar_segments))} unique speakers")

            # Step 3: Merge Whisper text with diarization speakers
            logger.info("Step 3: Merging transcription with speaker labels...")
            merged_segments = self._merge_transcription_and_diarization(
                whisper_segments,
                diar_segments
            )

            logger.info(f"Created {len(merged_segments)} merged segments")
            return merged_segments

        except Exception as e:
            logger.error(f"Error in transcribe_and_diarize: {e}")
            return []

    def _merge_transcription_and_diarization(
        self,
        whisper_segments: list,
        diar_segments: list
    ) -> List[SpeakerSegment]:
        """
        Merge Whisper transcription segments with diarization speaker labels.

        For each Whisper segment, find the overlapping diarization segment
        and assign the speaker label.

        Args:
            whisper_segments: List of Whisper segment objects with .start, .end, .text
            diar_segments: List of dicts with 'start', 'end', 'speaker'

        Returns:
            List of SpeakerSegment objects with both text and speaker labels
        """
        merged = []

        for w_seg in whisper_segments:
            # Find best matching diarization segment (maximum overlap)
            best_speaker = "SPEAKER_UNKNOWN"
            max_overlap = 0.0

            w_start = w_seg.start
            w_end = w_seg.end
            w_duration = w_end - w_start

            for d_seg in diar_segments:
                # Calculate overlap between whisper segment and diarization segment
                overlap_start = max(w_start, d_seg['start'])
                overlap_end = min(w_end, d_seg['end'])
                overlap = max(0, overlap_end - overlap_start)

                # Use segment with maximum overlap
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = d_seg['speaker']

            # Only assign speaker if overlap is significant (>50% of whisper segment)
            if max_overlap / w_duration < 0.5:
                logger.warning(f"Low overlap for segment at {w_start:.2f}s, using best match anyway")

            merged_segment = SpeakerSegment(
                speaker=best_speaker,
                start_time=w_start,
                end_time=w_end,
                text=w_seg.text.strip()
            )
            merged.append(merged_segment)

        return merged

    def simple_roster_mapping(
        self,
        segments: List[SpeakerSegment],
        participant_emails: List[str]
    ) -> Dict[str, str]:
        """
        Simple speaker mapping based on frequency (most talkative = first participant).

        Strategy:
        - Count segments per speaker
        - Most talkative speaker → first participant
        - Second most talkative → second participant
        - etc.

        Args:
            segments: List of segments with SPEAKER_XX labels
            participant_emails: List of participant emails from calendar/meeting

        Returns:
            Mapping dict: {"SPEAKER_00": "email@domain.com", ...}
        """
        # Count segments per speaker
        speaker_counts = {}
        for seg in segments:
            speaker_counts[seg.speaker] = speaker_counts.get(seg.speaker, 0) + 1

        # Sort by frequency (most talkative first)
        sorted_speakers = sorted(
            speaker_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Map to participants
        mapping = {}
        for i, (speaker, count) in enumerate(sorted_speakers):
            if i < len(participant_emails):
                mapping[speaker] = participant_emails[i]
                logger.info(f"Mapped {speaker} → {participant_emails[i]} ({count} segments)")
            else:
                mapping[speaker] = f"{speaker}"  # Keep original if not enough participants
                logger.warning(f"No participant for {speaker}, keeping original label")

        return mapping

    def map_speaker_names(
        self,
        segments: List[SpeakerSegment],
        speaker_mapping: dict
    ) -> List[SpeakerSegment]:
        """
        Map generic speaker labels (SPEAKER_00, SPEAKER_01) to actual names.

        Args:
            segments: List of segments with generic speaker labels
            speaker_mapping: Dict mapping labels to names, e.g.,
                            {"SPEAKER_00": "Akshara Pramod", "SPEAKER_01": "Vrinda Ahuja"}

        Returns:
            List of segments with mapped speaker names
        """
        mapped_segments = []

        for segment in segments:
            speaker = speaker_mapping.get(segment.speaker, segment.speaker)
            mapped_segment = SpeakerSegment(
                speaker=speaker,
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=segment.text
            )
            mapped_segments.append(mapped_segment)

        logger.info(f"Mapped {len(segments)} segments to named speakers")
        return mapped_segments

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
