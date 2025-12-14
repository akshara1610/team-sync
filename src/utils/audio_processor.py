"""
Audio Processing Module

Handles:
1. Speech-to-text transcription with Whisper
2. Speaker diarization with Pyannote.audio
3. Combining transcription and diarization into structured segments
"""
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
from loguru import logger

try:
    import whisper
    import torch
    from pyannote.audio import Pipeline
    import wave  # Built-in module for reading WAV files
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Install with: pip install openai-whisper pyannote-audio")
    raise

from src.models.schemas import SpeakerSegment, TranscriptData
from src.config import settings


class AudioProcessor:
    """
    Processes audio files to generate transcripts with speaker diarization.
    """

    def __init__(
        self,
        whisper_model: str = "base",
        hf_token: Optional[str] = None
    ):
        """
        Initialize audio processor.

        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            hf_token: HuggingFace token for Pyannote (required for diarization)
        """
        self.whisper_model_name = whisper_model
        self.hf_token = hf_token or settings.HF_TOKEN

        logger.info(f"Loading Whisper model: {whisper_model}")
        self.whisper_model = whisper.load_model(whisper_model)

        logger.info("Initializing Pyannote pipeline for speaker diarization...")
        if not self.hf_token:
            logger.warning("HF_TOKEN not set in .env. Diarization may not work.")
            logger.warning("Get token from: https://huggingface.co/settings/tokens")
            self.diarization_pipeline = None
        else:
            try:
                # Fix PyTorch 2.6+ weights_only issue - add all required safe globals
                import torch.serialization
                from pyannote.audio.core.task import Specifications

                torch.serialization.add_safe_globals([
                    torch.torch_version.TorchVersion,
                    Specifications
                ])

                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=self.hf_token
                )
                # Use GPU if available
                if torch.cuda.is_available():
                    self.diarization_pipeline.to(torch.device("cuda"))
                logger.info("✅ Pyannote pipeline loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Pyannote pipeline: {e}")
                logger.warning("Continuing without speaker diarization - all speakers will be labeled as SPEAKER_00")
                self.diarization_pipeline = None

    def transcribe_audio(self, audio_path: str) -> dict:
        """
        Transcribe audio file using Whisper.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)

        Returns:
            Whisper transcription result with segments
        """
        logger.info(f"🎤 Transcribing audio: {audio_path}")

        result = self.whisper_model.transcribe(
            audio_path,
            language="en",  # Can be auto-detected with language=None
            verbose=False
        )

        num_segments = len(result.get("segments", []))
        logger.info(f"✅ Transcription complete: {num_segments} segments")

        return result

    def diarize_audio(self, audio_path: str) -> Optional[dict]:
        """
        Perform speaker diarization using Pyannote.

        Args:
            audio_path: Path to audio file

        Returns:
            Diarization result with speaker segments, or None if diarization unavailable
        """
        if not self.diarization_pipeline:
            logger.warning("Diarization pipeline not available")
            return None

        logger.info(f"👥 Performing speaker diarization: {audio_path}")

        diarization = self.diarization_pipeline(audio_path)

        # Convert to serializable format
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })

        logger.info(f"✅ Diarization complete: {len(segments)} speaker segments")
        logger.info(f"   Detected {len(set(s['speaker'] for s in segments))} unique speakers")

        return {"segments": segments}

    def align_transcription_and_diarization(
        self,
        transcription: dict,
        diarization: Optional[dict]
    ) -> List[SpeakerSegment]:
        """
        Align Whisper transcription with Pyannote diarization.

        Args:
            transcription: Whisper result with text segments
            diarization: Pyannote result with speaker segments

        Returns:
            List of SpeakerSegment objects with text and speaker labels
        """
        segments = []

        if not diarization:
            # No diarization - use Whisper segments without speaker labels
            logger.info("No diarization available - using single speaker")
            for seg in transcription.get("segments", []):
                segments.append(SpeakerSegment(
                    speaker="SPEAKER_00",
                    text=seg["text"].strip(),
                    start_time=seg["start"],
                    end_time=seg["end"]
                ))
            return segments

        # Align by finding overlapping time ranges
        diar_segments = diarization["segments"]

        for transcript_seg in transcription.get("segments", []):
            t_start = transcript_seg["start"]
            t_end = transcript_seg["end"]
            t_mid = (t_start + t_end) / 2  # Midpoint of transcript segment

            # Find which speaker was talking at the midpoint
            speaker = "SPEAKER_00"  # Default
            for diar_seg in diar_segments:
                if diar_seg["start"] <= t_mid <= diar_seg["end"]:
                    speaker = diar_seg["speaker"]
                    break

            segments.append(SpeakerSegment(
                speaker=speaker,
                text=transcript_seg["text"].strip(),
                start_time=t_start,
                end_time=t_end
            ))

        logger.info(f"✅ Aligned {len(segments)} segments")
        return segments

    def process_audio_file(
        self,
        audio_path: str,
        meeting_title: str = "Meeting",
        participants: Optional[List[str]] = None
    ) -> TranscriptData:
        """
        Complete audio processing pipeline: transcribe + diarize + align.

        Args:
            audio_path: Path to audio file
            meeting_title: Title for the meeting
            participants: List of participant emails

        Returns:
            TranscriptData with complete transcript and speaker information
        """
        logger.info("="*80)
        logger.info(f"🎬 Starting audio processing: {audio_path}")
        logger.info("="*80)

        # Step 1: Transcribe
        transcription = self.transcribe_audio(audio_path)

        # Step 2: Diarize
        diarization = self.diarize_audio(audio_path)

        # Step 3: Align
        segments = self.align_transcription_and_diarization(transcription, diarization)

        # Get audio duration using wave module
        with wave.open(audio_path, 'rb') as audio_file:
            frames = audio_file.getnframes()
            rate = audio_file.getframerate()
            duration_seconds = frames / float(rate)

        # Create TranscriptData
        meeting_id = Path(audio_path).stem  # Use filename as meeting ID
        start_time = datetime.now() - timedelta(seconds=duration_seconds)

        transcript = TranscriptData(
            meeting_id=meeting_id,
            meeting_title=meeting_title,
            start_time=start_time,
            end_time=datetime.now(),
            segments=segments,
            participants=participants or []
        )

        logger.info("="*80)
        logger.info("✅ Audio processing complete!")
        logger.info(f"   Duration: {duration_seconds:.1f} seconds")
        logger.info(f"   Segments: {len(segments)}")
        logger.info(f"   Speakers: {len(set(s.speaker for s in segments))}")
        logger.info("="*80)

        return transcript


def test_audio_processor():
    """Test the audio processor with a sample file."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.audio_processor <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    if not Path(audio_file).exists():
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)

    processor = AudioProcessor(whisper_model="base")
    transcript = processor.process_audio_file(
        audio_file,
        meeting_title="Test Meeting",
        participants=["user1@example.com", "user2@example.com"]
    )

    print("\n" + "="*80)
    print("TRANSCRIPT PREVIEW")
    print("="*80)

    for i, seg in enumerate(transcript.segments[:10], 1):
        print(f"\n{i}. {seg.speaker} ({seg.start_time:.1f}s - {seg.end_time:.1f}s)")
        print(f"   {seg.text}")

    if len(transcript.segments) > 10:
        print(f"\n... ({len(transcript.segments) - 10} more segments)")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_audio_processor()
