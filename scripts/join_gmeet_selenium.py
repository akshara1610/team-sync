"""
Join a Google Meet using Selenium (more reliable for automation detection bypass)

This script uses undetected-chromedriver which bypasses Google's automation detection.
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.integrations.google_meet_bot_selenium import GoogleMeetBotSelenium
from src.agents.listener_agent import ListenerAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.models.schemas import TranscriptData
from src.utils.email_sender import print_meeting_minutes
from loguru import logger


def quick_join_test():
    """Quick test - just join and stay for 30 seconds."""

    print("\n🧪 QUICK JOIN TEST (30 seconds)")
    print("="*80)

    meeting_url = input("Meeting URL: ").strip()

    if not meeting_url:
        print("❌ No URL provided")
        return

    bot = GoogleMeetBotSelenium("Test Bot")

    print("\n🚪 Joining meeting...")
    success = bot.join_meeting(meeting_url)

    if success:
        print("✅ Joined! Staying for 30 seconds...")
        print("⚠️  A Chrome window is open - DO NOT close it!")
        bot.stay_in_meeting(30)
        print("✅ Test complete!")
    else:
        print("❌ Failed to join")
        if bot.driver:
            bot.driver.quit()


def join_and_record_meeting():
    """
    Complete flow:
    1. Join Google Meet
    2. Record audio
    3. Transcribe
    4. Generate MoM
    """

    print("\n" + "="*80)
    print("🎥 JOINING REAL GOOGLE MEET (Selenium)")
    print("="*80)

    # Step 1: Get meeting URL from user
    print("\n📝 Enter your Google Meet details:")
    meeting_url = input("Meeting URL (e.g., https://meet.google.com/abc-defg-hij): ").strip()

    if not meeting_url:
        print("❌ No URL provided. Exiting...")
        return

    bot_name = input("Bot display name [TeamSync AI Bot]: ").strip() or "TeamSync AI Bot"
    duration = input("Meeting duration in seconds [300 = 5 min]: ").strip()

    try:
        duration_seconds = int(duration) if duration else 300
    except:
        duration_seconds = 300

    # Step 2: Initialize the bot
    print(f"\n🤖 Initializing bot: {bot_name}")
    bot = GoogleMeetBotSelenium(bot_name=bot_name)

    # Step 3: Join the meeting
    print(f"\n🚪 Joining meeting: {meeting_url}")
    success = bot.join_meeting(meeting_url)

    if not success:
        print("❌ Failed to join meeting")
        if bot.driver:
            bot.driver.quit()
        return

    print("\n✅ Bot successfully joined the meeting!")
    print(f"🎙️  Bot will stay in meeting for {duration_seconds} seconds...")
    print("⚠️  A Chrome window is open - DO NOT close it!")

    # Note: Audio recording from system audio is complex and requires special setup
    # For now, we'll demonstrate the bot joining and staying in the meeting
    # In production, you'd use Google Meet's API or capture system audio with additional tools

    print("\n⚠️  Note: Audio recording from Google Meet requires additional setup:")
    print("   - macOS: BlackHole virtual audio device")
    print("   - Windows: Virtual Audio Cable")
    print("   - Linux: PulseAudio loopback")
    print("\n   For now, the bot will join and stay in the meeting.")
    print("   You can manually record the meeting using Google Meet's recording feature.")

    # Step 5: Stay in meeting for specified duration
    try:
        bot.stay_in_meeting(duration_seconds)
    except KeyboardInterrupt:
        print("\n⏸️  Stopping early...")

    # Step 6: Leave meeting
    bot.leave_meeting()

    print(f"\n✅ Meeting ended.")
    print("\n💡 To test transcription with a sample audio file, use option 3 in the menu")


async def transcribe_and_summarize(audio_path: str):
    """
    Transcribe audio and generate MoM.

    Args:
        audio_path: Path to recorded audio file
    """
    print("\n🎤 Transcribing audio (this may take a few minutes)...")

    # Initialize listener agent
    listener = ListenerAgent()

    # Transcribe and diarize
    segments = await listener.transcribe_and_diarize(audio_path)

    if not segments:
        print("❌ No speech detected in recording")
        return

    print(f"✅ Transcribed {len(segments)} segments")
    print(f"   Detected {len(set(seg.speaker for seg in segments))} speakers")

    # Get participant emails
    print("\n📧 Enter participant emails (comma-separated):")
    participants_input = input("Emails: ").strip()
    participants = [email.strip() for email in participants_input.split(',')] if participants_input else []

    # Map speakers if we have participants
    if participants:
        print("\n🗣️  Mapping speakers to participants...")
        speaker_mapping = listener.simple_roster_mapping(segments, participants)

        print("   Speaker mapping:")
        for speaker, email in speaker_mapping.items():
            count = sum(1 for seg in segments if seg.speaker == speaker)
            print(f"   {speaker} → {email} ({count} segments)")

        segments = listener.map_speaker_names(segments, speaker_mapping)

    # Create transcript
    meeting_title = input("\nMeeting title: ").strip() or "Google Meet Recording"

    transcript = TranscriptData(
        meeting_id=f"gmeet_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        meeting_title=meeting_title,
        start_time=datetime.now(),
        segments=segments,
        participants=participants or ["Unknown"]
    )

    # Generate summary
    print("\n📊 Generating Minutes of Meeting...")
    summarizer = SummarizerAgent()
    summary = summarizer.generate_summary(transcript)

    # Display MoM
    print("\n" + "="*80)
    print_meeting_minutes(summary)

    # Save
    output_path = f"data/summaries/{transcript.meeting_id}_summary.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summarizer.save_summary(summary, output_path)

    print(f"\n💾 Summary saved: {output_path}")


if __name__ == "__main__":
    print("\n=== TeamSync: Real Google Meet Integration (Selenium) ===\n")
    print("Choose mode:")
    print("1. Full flow (join → record → transcribe → MoM)")
    print("2. Quick test (join for 30 seconds)")
    print("3. Transcribe existing recording")
    print()

    choice = input("Choice (1/2/3): ").strip()

    if choice == "1":
        join_and_record_meeting()
    elif choice == "2":
        quick_join_test()
    elif choice == "3":
        audio_path = input("Path to audio file: ").strip()
        if audio_path:
            asyncio.run(transcribe_and_summarize(audio_path))
        else:
            print("❌ No path provided")
    else:
        print("Invalid choice")
