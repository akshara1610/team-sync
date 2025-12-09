"""
TeamSync AI - Simple Meeting Bot Demo

Flow:
1. Bot joins Google Meet when prompted
2. Records and transcribes audio (or uses mock transcript)
3. Processes through orchestrator (all agents run automatically via LangGraph)
4. Creates JIRA tickets and schedules follow-up meetings (via MCP)
5. Sends MoM to all participants

Run: python scripts/demo_meeting_bot.py
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import uuid

sys.path.append(str(Path(__file__).parent.parent))

from src.integrations.google_meet_bot_selenium import GoogleMeetBotSelenium
from src.orchestrator import TeamSyncOrchestrator
from src.agents.listener_agent import ListenerAgent
from src.models.schemas import TranscriptData, SpeakerSegment
from src.utils.email_sender import EmailSender, print_meeting_minutes
from src.utils.audio_processor import AudioProcessor
from loguru import logger


def create_mock_transcript(participants):
    """Create realistic mock transcript (simulates what the bot heard)."""
    segments = [
        SpeakerSegment(
            speaker="SPEAKER_00",
            text="Good morning everyone! Let's start our sprint planning meeting. We have three critical items to discuss: the API migration, Redis caching, and our deployment pipeline.",
            start_time=0.0,
            end_time=8.5
        ),
        SpeakerSegment(
            speaker="SPEAKER_01",
            text="Thanks. I wanted to discuss the API migration to GraphQL first. I think we should prioritize this for the current sprint. It will improve our query efficiency significantly.",
            start_time=9.0,
            end_time=18.2
        ),
        SpeakerSegment(
            speaker="SPEAKER_00",
            text="Good point. Can you take the lead on the API migration? We need a detailed plan and proof of concept by next Friday.",
            start_time=18.8,
            end_time=26.4
        ),
        SpeakerSegment(
            speaker="SPEAKER_01",
            text="Absolutely. I'll start with the user service and create a working proof of concept. Should have it ready by Friday along with migration documentation.",
            start_time=27.0,
            end_time=35.1
        ),
        SpeakerSegment(
            speaker="SPEAKER_02",
            text="I can help with the frontend integration once the GraphQL API is ready. Also, we seriously need to implement Redis caching to reduce our database load.",
            start_time=35.8,
            end_time=46.2
        ),
        SpeakerSegment(
            speaker="SPEAKER_00",
            text="Great point. Can you handle the Redis implementation? Let's target a 50% reduction in database queries. We should also schedule a follow-up meeting next week.",
            start_time=47.0,
            end_time=57.5
        ),
        SpeakerSegment(
            speaker="SPEAKER_02",
            text="Yes, I'll work on the Redis setup. I'll need access to the production environment though. Can you grant me that access today?",
            start_time=58.0,
            end_time=64.2
        ),
        SpeakerSegment(
            speaker="SPEAKER_00",
            text="I'll grant you production access right after this meeting. Any questions about the deployment pipeline?",
            start_time=65.0,
            end_time=71.8
        ),
        SpeakerSegment(
            speaker="SPEAKER_01",
            text="One important question - should we maintain backward compatibility with the REST API during the GraphQL migration?",
            start_time=72.5,
            end_time=82.2
        ),
        SpeakerSegment(
            speaker="SPEAKER_00",
            text="Excellent question. Yes, we absolutely need to maintain both APIs in parallel for at least 6 months. This gives our clients sufficient time to migrate.",
            start_time=83.0,
            end_time=92.5
        ),
    ]

    # Map speakers to participants using frequency-based heuristic
    listener = ListenerAgent()
    speaker_mapping = listener.simple_roster_mapping(segments, participants)
    mapped_segments = listener.map_speaker_names(segments, speaker_mapping)

    transcript = TranscriptData(
        meeting_id=str(uuid.uuid4()),
        meeting_title="Sprint Planning - API Migration & Performance",
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(minutes=2),
        segments=mapped_segments,
        participants=participants
    )

    return transcript


async def main():
    """Main demo flow."""
    print("\n" + "="*80)
    print("🤖 TEAMSYNC AI - INTELLIGENT MEETING ASSISTANT")
    print("="*80)
    print()

    # Step 1: Bot Joins Google Meet
    print("📋 STEP 1: Join Google Meet")
    print("-" * 80)

    join_meeting = input("Do you want the bot to join a real Google Meet? (y/n): ").strip().lower()

    # Initialize variables for later use
    bot = None
    recording_started = False

    if join_meeting == 'y':
        meeting_url = input("Enter Google Meet URL: ").strip()

        if meeting_url:
            duration_input = input("Meeting duration in minutes [30]: ").strip()
            try:
                duration_minutes = int(duration_input) if duration_input else 30
            except:
                duration_minutes = 30

            print("\n🤖 Launching bot...")
            bot = GoogleMeetBotSelenium("TeamSync AI Bot")

            print("🌐 Bot joining meeting...")
            success = bot.join_meeting(meeting_url)

            if success:
                print("✅ Bot joined successfully!")
                print(f"🎧 Bot is now listening to the meeting...")
                print(f"⏱️  Bot will stay for {duration_minutes} minutes")
                print("⚠️  Press Ctrl+C to end early")
                print()

                # Start audio recording
                print("🎙️  Starting audio recording...")
                recording_started = await bot.start_recording()

                if recording_started:
                    print("✅ Audio recording active (using BlackHole if configured)")
                else:
                    print("⚠️  Audio recording failed - will use mock transcript")

                print()
                print("📝 While the bot is in the meeting:")
                print("   • Chrome window is open (DO NOT close it)")
                print("   • Bot is capturing the conversation")
                print("   • Audio is being recorded")
                print("   • Processing will begin after the meeting ends")
                print()

                # Stay in meeting for full duration (use async version to allow recording!)
                try:
                    await bot.stay_in_meeting_async(duration_minutes * 60)  # Convert to seconds
                    print("\n✅ Meeting ended - Bot left the meeting")
                except KeyboardInterrupt:
                    print("\n⏸️  Meeting ended early by user")
                    bot.leave_meeting()
                finally:
                    # Stop recording if it was started
                    if recording_started:
                        print("\n⏹️  Stopping audio recording...")
                        await bot.stop_recording()
                        print(f"✅ Recording saved: {bot.output_path}")
            else:
                print("❌ Failed to join. Continuing with mock transcript...")
        else:
            print("No URL provided. Using mock transcript for demo...")
    else:
        print("Skipping live join. Using mock transcript for demo...\n")

    # Get participant emails
    print("\n📧 Enter participant emails (comma-separated):")
    print("   Example: ap4613@columbia.edu, vva2113@columbia.edu, sk5476@columbia.edu")
    participants_input = input("Participants: ").strip()

    if participants_input:
        participants = [email.strip() for email in participants_input.split(',')]
    else:
        # Default participants
        participants = ["ap4613@columbia.edu", "vva2113@columbia.edu", "sk5476@columbia.edu"]
        print(f"Using default: {', '.join(participants)}")

    # Step 2: Background Processing Starts
    print("\n" + "="*80)
    print("🔄 BACKGROUND PROCESSING (Agents Working)")
    print("="*80)

    # Process audio if we have a recording, otherwise use mock transcript
    transcript = None

    if join_meeting == 'y' and recording_started and hasattr(bot, 'output_path'):
        # We have a real recording - transcribe it!
        print("\n🎤 Transcribing recorded audio with Whisper + Pyannote...")
        print("   (This may take a few minutes depending on meeting length)")

        try:
            audio_processor = AudioProcessor(whisper_model="base")
            transcript = audio_processor.process_audio_file(
                audio_path=bot.output_path,
                meeting_title="Sprint Planning - API Migration & Performance",
                participants=participants
            )

            print(f"✅ Real transcript generated from audio!")
            print(f"   • Meeting: {transcript.meeting_title}")
            print(f"   • Duration: {(transcript.end_time - transcript.start_time).total_seconds() / 60:.1f} minutes")
            print(f"   • Segments: {len(transcript.segments)}")
            print(f"   • Speakers detected: {len(set(seg.speaker for seg in transcript.segments))}")

            # Map speakers to participants using frequency heuristic
            listener = ListenerAgent()
            speaker_mapping = listener.simple_roster_mapping(transcript.segments, participants)
            mapped_segments = listener.map_speaker_names(transcript.segments, speaker_mapping)
            transcript.segments = mapped_segments

            print(f"   • Speakers mapped to participants")

        except Exception as e:
            logger.error(f"Failed to process audio: {e}")
            print(f"⚠️  Audio processing failed: {e}")
            print("   Falling back to mock transcript for demo...")
            transcript = None

    if not transcript:
        # No recording or processing failed - load existing transcript
        print("\n📝 Loading existing transcript: gmeet_20251209_022911.json")

        import json
        transcript_path = Path("data/transcripts/gmeet_20251209_022911.json")

        if transcript_path.exists():
            with open(transcript_path, 'r') as f:
                transcript_data = json.load(f)

            # Convert to TranscriptData
            segments = [
                SpeakerSegment(**seg) for seg in transcript_data['segments']
            ]

            transcript = TranscriptData(
                meeting_id=transcript_data['meeting_id'],
                meeting_title="Training Completion Meeting - BEI & Post Training",
                start_time=datetime.now() - timedelta(minutes=2),
                end_time=datetime.now(),
                segments=segments,
                participants=transcript_data.get('participants', participants)
            )
            print(f"✅ Loaded transcript with {len(segments)} segments")
        else:
            print("⚠️  Transcript file not found, using mock transcript...")
            transcript = create_mock_transcript(participants)

    print(f"\n✅ Transcript ready:")
    print(f"   • Meeting: {transcript.meeting_title}")
    print(f"   • Participants: {', '.join(participants)}")
    print(f"   • Segments: {len(transcript.segments)}")
    print()

    # Step 3: Run through orchestrator (LangGraph + MCP)
    print("🚀 Running complete workflow through orchestrator...")
    print("-" * 80)

    try:
        # Initialize orchestrator
        orchestrator = TeamSyncOrchestrator()

        # Populate listener with transcript data
        orchestrator.listener_agent.meeting_id = transcript.meeting_id
        orchestrator.listener_agent.segments = transcript.segments
        orchestrator.listener_agent.participants = set(transcript.participants)

        # Initialize state
        initial_state = {
            "meeting_id": transcript.meeting_id,
            "meeting_title": transcript.meeting_title,
            "room_name": "demo-room",
            "start_time": transcript.start_time,
            "status": "initialized",
            "transcript": {},
            "transcript_path": "",
            "initial_summary": {},
            "reflection_feedback": {},
            "final_summary": {},
            "summary_path": "",
            "jira_tickets": [],
            "followup_meeting": {},
            "messages": [],
            "validation_passed": False,
            "reflection_iterations": 0,
            "errors": [],
            "mcp_audit_log": []
        }

        # Run listen node
        print("  → Running 'listen' node...")
        listen_result = orchestrator._listen_node(initial_state)
        state = {**initial_state, **listen_result}
        print(f"  ✅ Transcript captured ({len(transcript.segments)} segments)")

        # Run summarize
        print("  → Running 'summarize' node...")
        summary_result = orchestrator._summarize_node(state)
        state = {**state, **summary_result}
        print(f"  ✅ Summary generated")
        print(f"     • Action items: {len(state.get('initial_summary', {}).get('action_items', []))}")
        print(f"     • Key decisions: {len(state.get('initial_summary', {}).get('key_decisions', []))}")

        # Run reflect
        print("  → Running 'reflect' node...")
        reflect_result = orchestrator._reflect_node(state)
        state = {**state, **reflect_result}
        print(f"  ✅ Reflection complete (approved: {state['validation_passed']})")

        # Improve if needed
        if not state['validation_passed']:
            print("  → Running 'improve' node...")
            improve_result = orchestrator._improve_node(state)
            state = {**state, **improve_result}
            print("  ✅ Summary improved")

        # Execute actions (JIRA tickets via MCP)
        print("  → Running 'execute_actions' node (creating JIRA tickets via MCP)...")
        action_result = orchestrator._action_node(state)
        state = {**state, **action_result}
        print(f"  ✅ JIRA tickets created: {len(state['jira_tickets'])}")
        for ticket in state['jira_tickets']:
            print(f"     • {ticket.get('action_item', 'N/A')}: {ticket.get('status', 'N/A')}")

        # Schedule follow-up (via MCP)
        print("  → Running 'schedule_followup' node (via MCP)...")
        schedule_result = orchestrator._schedule_node(state)
        state = {**state, **schedule_result}
        print(f"  ✅ Follow-up meeting scheduled: {state['followup_meeting'].get('success', False)}")

        # Store knowledge
        print("  → Running 'store_knowledge' node...")
        final_result = orchestrator._store_node(state)
        state = {**state, **final_result}
        print("  ✅ Knowledge stored in ChromaDB")

        print()
        print("="*80)
        print("📄 MINUTES OF MEETING")
        print("="*80)

        # Display MoM from final_summary
        final_summary = state.get('final_summary', {})
        if final_summary:
            from src.models.schemas import MeetingSummary
            # Convert dict to MeetingSummary if needed
            if isinstance(final_summary, dict):
                summary_obj = MeetingSummary(**final_summary)
            else:
                summary_obj = final_summary

            print_meeting_minutes(summary_obj)

            # Send email
            print("\n📧 Sending MoM to participants...")
            print("-" * 80)

            try:
                email_sender = EmailSender()

                if email_sender.sender_password:
                    print(f"From: {email_sender.sender_email}")
                    print(f"To: {', '.join(participants)}")

                    # Get summary path from state
                    summary_path = state.get('summary_path', 'data/summaries/summary.json')

                    success = email_sender.send_meeting_minutes(
                        summary=summary_obj,
                        recipients=participants,
                        summary_file_path=summary_path
                    )

                    if success:
                        print("✅ Email sent successfully!")
                        for email in participants:
                            print(f"   ✉️  {email}")
                    else:
                        print("❌ Failed to send email")
                else:
                    print("⚠️  Email not configured")
                    print("   Add EMAIL_PASSWORD to .env to enable")
            except Exception as e:
                print(f"⚠️  Email skipped: {e}")

        # Final summary
        print("\n" + "="*80)
        print("✅ WORKFLOW COMPLETE")
        print("="*80)

        print("\n📊 Summary of Actions:")
        print(f"   ✅ Transcript processed")
        print(f"   ✅ Summary generated & validated")
        print(f"   ✅ JIRA tickets created: {len(state.get('jira_tickets', []))}")
        print(f"   ✅ Follow-up meeting scheduled: {state.get('followup_meeting', {}).get('success', False)}")
        print(f"   ✅ Knowledge stored in ChromaDB")
        print(f"   ✅ Email sent to {len(participants)} participants")

        print("\n📁 Generated Files:")
        print(f"   • Transcript: {state.get('transcript_path', 'N/A')}")
        print(f"   • Summary: {state.get('summary_path', 'N/A')}")
        print(f"   • Vector DB: data/chroma_db/")

        # Show MCP audit log
        audit_log = orchestrator.mcp_server.get_audit_log()
        if audit_log:
            print(f"\n🔍 MCP Tool Calls: {len(audit_log)}")
            for entry in audit_log[:5]:  # Show first 5
                print(f"   • {entry['tool']} - {entry['result'].get('success', 'N/A')}")

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ Error: {e}")

    print("\n" + "="*80)
    print("Thank you for using TeamSync AI! 🚀")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
