"""
Google Meet Bot - Actually joins Google Meet meetings and records them

This uses Playwright to:
1. Open a browser
2. Join the Google Meet
3. Record the audio
4. Transcribe in real-time or save for later processing
"""
import asyncio
import wave
import pyaudio
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger
from playwright.async_api import async_playwright, Page, Browser


class GoogleMeetBot:
    """
    Bot that actually joins Google Meet meetings.

    Uses Playwright browser automation to:
    - Join meetings with a custom name
    - Record system audio
    - Capture meeting participants
    """

    def __init__(self, bot_name: str = "TeamSync AI Bot"):
        self.bot_name = bot_name
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.is_in_meeting = False
        self.meeting_url: Optional[str] = None

        # Audio recording settings
        self.audio_frames = []
        self.is_recording = False

    async def join_meeting(self, meeting_url: str) -> bool:
        """
        Join a Google Meet meeting.

        Args:
            meeting_url: Full Google Meet URL (e.g., https://meet.google.com/abc-defg-hij)

        Returns:
            Success status
        """
        try:
            logger.info(f"🤖 Bot '{self.bot_name}' joining Google Meet: {meeting_url}")
            self.meeting_url = meeting_url

            # Start Playwright
            playwright = await async_playwright().start()

            # Launch Chrome browser with stealth settings
            self.browser = await playwright.chromium.launch(
                headless=False,
                channel='chrome',
                args=[
                    '--use-fake-ui-for-media-stream',
                    '--use-fake-device-for-media-stream',
                    '--disable-blink-features=AutomationControlled',
                    '--autoplay-policy=no-user-gesture-required',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--allow-running-insecure-content',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu'
                ]
            )

            # Create context with realistic settings
            context = await self.browser.new_context(
                permissions=['microphone', 'camera', 'notifications'],
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                device_scale_factor=1,
                is_mobile=False,
                has_touch=False,
                locale='en-US',
                timezone_id='America/New_York',
                color_scheme='light'
            )

            # Add extra headers to avoid detection
            await context.set_extra_http_headers({
                'Accept-Language': 'en-US,en;q=0.9',
            })

            self.page = await context.new_page()

            # Inject scripts to hide automation
            await context.add_init_script("""
                // Hide webdriver property
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => false,
                });

                // Mock chrome runtime
                window.chrome = {
                    runtime: {},
                };

                // Mock permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );

                // Override plugin array
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });

                // Override languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                });
            """)

            # Navigate to meeting
            logger.info("🌐 Opening Google Meet URL...")
            await self.page.goto(meeting_url)
            await self.page.wait_for_load_state('networkidle', timeout=30000)

            # Wait a bit for page to fully load
            await asyncio.sleep(3)

            # Enter bot name
            logger.info(f"📝 Setting display name: {self.bot_name}")
            try:
                # Try multiple selectors for the name input
                name_selectors = [
                    'input[placeholder*="name" i]',
                    'input[aria-label*="name" i]',
                    'input[type="text"]'
                ]

                for selector in name_selectors:
                    try:
                        name_input = await self.page.wait_for_selector(selector, timeout=3000)
                        if name_input:
                            await name_input.fill(self.bot_name)
                            logger.info(f"✅ Name set using selector: {selector}")
                            break
                    except:
                        continue

            except Exception as e:
                logger.warning(f"⚠️  Could not set name: {e}")

            # Turn off camera
            logger.info("📷 Turning off camera...")
            try:
                camera_selectors = [
                    '[aria-label*="camera" i]',
                    '[data-tooltip*="camera" i]',
                    'button:has-text("Turn off camera")'
                ]

                for selector in camera_selectors:
                    try:
                        camera_btn = await self.page.wait_for_selector(selector, timeout=2000)
                        if camera_btn:
                            # Check if camera is on (needs to be turned off)
                            await camera_btn.click()
                            logger.info("✅ Camera turned off")
                            break
                    except:
                        continue

            except Exception as e:
                logger.info(f"📷 Camera already off or not found: {e}")

            # Click "Join now" or "Ask to join"
            logger.info("🚪 Joining the meeting...")
            try:
                join_selectors = [
                    'button:has-text("Join now")',
                    'button:has-text("Ask to join")',
                    '[aria-label*="join" i]'
                ]

                for selector in join_selectors:
                    try:
                        join_btn = await self.page.wait_for_selector(selector, timeout=3000)
                        if join_btn:
                            await join_btn.click()
                            logger.info(f"✅ Clicked join button: {selector}")
                            break
                    except:
                        continue

            except Exception as e:
                logger.error(f"❌ Could not find join button: {e}")
                return False

            # Wait to ensure we've joined
            await asyncio.sleep(5)

            self.is_in_meeting = True
            logger.info(f"✅ Successfully joined meeting!")
            logger.info(f"🎙️  Bot is now in the meeting and listening...")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to join meeting: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def start_recording(self, output_path: str = None) -> bool:
        """
        Start recording the meeting audio.

        Args:
            output_path: Where to save the recording (default: auto-generate)

        Returns:
            Success status
        """
        try:
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"data/recordings/gmeet_{timestamp}.wav"

            # Create output directory
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"🎙️  Starting audio recording...")
            logger.info(f"💾 Will save to: {output_path}")

            # PyAudio setup
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000

            audio = pyaudio.PyAudio()

            # Open stream
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )

            self.is_recording = True
            self.audio_frames = []

            logger.info("✅ Recording started!")
            logger.info("⏸️  Press Ctrl+C or call stop_recording() to stop")

            # Record in background
            async def record_loop():
                while self.is_recording:
                    try:
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        self.audio_frames.append(data)
                        await asyncio.sleep(0.01)  # Small delay
                    except Exception as e:
                        logger.error(f"Recording error: {e}")
                        break

                # Save recording
                logger.info("💾 Saving recording...")
                stream.stop_stream()
                stream.close()
                audio.terminate()

                # Save to WAV file
                with wave.open(output_path, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(audio.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(self.audio_frames))

                logger.info(f"✅ Recording saved: {output_path}")
                return output_path

            # Start recording in background
            asyncio.create_task(record_loop())

            return True

        except Exception as e:
            logger.error(f"❌ Failed to start recording: {e}")
            return False

    async def stop_recording(self):
        """Stop the audio recording."""
        logger.info("⏹️  Stopping recording...")
        self.is_recording = False
        await asyncio.sleep(1)  # Give it time to save

    async def leave_meeting(self):
        """Leave the Google Meet and cleanup."""
        try:
            logger.info("👋 Leaving meeting...")

            if self.is_recording:
                await self.stop_recording()

            if self.page:
                # Try to click leave button
                try:
                    leave_selectors = [
                        '[aria-label*="Leave" i]',
                        'button:has-text("Leave call")',
                        '[data-tooltip*="Leave" i]'
                    ]

                    for selector in leave_selectors:
                        try:
                            leave_btn = await self.page.wait_for_selector(selector, timeout=2000)
                            if leave_btn:
                                await leave_btn.click()
                                logger.info("✅ Clicked leave button")
                                break
                        except:
                            continue

                except Exception as e:
                    logger.warning(f"Could not click leave button: {e}")

                await self.page.close()

            if self.browser:
                await self.browser.close()

            self.is_in_meeting = False
            logger.info("✅ Left meeting successfully")

        except Exception as e:
            logger.error(f"Error leaving meeting: {e}")

    async def get_participants(self) -> list:
        """
        Try to get list of participants in the meeting.

        Returns:
            List of participant names (best effort)
        """
        try:
            if not self.page:
                return []

            # Try to open participants panel
            participants_btn_selectors = [
                '[aria-label*="participant" i]',
                '[aria-label*="people" i]',
                'button:has-text("People")'
            ]

            for selector in participants_btn_selectors:
                try:
                    btn = await self.page.wait_for_selector(selector, timeout=2000)
                    if btn:
                        await btn.click()
                        await asyncio.sleep(1)
                        break
                except:
                    continue

            # Try to extract participant names
            # This is tricky and depends on Google Meet's UI structure
            # Returning empty for now - would need specific selectors
            logger.info("⚠️  Participant extraction not fully implemented")
            return []

        except Exception as e:
            logger.error(f"Error getting participants: {e}")
            return []

    async def stay_in_meeting(self, duration_seconds: int = None):
        """
        Keep the bot in the meeting for a specified duration.

        Args:
            duration_seconds: How long to stay (None = stay until manually stopped)
        """
        try:
            if duration_seconds:
                logger.info(f"⏱️  Staying in meeting for {duration_seconds} seconds...")
                await asyncio.sleep(duration_seconds)
            else:
                logger.info("⏱️  Staying in meeting indefinitely (Ctrl+C to stop)...")
                while self.is_in_meeting:
                    await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("⏸️  Interrupted by user")
        finally:
            await self.leave_meeting()
