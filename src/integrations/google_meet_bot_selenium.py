"""
Google Meet Bot using Selenium with undetected-chromedriver

This approach uses undetected-chromedriver which is specifically designed
to bypass automation detection by Google services.
"""
import asyncio
import wave
import pyaudio
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger

try:
    import undetected_chromedriver as uc
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
except ImportError:
    logger.error("undetected-chromedriver not installed. Run: pip install undetected-chromedriver selenium")
    raise


class GoogleMeetBotSelenium:
    """
    Bot that joins Google Meet using Selenium with undetected-chromedriver.

    This approach is more likely to succeed as it bypasses Google's automation detection.
    """

    def __init__(self, bot_name: str = "TeamSync AI Bot"):
        self.bot_name = bot_name
        self.driver = None
        self.is_in_meeting = False
        self.meeting_url: Optional[str] = None

        # Audio recording settings
        self.audio_frames = []
        self.is_recording = False

    def join_meeting(self, meeting_url: str) -> bool:
        """
        Join a Google Meet meeting.

        Args:
            meeting_url: Full Google Meet URL

        Returns:
            Success status
        """
        try:
            logger.info(f"🤖 Bot '{self.bot_name}' joining Google Meet: {meeting_url}")
            self.meeting_url = meeting_url

            # Create Chrome options
            options = uc.ChromeOptions()

            # Auto-allow camera and microphone
            options.add_argument('--use-fake-ui-for-media-stream')
            options.add_argument('--use-fake-device-for-media-stream')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--autoplay-policy=no-user-gesture-required')

            # Make it look more like a real browser
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--no-sandbox')
            options.add_argument('--window-size=1920,1080')

            # Set preferences
            prefs = {
                "profile.default_content_setting_values.media_stream_mic": 1,
                "profile.default_content_setting_values.media_stream_camera": 1,
                "profile.default_content_setting_values.notifications": 1,
            }
            options.add_experimental_option("prefs", prefs)

            logger.info("🌐 Launching Chrome browser...")

            # Create undetected Chrome driver
            # Specify Chrome version 142 to match installed Chrome
            self.driver = uc.Chrome(options=options, version_main=142, use_subprocess=True)

            # Navigate to meeting
            logger.info("📍 Opening Google Meet URL...")
            self.driver.get(meeting_url)

            # Wait for page to load
            time.sleep(5)

            # Check if we hit the unsupported page
            if "unsupported" in self.driver.current_url.lower():
                logger.error("❌ Google Meet says browser is unsupported")
                return False

            # Enter bot name
            logger.info(f"📝 Setting display name: {self.bot_name}")
            try:
                wait = WebDriverWait(self.driver, 10)

                # Try different selectors for name input
                name_input = None
                selectors = [
                    (By.CSS_SELECTOR, 'input[placeholder*="name" i]'),
                    (By.CSS_SELECTOR, 'input[aria-label*="name" i]'),
                    (By.CSS_SELECTOR, 'input[type="text"]'),
                    (By.XPATH, '//input[contains(@placeholder, "name")]'),
                ]

                for by, selector in selectors:
                    try:
                        name_input = wait.until(EC.presence_of_element_located((by, selector)))
                        if name_input:
                            name_input.clear()
                            name_input.send_keys(self.bot_name)
                            logger.info(f"✅ Name set successfully")
                            break
                    except TimeoutException:
                        continue

            except Exception as e:
                logger.warning(f"⚠️  Could not set name: {e}")

            # Turn off camera
            logger.info("📷 Turning off camera...")
            try:
                camera_selectors = [
                    (By.CSS_SELECTOR, '[aria-label*="camera" i]'),
                    (By.CSS_SELECTOR, '[data-is-muted="false"]'),
                    (By.XPATH, '//button[contains(@aria-label, "camera")]'),
                ]

                for by, selector in camera_selectors:
                    try:
                        camera_btn = WebDriverWait(self.driver, 5).until(
                            EC.element_to_be_clickable((by, selector))
                        )
                        camera_btn.click()
                        logger.info("✅ Camera turned off")
                        time.sleep(1)
                        break
                    except:
                        continue

            except Exception as e:
                logger.info(f"📷 Camera control skipped: {e}")

            # Click "Join now" or "Ask to join"
            logger.info("🚪 Joining the meeting...")
            try:
                join_selectors = [
                    (By.XPATH, '//button//span[contains(text(), "Join now")]'),
                    (By.XPATH, '//button//span[contains(text(), "Ask to join")]'),
                    (By.CSS_SELECTOR, '[aria-label*="join" i]'),
                    (By.XPATH, '//button[contains(., "Join")]'),
                ]

                joined = False
                for by, selector in join_selectors:
                    try:
                        join_btn = WebDriverWait(self.driver, 5).until(
                            EC.element_to_be_clickable((by, selector))
                        )
                        join_btn.click()
                        logger.info(f"✅ Clicked join button")
                        joined = True
                        break
                    except:
                        continue

                if not joined:
                    logger.error("❌ Could not find join button")
                    return False

            except Exception as e:
                logger.error(f"❌ Failed to click join: {e}")
                return False

            # Wait to ensure we've joined
            time.sleep(5)

            self.is_in_meeting = True
            logger.info(f"✅ Successfully joined meeting!")
            logger.info(f"🎙️  Bot is now in the meeting")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to join meeting: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def start_recording(self, output_path: str = None) -> bool:
        """Start recording the meeting audio from BlackHole."""
        try:
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"data/recordings/gmeet_{timestamp}.wav"

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"🎙️  Starting audio recording...")
            logger.info(f"💾 Will save to: {output_path}")

            # PyAudio setup
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 2  # BlackHole is 2-channel
            RATE = 48000  # BlackHole native sample rate

            audio = pyaudio.PyAudio()

            # Find BlackHole device
            blackhole_index = None
            for i in range(audio.get_device_count()):
                dev_info = audio.get_device_info_by_index(i)
                if "BlackHole" in dev_info['name']:
                    blackhole_index = i
                    logger.info(f"Found BlackHole device: {dev_info['name']}")
                    break

            if blackhole_index is None:
                logger.error("BlackHole not found! Install with: brew install blackhole-2ch")
                logger.info("Using default input device instead...")
                blackhole_index = None

            # Open stream from BlackHole
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=blackhole_index,
                frames_per_buffer=CHUNK
            )

            self.output_path = output_path  # Store for later

            self.is_recording = True
            self.audio_frames = []

            logger.info("✅ Recording started!")

            # Record in background
            async def record_loop():
                while self.is_recording:
                    try:
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        self.audio_frames.append(data)
                        await asyncio.sleep(0.01)
                    except Exception as e:
                        logger.error(f"Recording error: {e}")
                        break

                # Save recording
                logger.info("💾 Saving recording...")
                stream.stop_stream()
                stream.close()
                audio.terminate()

                with wave.open(output_path, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(audio.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(self.audio_frames))

                logger.info(f"✅ Recording saved: {output_path}")
                return output_path

            asyncio.create_task(record_loop())
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start recording: {e}")
            return False

    async def stop_recording(self):
        """Stop the audio recording."""
        logger.info("⏹️  Stopping recording...")
        self.is_recording = False
        await asyncio.sleep(1)

    def leave_meeting(self):
        """Leave the Google Meet and cleanup."""
        try:
            logger.info("👋 Leaving meeting...")

            if self.driver:
                # Try to click leave button
                try:
                    leave_selectors = [
                        (By.XPATH, '//button[contains(@aria-label, "Leave")]'),
                        (By.XPATH, '//button//span[contains(text(), "Leave")]'),
                    ]

                    for by, selector in leave_selectors:
                        try:
                            leave_btn = WebDriverWait(self.driver, 3).until(
                                EC.element_to_be_clickable((by, selector))
                            )
                            leave_btn.click()
                            logger.info("✅ Clicked leave button")
                            break
                        except:
                            continue

                except Exception as e:
                    logger.warning(f"Could not click leave button: {e}")

                time.sleep(2)
                self.driver.quit()

            self.is_in_meeting = False
            logger.info("✅ Left meeting successfully")

        except Exception as e:
            logger.error(f"Error leaving meeting: {e}")

    async def stay_in_meeting_async(self, duration_seconds: int = None):
        """
        Keep the bot in the meeting for a specified duration (async version).
        Use this when recording audio to allow the recording task to run.

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
        except asyncio.CancelledError:
            logger.info("⏸️  Meeting task cancelled")
        finally:
            self.leave_meeting()

    def stay_in_meeting(self, duration_seconds: int = None):
        """
        Keep the bot in the meeting for a specified duration (sync version).
        WARNING: This blocks the event loop - use stay_in_meeting_async() when recording!

        Args:
            duration_seconds: How long to stay (None = stay until manually stopped)
        """
        try:
            if duration_seconds:
                logger.info(f"⏱️  Staying in meeting for {duration_seconds} seconds...")
                time.sleep(duration_seconds)
            else:
                logger.info("⏱️  Staying in meeting indefinitely (Ctrl+C to stop)...")
                while self.is_in_meeting:
                    time.sleep(1)

        except KeyboardInterrupt:
            logger.info("⏸️  Interrupted by user")
        finally:
            self.leave_meeting()
