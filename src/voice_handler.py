"""
Voice handling module for speech-to-text and text-to-speech.
"""
import os
import tempfile
import wave
from pathlib import Path
from typing import Optional, Literal
import numpy as np
import sounddevice as sd
import soundfile as sf
from rich.console import Console

console = Console()

# TTS Options
TTS_ENGINE = Literal["gtts", "openai", "elevenlabs"]


class VoiceHandler:
    """Handle voice input/output for the interview agent."""
    
    def __init__(
        self,
        tts_engine: TTS_ENGINE = "gtts",
        whisper_model: str = "base",
        sample_rate: int = 16000,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize voice handler.
        
        Args:
            tts_engine: TTS engine to use ('gtts', 'openai', or 'elevenlabs')
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            sample_rate: Audio sample rate in Hz
            openai_api_key: OpenAI API key for TTS (if using openai engine)
        """
        self.tts_engine = tts_engine
        self.sample_rate = sample_rate
        self.whisper_model_name = whisper_model
        
        # Initialize Whisper model
        console.print(f"[yellow]Loading Whisper model '{whisper_model}'...[/yellow]")
        import whisper
        self.whisper_model = whisper.load_model(whisper_model)
        console.print("[green]✓ Whisper model loaded[/green]")
        
        # Initialize TTS engine
        if tts_engine == "openai":
            if not openai_api_key:
                openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI TTS")
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize pygame for audio playback
        import pygame
        pygame.mixer.init()
        self.pygame = pygame
        
        # Recording state
        self.is_recording = False
        self.recording_data = []
    
    def record_audio(
        self,
        duration: Optional[float] = None,
        silence_threshold: float = 0.01,
        silence_duration: float = 2.0
    ) -> np.ndarray:
        """
        Record audio from microphone.
        
        Args:
            duration: Fixed duration in seconds (None for silence detection)
            silence_threshold: Amplitude threshold for silence detection
            silence_duration: Duration of silence before stopping (seconds)
            
        Returns:
            Recorded audio as numpy array
        """
        console.print("\n[bold cyan]🎤 Recording... (speak now)[/bold cyan]")
        
        if duration:
            # Fixed duration recording
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            console.print("[green]✓ Recording complete[/green]")
            return audio.flatten()
        
        else:
            # Record with silence detection
            return self._record_with_silence_detection(
                silence_threshold,
                silence_duration
            )
    
    def _record_with_silence_detection(
        self,
        silence_threshold: float,
        silence_duration: float
    ) -> np.ndarray:
        """Record audio until silence is detected."""
        chunk_duration = 0.1  # 100ms chunks
        chunk_samples = int(chunk_duration * self.sample_rate)
        silence_chunks = int(silence_duration / chunk_duration)
        
        audio_chunks = []
        silent_chunks = 0
        started = False
        
        console.print("[dim]Listening for speech...[/dim]")
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            ) as stream:
                while True:
                    chunk, _ = stream.read(chunk_samples)
                    chunk = chunk.flatten()
                    
                    # Calculate volume
                    volume = np.abs(chunk).mean()
                    
                    # Check if speech detected
                    if volume > silence_threshold:
                        if not started:
                            console.print("[green]🔴 Speech detected, recording...[/green]")
                            started = True
                        audio_chunks.append(chunk)
                        silent_chunks = 0
                    elif started:
                        audio_chunks.append(chunk)
                        silent_chunks += 1
                        
                        # Stop if enough silence
                        if silent_chunks >= silence_chunks:
                            console.print("[green]✓ Recording complete[/green]")
                            break
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Recording cancelled[/yellow]")
        
        if not audio_chunks:
            console.print("[red]No audio recorded[/red]")
            return np.array([])
        
        return np.concatenate(audio_chunks)
    
    def transcribe_audio(self, audio: np.ndarray, language: Optional[str] = None) -> str:
        """
        Transcribe audio to text using Whisper.
        
        Args:
            audio: Audio data as numpy array
            language: Language code (None for auto-detection)
            
        Returns:
            Transcribed text
        """
        if len(audio) == 0:
            return ""
        
        console.print("[yellow]Transcribing...[/yellow]")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Write audio to file
            sf.write(tmp_path, audio, self.sample_rate)
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                tmp_path,
                language=language,
                fp16=False
            )
            
            text = result["text"].strip()
            
            if text:
                console.print(f"[green]✓ Transcribed:[/green] {text}")
            else:
                console.print("[yellow]⚠ No speech detected[/yellow]")
            
            return text
            
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
    
    def speak(self, text: str, language: str = "en") -> None:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
            language: Language code
        """
        if not text:
            return
        
        console.print(f"[cyan]🔊 Agent:[/cyan] {text}")
        
        # Generate audio file
        audio_file = self._generate_speech(text, language)
        
        if audio_file:
            # Play audio
            self._play_audio(audio_file)
            
            # Clean up
            Path(audio_file).unlink(missing_ok=True)
    
    def _generate_speech(self, text: str, language: str) -> Optional[str]:
        """Generate speech audio file from text."""
        tmp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            if self.tts_engine == "gtts":
                return self._gtts_generate(text, language, tmp_path)
            elif self.tts_engine == "openai":
                return self._openai_tts_generate(text, tmp_path)
            else:
                console.print(f"[red]Unsupported TTS engine: {self.tts_engine}[/red]")
                return None
        except Exception as e:
            console.print(f"[red]TTS Error: {str(e)}[/red]")
            Path(tmp_path).unlink(missing_ok=True)
            return None
    
    def _gtts_generate(self, text: str, language: str, output_path: str) -> str:
        """Generate speech using Google TTS (free)."""
        from gtts import gTTS
        
        # Map language codes
        lang_map = {"en": "en", "es": "es", "fr": "fr", "de": "de"}
        gtts_lang = lang_map.get(language, "en")
        
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        tts.save(output_path)
        
        return output_path
    
    def _openai_tts_generate(self, text: str, output_path: str) -> str:
        """Generate speech using OpenAI TTS (premium)."""
        response = self.openai_client.audio.speech.create(
            model="tts-1",  # or "tts-1-hd" for higher quality
            voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
            input=text
        )
        
        response.stream_to_file(output_path)
        return output_path
    
    def _play_audio(self, audio_file: str) -> None:
        """Play audio file."""
        try:
            self.pygame.mixer.music.load(audio_file)
            self.pygame.mixer.music.play()
            
            # Wait for playback to finish
            while self.pygame.mixer.music.get_busy():
                self.pygame.time.Clock().tick(10)
                
        except Exception as e:
            console.print(f"[red]Playback error: {str(e)}[/red]")
    
    def listen_and_transcribe(
        self,
        duration: Optional[float] = None,
        language: Optional[str] = None
    ) -> str:
        """
        Listen to audio and transcribe it.
        
        Args:
            duration: Recording duration (None for silence detection)
            language: Language for transcription
            
        Returns:
            Transcribed text
        """
        audio = self.record_audio(duration=duration)
        
        if len(audio) == 0:
            return ""
        
        return self.transcribe_audio(audio, language=language)
    
    def test_microphone(self) -> bool:
        """
        Test microphone functionality.
        
        Returns:
            True if microphone works, False otherwise
        """
        console.print("\n[bold cyan]Testing Microphone...[/bold cyan]")
        console.print("Speak something for 3 seconds...")
        
        try:
            audio = self.record_audio(duration=3.0)
            text = self.transcribe_audio(audio)
            
            if text:
                console.print(f"[green]✓ Microphone test successful![/green]")
                console.print(f"[green]Heard:[/green] {text}")
                return True
            else:
                console.print("[yellow]⚠ No speech detected. Check microphone.[/yellow]")
                return False
                
        except Exception as e:
            console.print(f"[red]✗ Microphone test failed: {str(e)}[/red]")
            return False
    
    def test_speakers(self) -> bool:
        """
        Test speaker functionality.
        
        Returns:
            True if speakers work, False otherwise
        """
        console.print("\n[bold cyan]Testing Speakers...[/bold cyan]")
        
        try:
            self.speak("Hello! This is a speaker test. Can you hear me?")
            console.print("[green]✓ Speaker test complete[/green]")
            
            response = input("Did you hear the message? (y/n): ")
            return response.lower() == 'y'
            
        except Exception as e:
            console.print(f"[red]✗ Speaker test failed: {str(e)}[/red]")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'pygame'):
            self.pygame.mixer.quit()


# Convenience functions
def create_voice_handler(
    tts_engine: TTS_ENGINE = "gtts",
    whisper_model: str = "base"
) -> VoiceHandler:
    """
    Create and return a VoiceHandler instance.
    
    Args:
        tts_engine: TTS engine to use
        whisper_model: Whisper model size
        
    Returns:
        VoiceHandler instance
    """
    return VoiceHandler(tts_engine=tts_engine, whisper_model=whisper_model)


def test_voice_setup() -> bool:
    """
    Test complete voice setup.
    
    Returns:
        True if everything works, False otherwise
    """
    console.print("\n[bold cyan]Voice Setup Test[/bold cyan]\n")
    
    try:
        handler = create_voice_handler()
        
        # Test microphone
        mic_ok = handler.test_microphone()
        
        # Test speakers
        speakers_ok = handler.test_speakers()
        
        handler.cleanup()
        
        if mic_ok and speakers_ok:
            console.print("\n[green]✓ All voice tests passed![/green]")
            return True
        else:
            console.print("\n[yellow]⚠ Some voice tests failed[/yellow]")
            return False
            
    except Exception as e:
        console.print(f"\n[red]✗ Voice setup test failed: {str(e)}[/red]")
        return False


if __name__ == "__main__":
    # Run voice setup test
    test_voice_setup()
