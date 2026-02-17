"""
Watty Voice Engine — Local-First Speech Pipeline
=================================================
One tool: watty_voice(action=...). 9 actions.
speak, listen, conversation, hotword_start, hotword_stop,
list_voices, set_voice, stop, download_models.

Degradation chain (each falls back to the next):
  TTS:   Piper (ONNX, local) -> edge-tts (cloud)
  STT:   faster-whisper (CTranslate2) -> Google STT (cloud)
  VAD:   Silero (torch) -> energy threshold (numpy)
  Audio: sounddevice -> speech_recognition.Microphone -> pygame

February 2026
"""

import asyncio, json, struct, threading, time, wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from mcp.types import Tool, TextContent
from watty.config import WATTY_HOME, VOICE_MODELS_DIR

# ── Paths & Constants ────────────────────────────────────────
VOICE_DIR = WATTY_HOME / "voice"
VOICE_DIR.mkdir(parents=True, exist_ok=True)
VOICE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
PIPER_MODELS_DIR = VOICE_MODELS_DIR / "piper"
PIPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
CONVO_LOG = VOICE_DIR / "conversations.jsonl"
VOICE_CONFIG_PATH = VOICE_DIR / "config.json"
DEFAULT_VOICE = "en-US-AndrewNeural"
RATE = 16000
WAKE_WORDS = ["hey watty", "hey wadi", "hey woody", "hey body", "a watty", "hey what he"]
DEFAULT_CONFIG = {
    "tts_engine": "piper", "tts_model": "en_US-amy-medium",
    "stt_engine": "faster-whisper", "stt_model_size": "base",
    "stt_hotword_model_size": "tiny", "vad_enabled": True,
    "vad_threshold": 0.5, "vad_min_speech_ms": 250, "vad_min_silence_ms": 500,
}

def _load_config() -> dict:
    if VOICE_CONFIG_PATH.exists():
        try: return json.loads(VOICE_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception: pass
    return DEFAULT_CONFIG.copy()

def _save_config(cfg: dict):
    VOICE_CONFIG_PATH.write_text(json.dumps(cfg, indent=4), encoding="utf-8")

# ── Module State ─────────────────────────────────────────────
_current_voice = DEFAULT_VOICE
_mixer_initialized = False
_speaking = False
_listening = False
_hotword_thread: Optional[threading.Thread] = None
_hotword_running = False
_piper: Optional["PiperTTS"] = None
_whisper: Optional["WhisperSTT"] = None
_vad: Optional["SileroVAD"] = None
_capture: Optional["AudioCapture"] = None

def _get_piper() -> "PiperTTS":
    global _piper
    if _piper is None: _piper = PiperTTS()
    return _piper

def _get_whisper() -> "WhisperSTT":
    global _whisper
    if _whisper is None: _whisper = WhisperSTT()
    return _whisper

def _get_vad() -> "SileroVAD":
    global _vad
    if _vad is None: _vad = SileroVAD()
    return _vad

def _get_capture() -> "AudioCapture":
    global _capture
    if _capture is None: _capture = AudioCapture()
    return _capture

# ── Logging ──────────────────────────────────────────────────
def _log_voice_event(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        with open(VOICE_DIR / "voice.log", "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception: pass

def _log_conversation(hugo_said: str, watty_said: str):
    entry = {"timestamp": datetime.now().isoformat(), "hugo": hugo_said, "watty": watty_said}
    try:
        with open(CONVO_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception: pass

# ── 1. PiperTTS ──────────────────────────────────────────────
class PiperTTS:
    """Local ONNX TTS. Falls back to edge-tts."""
    def __init__(self):
        self.available = False
        try:
            import piper; self.available = True
        except ImportError: pass
        self._config = _load_config()

    def synthesize(self, text: str, voice: Optional[str] = None, rate: str = "+0%") -> bytes:
        if self.available:
            try: return self._synth_piper(text, voice)
            except Exception as e:
                _log_voice_event(f"Piper failed ({e}), falling back to edge-tts")
        return self._synth_edge(text, voice, rate)

    def _synth_piper(self, text: str, voice: Optional[str] = None) -> bytes:
        model_name = voice or self._config.get("tts_model", "en_US-amy-medium")
        model_path = PIPER_MODELS_DIR / f"{model_name}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Piper model not found: {model_path}")
        from piper import PiperVoice
        pv = PiperVoice.load(str(model_path))
        return b"".join(chunk for chunk in pv.synthesize_stream_raw(text))

    def _synth_edge(self, text: str, voice: Optional[str] = None, rate: str = "+0%") -> bytes:
        import edge_tts
        edge_voice = voice if voice and "Neural" in str(voice) else _current_voice
        tmp_path = str(VOICE_DIR / "current_speech.mp3")
        loop = asyncio.new_event_loop()
        loop.run_until_complete(edge_tts.Communicate(text, edge_voice, rate=rate).save(tmp_path))
        loop.close()
        return Path(tmp_path).read_bytes()

    def list_voices(self) -> list[dict]:
        voices = []
        for onnx in PIPER_MODELS_DIR.glob("*.onnx"):
            cfg_path = onnx.with_suffix(".onnx.json")
            meta = {}
            if cfg_path.exists():
                try: meta = json.loads(cfg_path.read_text(encoding="utf-8"))
                except Exception: pass
            voices.append({"name": onnx.stem, "path": str(onnx),
                           "language": meta.get("language", {}).get("code", "en"),
                           "quality": meta.get("quality", "unknown")})
        return voices

    def download_model(self, model_name: str) -> str:
        try: from huggingface_hub import hf_hub_download
        except ImportError: return "huggingface_hub not installed. Run: pip install huggingface-hub"
        lang = model_name.split("-")[0] if "-" in model_name else "en_US"
        lang_dir = lang.replace("_", "/")
        try:
            for suffix in [".onnx", ".onnx.json"]:
                fname = f"{lang_dir}/{model_name}/low/{model_name}{suffix}"
                try: hf_hub_download(repo_id="rhasspy/piper-voices", filename=fname,
                                     local_dir=str(PIPER_MODELS_DIR), local_dir_use_symlinks=False)
                except Exception:
                    fname = f"{lang_dir}/{model_name}/medium/{model_name}{suffix}"
                    hf_hub_download(repo_id="rhasspy/piper-voices", filename=fname,
                                    local_dir=str(PIPER_MODELS_DIR), local_dir_use_symlinks=False)
            return f"Downloaded {model_name} to {PIPER_MODELS_DIR}"
        except Exception as e: return f"Download failed: {e}"

# ── 2. WhisperSTT ────────────────────────────────────────────
class WhisperSTT:
    """Local CTranslate2 STT. Falls back to Google Speech Recognition."""
    def __init__(self):
        self.available = False
        self._models: dict = {}
        try:
            import faster_whisper; self.available = True
        except ImportError: pass
        self._config = _load_config()

    def _get_model(self, size: str = "base"):
        if size not in self._models:
            from faster_whisper import WhisperModel
            device = "cuda" if self._cuda_available() else "cpu"
            compute = "float16" if device == "cuda" else "int8"
            self._models[size] = WhisperModel(size, device=device, compute_type=compute)
            _log_voice_event(f"Loaded whisper model: {size} on {device}")
        return self._models[size]

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch; return torch.cuda.is_available()
        except ImportError: return False

    def transcribe(self, audio_data: np.ndarray, language: str = "en",
                   model_size: Optional[str] = None) -> str:
        if self.available:
            try: return self._transcribe_whisper(audio_data, language, model_size)
            except Exception as e:
                _log_voice_event(f"Whisper failed ({e}), falling back to Google STT")
        return self._transcribe_google(audio_data)

    def _transcribe_whisper(self, audio_data: np.ndarray, language: str,
                            model_size: Optional[str] = None) -> str:
        size = model_size or self._config.get("stt_model_size", "base")
        audio_f = audio_data.astype(np.float32) / 32768.0 if audio_data.dtype == np.int16 \
            else audio_data.astype(np.float32)
        segments, _ = self._get_model(size).transcribe(audio_f, language=language, beam_size=3)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def _transcribe_google(self, audio_data: np.ndarray) -> str:
        import speech_recognition as sr
        audio_int16 = audio_data if audio_data.dtype == np.int16 \
            else (audio_data * 32768).astype(np.int16)
        audio_obj = sr.AudioData(audio_int16.tobytes(), RATE, 2)
        try: return sr.Recognizer().recognize_google(audio_obj).strip()
        except sr.UnknownValueError: return ""
        except sr.RequestError as e: return f"(transcription error: {e})"

# ── 3. SileroVAD ─────────────────────────────────────────────
class SileroVAD:
    """Neural VAD. Falls back to energy threshold."""
    def __init__(self):
        self.available = False
        self._model = None
        self._torch = None
        self._config = _load_config()
        try:
            import torch
            self._torch = torch
            model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                      force_reload=False, trust_repo=True)
            self._model = model
            self.available = True
            _log_voice_event("Silero VAD loaded")
        except Exception: pass

    def is_speech(self, audio_chunk: np.ndarray, threshold: Optional[float] = None) -> bool:
        thresh = threshold or self._config.get("vad_threshold", 0.5)
        if self.available and self._model is not None:
            try:
                audio = audio_chunk.astype(np.float32) / 32768.0 if audio_chunk.dtype == np.int16 \
                    else audio_chunk.astype(np.float32)
                return self._model(self._torch.from_numpy(audio), RATE).item() > thresh
            except Exception: pass
        return self._energy_vad(audio_chunk)

    def _energy_vad(self, chunk: np.ndarray, threshold: int = 300) -> bool:
        if chunk.dtype != np.int16: chunk = (chunk * 32768).astype(np.int16)
        return np.sqrt(np.mean(chunk.astype(np.float64) ** 2)) > threshold

    def reset(self):
        if self.available and self._model is not None:
            try: self._model.reset_states()
            except Exception: pass

# ── 4. AudioCapture ──────────────────────────────────────────
class AudioCapture:
    """Low-latency audio capture. Falls back to speech_recognition.Microphone."""
    def __init__(self):
        self.available = False
        self._sd = None
        self._stream = None
        try:
            import sounddevice as sd; self._sd = sd; self.available = True
        except ImportError: pass

    def start(self, callback: Callable[[np.ndarray], None]):
        if self.available:
            self._stream = self._sd.InputStream(
                samplerate=RATE, channels=1, dtype="int16",
                blocksize=int(RATE * 0.032),
                callback=lambda indata, frames, t, status: callback(indata[:, 0].copy()))
            self._stream.start()

    def stop(self):
        if self._stream is not None:
            try: self._stream.stop(); self._stream.close()
            except Exception: pass
            self._stream = None

    def record(self, duration_sec: float) -> np.ndarray:
        if self.available:
            audio = self._sd.rec(int(RATE * duration_sec), samplerate=RATE, channels=1, dtype="int16")
            self._sd.wait()
            return audio[:, 0]
        try:
            import speech_recognition as sr
            rec = sr.Recognizer(); mic = sr.Microphone(sample_rate=RATE)
            with mic as source:
                rec.adjust_for_ambient_noise(source, duration=0.3)
                audio = rec.listen(source, timeout=5, phrase_time_limit=int(duration_sec))
            return np.frombuffer(audio.get_raw_data(convert_rate=RATE, convert_width=2), dtype=np.int16)
        except Exception as e:
            _log_voice_event(f"AudioCapture fallback failed: {e}")
            return np.zeros(int(RATE * duration_sec), dtype=np.int16)

# ── Audio Playback ───────────────────────────────────────────
def _init_mixer():
    global _mixer_initialized
    if _mixer_initialized: return True
    try:
        import pygame; pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=2048)
        _mixer_initialized = True; return True
    except Exception: return False

def _play_audio(audio_bytes: bytes, is_pcm: bool = False):
    if is_pcm: _play_pcm(audio_bytes)
    else: _play_mp3(audio_bytes)

def _play_mp3(mp3_bytes: bytes):
    import pygame
    tmp = VOICE_DIR / "current_speech.mp3"; tmp.write_bytes(mp3_bytes)
    if not _init_mixer(): return
    pygame.mixer.music.load(str(tmp)); pygame.mixer.music.play()
    while pygame.mixer.music.get_busy(): pygame.time.wait(100)

def _play_pcm(pcm_bytes: bytes):
    try:
        import sounddevice as sd
        sd.play(np.frombuffer(pcm_bytes, dtype=np.int16), samplerate=RATE); sd.wait(); return
    except Exception: pass
    wav_path = VOICE_DIR / "current_speech.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(RATE); wf.writeframes(pcm_bytes)
    try:
        import pygame
        if not _init_mixer(): return
        pygame.mixer.music.load(str(wav_path)); pygame.mixer.music.play()
        while pygame.mixer.music.get_busy(): pygame.time.wait(100)
    except Exception: pass

# ── Speak / Stop / Listen ────────────────────────────────────
def _speak_sync(text: str, voice: str, rate: str = "+0%"):
    global _speaking
    _speaking = True
    try:
        piper = _get_piper()
        audio = piper.synthesize(text, voice, rate)
        _play_audio(audio, is_pcm=piper.available)
    except Exception as e: _log_voice_event(f"Speak error: {e}")
    finally: _speaking = False

def _speak_async(text: str, voice: str, rate: str = "+0%"):
    global _speaking
    if _speaking: _stop_speaking()
    t = threading.Thread(target=_speak_sync, args=(text, voice, rate), daemon=True, name="watty-voice")
    t.start(); return t

def _stop_speaking():
    global _speaking
    try:
        import pygame
        if _mixer_initialized: pygame.mixer.music.stop()
    except Exception: pass
    try:
        import sounddevice as sd; sd.stop()
    except Exception: pass
    _speaking = False

def _listen_mic(max_seconds: int = 10) -> str:
    audio = _get_capture().record(float(max_seconds))
    if audio is None or len(audio) == 0 or np.max(np.abs(audio)) < 50: return ""
    return _get_whisper().transcribe(audio)

# ── Response Generation ──────────────────────────────────────
def _generate_response(request: str) -> str:
    rl = request.lower()
    try:
        from watty.brain import Brain
        brain = Brain()
        if any(w in rl for w in ["how are you", "status", "how's it going", "what's up"]):
            return f"I'm running well. {brain.stats().get('total_memories', 0)} memories in my brain. All systems operational."
        if any(w in rl for w in ["what do you know", "tell me about", "remember", "recall"]):
            results = brain.recall(request, top_k=3)
            if results: return f"Here's what I found: {results[0].get('content', '')[:200]}"
            return "I don't have anything on that in my memory."
        if any(w in rl for w in ["what time", "what day"]):
            now = datetime.now()
            return f"It's {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d')}."
        if any(w in rl for w in ["training", "gpu", "model"]):
            return "The training pipeline is running on the GPU. I'll check the progress for you."
        return f"I heard you say: {request}. I'll work on that."
    except Exception:
        return f"I heard: {request}. Let me think about that."

# ── Hotword Loop (VAD-gated) ─────────────────────────────────
def _hotword_loop():
    global _hotword_running
    vad, capture, whisper = _get_vad(), _get_capture(), _get_whisper()
    config = _load_config()
    min_speech_chunks = max(1, int(config.get("vad_min_speech_ms", 250) / 32))
    min_silence_chunks = max(1, int(config.get("vad_min_silence_ms", 500) / 32))
    chunk_samples = int(RATE * 0.032)
    ring_buf_size = int(RATE * 0.3)  # 300ms pre-speech padding

    _log_voice_event("Hotword listener started (VAD-gated). Listening for 'Hey Watty'...")

    while _hotword_running:
        try:
            ring_buf = np.zeros(ring_buf_size, dtype=np.int16)
            speech_chunks: list[np.ndarray] = []
            speech_count = silence_count = 0
            in_speech = False

            if capture.available:
                audio_queue: list[np.ndarray] = []
                lock = threading.Lock()
                capture.start(lambda c: (lock.acquire(), audio_queue.append(c.copy()), lock.release()))
                try:
                    for _ in range(int(30 / 0.032)):  # 30s max per cycle
                        if not _hotword_running: break
                        time.sleep(0.032)
                        with lock: chunks, audio_queue[:] = list(audio_queue), []
                        for chunk in chunks:
                            if len(chunk) < chunk_samples: continue
                            ring_buf = np.roll(ring_buf, -len(chunk))
                            ring_buf[-len(chunk):] = chunk[:ring_buf_size]
                            if vad.is_speech(chunk):
                                speech_count += 1; silence_count = 0
                                if speech_count >= min_speech_chunks:
                                    if not in_speech:
                                        in_speech = True; speech_chunks.append(ring_buf.copy())
                                    speech_chunks.append(chunk)
                            else:
                                silence_count += 1
                                if in_speech:
                                    speech_chunks.append(chunk)
                                    if silence_count >= min_silence_chunks: break
                                else: speech_count = 0
                        if in_speech and silence_count >= min_silence_chunks: break
                finally: capture.stop(); vad.reset()
            else:
                audio_data = capture.record(3.0)
                if np.max(np.abs(audio_data)) > 300:
                    speech_chunks = [audio_data]; in_speech = True

            if not in_speech or not speech_chunks: continue

            # Transcribe with tiny model for hotword detection
            full_audio = np.concatenate(speech_chunks)
            text = whisper.transcribe(full_audio,
                                      model_size=config.get("stt_hotword_model_size", "tiny")).lower().strip()
            if not text or not any(w in text for w in WAKE_WORDS): continue

            _log_voice_event(f"Wake word detected in: '{text}'")
            _speak_sync("Yes Hugo?", _current_voice)

            # Listen for command
            _log_voice_event("Listening for command...")
            cmd_audio = capture.record(15.0)
            if cmd_audio is None or len(cmd_audio) == 0 or np.max(np.abs(cmd_audio)) < 50:
                _speak_sync("I didn't catch that.", _current_voice); continue

            request = whisper.transcribe(cmd_audio).strip()
            if not request:
                _speak_sync("I didn't understand. Try again.", _current_voice); continue

            _log_voice_event(f"Hugo said: '{request}'")
            response = _generate_response(request)
            _speak_sync(response, _current_voice)
            _log_conversation(request, response)
        except Exception as e:
            _log_voice_event(f"Hotword error: {e}"); time.sleep(1)

    _log_voice_event("Hotword listener stopped.")

def _start_hotword() -> str:
    global _hotword_thread, _hotword_running
    if _hotword_running: return "Hotword listener is already running."
    _hotword_running = True
    _hotword_thread = threading.Thread(target=_hotword_loop, daemon=True, name="watty-hotword")
    _hotword_thread.start()
    return "Hotword listener started (VAD-gated). Say 'Hey Watty' to talk."

def _stop_hotword() -> str:
    global _hotword_running
    _hotword_running = False
    return "Hotword listener stopped."

# ── Download Models ──────────────────────────────────────────
def _download_models() -> str:
    results, config = [], _load_config()
    piper = _get_piper()
    results.append(f"TTS ({config.get('tts_model', 'en_US-amy-medium')}): "
                   f"{piper.download_model(config.get('tts_model', 'en_US-amy-medium'))}")
    whisper = _get_whisper()
    if whisper.available:
        for label, key in [("hotword", "stt_hotword_model_size"), ("main", "stt_model_size")]:
            size = config.get(key, "base" if label == "main" else "tiny")
            try: whisper._get_model(size); results.append(f"STT {label} ({size}): loaded OK")
            except Exception as e: results.append(f"STT {label} ({size}): {e}")
    else: results.append("STT: faster-whisper not installed, using Google STT fallback")
    results.append(f"VAD: {'Silero (torch)' if _get_vad().available else 'energy threshold (fallback)'}")
    results.append(f"Audio: {'sounddevice' if _get_capture().available else 'speech_recognition (fallback)'}")
    _save_config(config); results.append(f"Config saved to {VOICE_CONFIG_PATH}")
    return "\n".join(results)

# ── Voice Listing ────────────────────────────────────────────
async def _list_voices_combined() -> str:
    lines = []
    local = _get_piper().list_voices()
    if local:
        lines.append(f"Local Piper voices ({len(local)}):")
        lines.extend(f"  {v['name']} — {v['quality']} ({v['language']})" for v in local)
        lines.append("")
    try:
        import edge_tts
        voices = await edge_tts.list_voices()
        en = [v for v in voices if v["Locale"].startswith("en-")]
        lines.append(f"Cloud edge-tts voices ({len(en)} English):")
        for v in en[:30]:
            marker = " <<< current" if v["ShortName"] == _current_voice else ""
            lines.append(f"  {v['ShortName']} — {v['Gender']}{marker}")
    except Exception as e: lines.append(f"Edge-tts listing error: {e}")
    return "\n".join(lines)

# ── Tool Definition ──────────────────────────────────────────
TOOLS = [Tool(
    name="watty_voice",
    description=(
        "Watty's voice engine — local-first speech pipeline.\n"
        "TTS: Piper (local ONNX) -> edge-tts (cloud fallback).\n"
        "STT: faster-whisper (local) -> Google STT (cloud fallback).\n"
        "VAD: Silero neural -> energy threshold fallback.\n"
        "Actions: speak, listen, conversation, hotword_start, hotword_stop, "
        "list_voices, set_voice, stop, download_models."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "Action to perform",
                       "enum": ["speak", "listen", "conversation", "hotword_start",
                                "hotword_stop", "list_voices", "set_voice", "stop",
                                "download_models"]},
            "text": {"type": "string", "description": "speak/conversation: Text to say aloud"},
            "voice": {"type": "string",
                      "description": "set_voice: Voice name (e.g. 'en-US-AndrewNeural' or piper model)"},
            "rate": {"type": "string",
                     "description": "speak: Speech rate (e.g. '+20%', '-10%'). Default: '+0%'"},
            "duration": {"type": "integer", "description": "listen: Max recording seconds (default: 10)"},
        },
        "required": ["action"],
    },
)]

# ── Handler ──────────────────────────────────────────────────
async def handle_voice(arguments: dict) -> list[TextContent]:
    global _current_voice, _listening
    action = arguments.get("action", "speak")

    if action == "speak":
        text = arguments.get("text", "")
        if not text: return [TextContent(type="text", text="Nothing to say.")]
        rate = arguments.get("rate", "+0%")
        _speak_async(text, _current_voice, rate=rate)
        engine = "Piper (local)" if _get_piper().available else "edge-tts (cloud)"
        return [TextContent(type="text",
            text=f"Speaking [{engine}]: \"{text[:100]}{'...' if len(text) > 100 else ''}\" (voice: {_current_voice})")]

    elif action == "listen":
        duration = arguments.get("duration", 10)
        _listening = True
        try:
            transcript = _listen_mic(max_seconds=duration); _listening = False
            if not transcript: return [TextContent(type="text", text="(silence — nothing detected)")]
            engine = "faster-whisper (local)" if _get_whisper().available else "Google STT (cloud)"
            return [TextContent(type="text", text=f"Hugo said [{engine}]: \"{transcript}\"")]
        except Exception as e:
            _listening = False; return [TextContent(type="text", text=f"Listen error: {e}")]

    elif action == "conversation":
        text = arguments.get("text", ""); rate = arguments.get("rate", "+0%")
        duration = arguments.get("duration", 10)
        if text: _speak_sync(text, _current_voice, rate=rate)
        time.sleep(0.3); _listening = True
        try:
            transcript = _listen_mic(max_seconds=duration); _listening = False
            result = f"Watty said: \"{text[:80]}{'...' if len(text) > 80 else ''}\"\n"
            if transcript:
                result += f"Hugo replied: \"{transcript}\""; _log_conversation(transcript, text)
            else: result += "Hugo replied: (silence)"
            return [TextContent(type="text", text=result)]
        except Exception as e:
            _listening = False; return [TextContent(type="text", text=f"Conversation error: {e}")]

    elif action == "hotword_start":
        return [TextContent(type="text", text=_start_hotword())]
    elif action == "hotword_stop":
        return [TextContent(type="text", text=_stop_hotword())]

    elif action == "list_voices":
        try: return [TextContent(type="text", text=await _list_voices_combined())]
        except Exception as e: return [TextContent(type="text", text=f"Error listing voices: {e}")]

    elif action == "set_voice":
        voice = arguments.get("voice", "")
        if not voice: return [TextContent(type="text", text=f"Current voice: {_current_voice}")]
        _current_voice = voice
        return [TextContent(type="text", text=f"Voice set to: {_current_voice}")]

    elif action == "stop":
        _stop_speaking()
        if _hotword_running: _stop_hotword()
        return [TextContent(type="text", text="Speech stopped. Hotword listener stopped.")]

    elif action == "download_models":
        try: return [TextContent(type="text", text=f"Model download results:\n{_download_models()}")]
        except Exception as e: return [TextContent(type="text", text=f"Download error: {e}")]

    return [TextContent(type="text", text=f"Unknown action: {action}")]

HANDLERS = {"watty_voice": handle_voice}
