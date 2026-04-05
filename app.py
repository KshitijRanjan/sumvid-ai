import os
import json
import tempfile
import subprocess
import threading
import time
from pathlib import Path

import streamlit as st
import whisper
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SumVid.ai – Long-to-Short",
    page_icon="🎬",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Force white background on main content area only */
    .stApp {
        background-color: #ffffff;
    }
    section.main > div {
        background-color: #ffffff;
    }

    /* Default text colour for main area */
    .stApp, .stApp p, .stApp label, .stApp div {
        color: #1e293b;
    }

    /* Reduce Streamlit's default top padding to bring title closer to top */
    .block-container {
        padding-top: 3.5rem !important;
    }

.main-title {
        font-family: 'Inter', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: #0f172a;
        margin: 0 0 0.2rem 0;
    }

    .sub-description {
        font-size: 0.88rem;
        font-style: italic;
        color: #64748b;
        margin: 0;
    }

    .upload-label {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: #6366f1;
        margin-bottom: 0.3rem;
    }

    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border-radius: 8px;
    }

    /* Pastel blue Generate button */
    button[data-testid="baseButton-primary"] {
        background-color: #5b9cf5 !important;
        color: #ffffff !important;
        border: none !important;
        font-size: 1.05rem !important;
        border-radius: 10px !important;
    }
    button[data-testid="baseButton-primary"]:hover {
        background-color: #3b82f6 !important;
    }

    /* Compact file uploader — hide drag-drop text, keep button */
    [data-testid="stFileUploaderDropzone"] {
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 10px 16px !important;
        background: #fafafa !important;
    }
    [data-testid="stFileUploaderDropzone"] span:not(:last-child),
    [data-testid="stFileUploaderDropzone"] small {
        display: none !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: #6366f1 !important;
        color: white !important;
        border-radius: 20px !important;
        border: none !important;
        padding: 5px 18px !important;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
    }

    .stProgress > div > div {
        background-color: #6366f1;
    }

    /* Sidebar — smaller, tighter font */
    [data-testid="stSidebar"] {
        font-size: 0.82rem;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 0.95rem;
        font-weight: 600;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] p {
        font-size: 0.78rem;
    }

    /* Chat section */
    .chat-label {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }

    .chat-sublabel {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    }

    /* Remove border/shadow from chat input bar */
    [data-testid="stChatInput"] {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    /* User message bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #eef2ff;
        border-radius: 12px;
    }

    /* Assistant message bubble */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #f8fafc;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar – configuration
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    target_duration = st.slider(
        "Target summary duration (seconds)",
        min_value=60,
        max_value=600,
        value=240,
        step=30,
        help="How long should the final highlight reel be?",
    )

    whisper_model_size = st.selectbox(
        "Whisper model",
        options=["tiny", "base"],
        index=1,
        help="'base' is more accurate; 'tiny' is faster.",
    )

    whisper_language = st.selectbox(
        "Video language",
        options=["Auto-detect", "English", "Hindi"],
        index=0,
        help="Set the spoken language to improve transcription accuracy.",
    )
    _lang_map = {"Auto-detect": None, "English": "en", "Hindi": "hi"}
    whisper_language_code = _lang_map[whisper_language]

    _env_key = os.getenv("ANTHROPIC_API_KEY", "")
    if _env_key:
        anthropic_api_key = _env_key
        st.caption("Anthropic API Key: configured via environment.")
    else:
        anthropic_api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Required to call Claude for segment selection.",
        )

    st.markdown("---")
    st.caption("SumVid.ai · Long-to-Short Video Summariser")

# ─────────────────────────────────────────────
# Sticky header
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🎬 SumVid.ai — Long-to-Short Video Summariser</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-description">Upload a long MP4 and receive a concise highlight reel that preserves the narrative arc: Beginning → Middle → End.</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────

def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds using ffprobe, or 0 if unavailable."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def compress_video(input_path: str, output_path: str, progress_cb) -> None:
    """
    Transcode video to 480p, 800kbps video + 64kbps audio.
    Streams FFmpeg progress so the UI bar updates in real time.
    """
    total_duration = get_video_duration(input_path)
    progress_cb(0.02, "Starting compression…")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", "scale=-2:480",
        "-c:v", "libx264",
        "-b:v", "800k",
        "-c:a", "aac",
        "-b:a", "64k",
        "-movflags", "+faststart",
        "-progress", "pipe:1",  # stream structured progress to stdout
        "-nostats",
        output_path,
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in process.stdout:
        line = line.strip()
        if line.startswith("out_time_ms="):
            try:
                elapsed_s = int(line.split("=")[1]) / 1_000_000
                if total_duration > 0:
                    frac = min(0.05 + 0.93 * (elapsed_s / total_duration), 0.98)
                    progress_cb(frac, f"Compressing… {elapsed_s:.0f}s / {total_duration:.0f}s")
            except (ValueError, IndexError):
                pass

    process.wait()

    if process.returncode != 0:
        stderr = process.stderr.read()
        raise RuntimeError(f"Video compression failed:\n{stderr}")

    compressed_mb = os.path.getsize(output_path) / 1_048_576
    progress_cb(1.0, f"Compressed to {compressed_mb:.0f} MB.")


def extract_audio(video_path: str, audio_path: str, progress_cb) -> None:
    """Extract mono 16 kHz WAV from video using FFmpeg."""
    progress_cb(0.1, "Extracting audio track…")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                   # no video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "16000",          # 16 kHz – Whisper optimum
        "-ac", "1",              # mono
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{result.stderr}")
    progress_cb(1.0, "Audio extracted.")


def transcribe_audio(audio_path: str, model_size: str, progress_cb, language: str | None = None) -> list[dict]:
    """
    Run Whisper in a background thread so the main thread can update the
    progress bar every few seconds. Progress is estimated from elapsed time
    vs expected duration (Whisper base ≈ 8–12x slower than realtime on CPU).
    """
    audio_duration = get_video_duration(audio_path)
    # Conservative estimate for shared CPU: base ~10x slower than realtime
    estimated_secs = max(audio_duration * 10, 30)

    progress_cb(0.05, f"Loading Whisper '{model_size}' model…")
    model = whisper.load_model(model_size)
    progress_cb(0.12, f"Transcribing {audio_duration:.0f}s of audio — est. {estimated_secs / 60:.0f} min. Please wait…")

    result_holder: list = [None]
    error_holder: list = [None]
    done = threading.Event()

    def _run():
        try:
            result_holder[0] = model.transcribe(audio_path, verbose=False, language=language)
        except Exception as exc:
            error_holder[0] = exc
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True).start()

    start = time.time()
    while not done.wait(timeout=3):
        elapsed = time.time() - start
        frac = min(0.12 + 0.83 * (elapsed / estimated_secs), 0.95)
        m, s = divmod(int(elapsed), 60)
        progress_cb(frac, f"Transcribing… {m:02d}:{s:02d} elapsed")

    if error_holder[0]:
        raise error_holder[0]

    segments = [
        {"text": seg["text"].strip(), "start": round(seg["start"], 2), "end": round(seg["end"], 2)}
        for seg in result_holder[0]["segments"]
    ]
    progress_cb(1.0, f"Transcription complete — {len(segments)} segments found.")
    return segments


def enforce_duration(segments: list[dict], target_secs: int, tolerance: float = 0.50) -> list[dict]:
    """
    Greedily include segments in chronological order until the budget is used.
    Drops whole segments rather than truncating mid-segment to avoid jarring cuts.
    Allows up to (target_secs * (1 + tolerance)) total duration.
    """
    budget = target_secs * (1 + tolerance)
    result = []
    running = 0.0
    for seg in segments:
        dur = seg["end"] - seg["start"]
        if running + dur <= budget:
            result.append(seg)
            running += dur
    return result


def select_segments_with_claude(
    segments: list[dict],
    target_secs: int,
    api_key: str,
    progress_cb,
    user_instructions: str = "",
) -> list[dict]:
    """
    Ask Claude to pick segments totalling ~target_secs seconds.
    Returns a list of {"start": float, "end": float} dicts in chronological order.
    """
    progress_cb(0.1, "Sending transcript to Claude for analysis…")

    client = anthropic.Anthropic(api_key=api_key)

    transcript_json = json.dumps(segments, indent=2)

    lower = int(target_secs)
    upper = int(target_secs * 1.50)

    system_prompt = (
        "You are a professional video editor with expertise in narrative storytelling. "
        "Your task is to select the most compelling segments from a video transcript to create "
        "a highlight reel that maintains logical narrative flow."
    )

    user_prompt = f"""Below is a timestamped transcript from a video.
Select segments whose TOTAL duration is strictly between {lower}s and {upper}s (target: {target_secs}s).
Track your running total as you select each segment. Stop adding segments once the total is within 20s of {target_secs}s. Do not exceed {upper}s.

CRITICAL RULES:
1. The selected segments MUST maintain a logical narrative arc:
   - BEGINNING: Set context / introduce key topic (~20% of total duration)
   - MIDDLE: Core content, key insights, or pivotal moments (~60% of total duration)
   - END: Conclusion, call-to-action, or satisfying resolution (~20% of total duration)
2. Prefer segments with high information density (key insights, conclusions, strong quotes).
3. Avoid filler, repetition, or tangential discussions.
4. Each selected segment should be between 10 and 90 seconds long for smooth viewing.
5. Keep segments in chronological order.
"""

    if user_instructions.strip():
        user_prompt += f"""
USER INSTRUCTIONS — these take priority over the general rules above:
{user_instructions.strip()}
"""

    user_prompt += f"""
Respond ONLY with a valid JSON array (no markdown, no explanation) in this exact format:
[
  {{"start": 0.0, "end": 45.2}},
  {{"start": 120.5, "end": 178.3}},
  ...
]

TRANSCRIPT:
{transcript_json}
"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{"role": "user", "content": user_prompt}],
        system=system_prompt,
    )

    progress_cb(0.8, "Parsing Claude's segment selections…")

    raw = message.content[0].text.strip()

    # Strip possible markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    selected = json.loads(raw)

    # Validate and sort chronologically
    selected = sorted(
        [s for s in selected if "start" in s and "end" in s and s["end"] > s["start"]],
        key=lambda x: x["start"],
    )

    total = sum(s["end"] - s["start"] for s in selected)
    progress_cb(1.0, f"Selected {len(selected)} segments totalling {total:.0f}s.")
    return selected


def build_highlight_reel(
    video_path: str,
    segments: list[dict],
    output_path: str,
    progress_cb,
) -> None:
    """
    Use FFmpeg filter_complex to cut and concatenate segments without full re-encoding.
    Strategy: copy-stream concat via the concat demuxer (fastest, no quality loss).
    Falls back to re-encode if copy fails (e.g. non-keyframe cuts).
    """
    progress_cb(0.05, "Preparing FFmpeg cut list…")

    with tempfile.TemporaryDirectory() as clip_dir:
        clip_paths = []

        # Step 1 – extract each segment as a separate clip (stream copy, fast)
        for i, seg in enumerate(segments):
            clip_path = os.path.join(clip_dir, f"clip_{i:04d}.mp4")
            duration = seg["end"] - seg["start"]
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(seg["start"]),
                "-i", video_path,
                "-t", str(duration),
                "-c", "copy",        # stream copy – no re-encode
                "-avoid_negative_ts", "make_zero",
                clip_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # Fallback: re-encode this clip
                cmd[-5] = "-c:v"
                cmd.insert(-4, "libx264")
                cmd.insert(-4, "-c:a")
                cmd.insert(-4, "aac")
                result = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", str(seg["start"]),
                        "-i", video_path,
                        "-t", str(duration),
                        "-c:v", "libx264",
                        "-c:a", "aac",
                        "-avoid_negative_ts", "make_zero",
                        clip_path,
                    ],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"FFmpeg failed on segment {i} ({seg['start']}–{seg['end']}):\n{result.stderr}"
                    )

            clip_paths.append(clip_path)
            frac = 0.05 + 0.75 * (i + 1) / len(segments)
            progress_cb(frac, f"Extracted clip {i + 1}/{len(segments)}…")

        # Step 2 – write a concat list file
        concat_list = os.path.join(clip_dir, "concat_list.txt")
        with open(concat_list, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{cp}'\n")

        # Step 3 – concatenate all clips into final output
        progress_cb(0.85, "Concatenating clips into final highlight reel…")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Fallback: re-encode on concat (fixes codec mismatch between clips)
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list,
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    output_path,
                ],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg concat failed:\n{result.stderr}")

    progress_cb(1.0, "Highlight reel ready!")


def format_segments_table(segments: list[dict]) -> list[dict]:
    """Convert raw segment list to a display-friendly format."""
    rows = []
    for i, s in enumerate(segments, 1):
        dur = s["end"] - s["start"]
        rows.append(
            {
                "#": i,
                "Start": f"{int(s['start'] // 60):02d}:{s['start'] % 60:05.2f}",
                "End": f"{int(s['end'] // 60):02d}:{s['end'] % 60:05.2f}",
                "Duration": f"{dur:.1f}s",
            }
        )
    return rows


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "user_instruction" not in st.session_state:
    st.session_state.user_instruction = ""
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "ready_to_generate" not in st.session_state:
    st.session_state.ready_to_generate = False

# ─────────────────────────────────────────────
# Phase 1 — Upload
# ─────────────────────────────────────────────
with st.chat_message("assistant"):
    st.markdown("**Upload your video to get started.** I'll transcribe it and create a highlight reel.")

uploaded_file = st.file_uploader(
    label="📎 Attach video",
    label_visibility="collapsed",
    type=["mp4", "mov", "mkv", "avi"],
    help="Supported: MP4, MOV, MKV, AVI.",
)

# ─────────────────────────────────────────────
# Phase 2 — Instruction (shown after upload)
# ─────────────────────────────────────────────
if uploaded_file is not None:

    if not check_ffmpeg():
        st.error(
            "❌ **FFmpeg not found.**\n\n"
            "- macOS: `brew install ffmpeg`\n"
            "- Ubuntu: `sudo apt install ffmpeg`"
        )
        st.stop()

    if not anthropic_api_key:
        st.warning("⚠️ Please enter your **Anthropic API Key** in the sidebar to proceed.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(f"📎 **{uploaded_file.name}** · {uploaded_file.size / 1_048_576:.1f} MB ✅")

    with st.chat_message("assistant"):
        st.markdown(
            "**What should the highlight reel focus on?** *(optional)*\n\n"
            "Describe what to prioritise — e.g. *'Focus on the Q&A'*, *'Skip the intro'*. "
            "Or press **Enter** to skip and use the default narrative selection."
        )

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.ready_to_generate:
        if st.button("Skip →  Generate with default settings", type="secondary"):
            st.session_state.ready_to_generate = True
            st.rerun()
    elif st.session_state.user_instruction:
        if st.button("✕ Clear instruction", type="secondary"):
            st.session_state.user_instruction = ""
            st.session_state.chat_messages = []
            st.session_state.ready_to_generate = False
            st.rerun()

# ─────────────────────────────────────────────
# Chat input — disabled until file uploaded
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.chat_input("Upload your video above to continue…", disabled=True)
elif not st.session_state.ready_to_generate:
    if prompt := st.chat_input("What should the highlight reel focus on?"):
        if prompt.strip():
            st.session_state.chat_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"Got it! I'll prioritise: **{prompt}**"},
            ]
            st.session_state.user_instruction = prompt
        st.session_state.ready_to_generate = True
        st.rerun()

# ─────────────────────────────────────────────
# Phase 3 — Generate (shown after instruction step)
# ─────────────────────────────────────────────
if uploaded_file is not None and st.session_state.ready_to_generate:

    if st.button("🎬 Generate Highlight Reel", type="primary", use_container_width=True):

        # Persistent temp dir for the session (cleaned up on exit)
        with tempfile.TemporaryDirectory(prefix="sumvid_") as tmp_dir:

            original_path    = os.path.join(tmp_dir, "input.mp4")
            compressed_path  = os.path.join(tmp_dir, "input_compressed.mp4")
            audio_path       = os.path.join(tmp_dir, "audio.wav")
            output_path      = os.path.join(tmp_dir, "summary.mp4")

            # ── Save uploaded file (chunked to avoid RAM spike) ────
            with open(original_path, "wb") as f:
                CHUNK = 8 * 1024 * 1024  # 8 MB chunks
                while True:
                    chunk = uploaded_file.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)

            # ── Compress if file exceeds 200 MB ───────────────────
            # Compressed copy is used only for audio extraction + Whisper.
            # Final video cut always uses the original for full quality.
            SIZE_THRESHOLD_MB = 200
            file_size_mb = os.path.getsize(original_path) / 1_048_576
            processing_path = original_path  # file used for audio/transcription
            if file_size_mb > SIZE_THRESHOLD_MB:
                st.markdown("### Pre-processing — Video Compression")
                bar0 = st.progress(0.0)
                status0 = st.empty()

                def cb0(frac, msg):
                    bar0.progress(min(frac, 1.0))
                    status0.caption(msg)

                try:
                    compress_video(original_path, compressed_path, cb0)
                    processing_path = compressed_path  # use compressed for Whisper only
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()

            # ── Stage 1: Audio Extraction ──────────────────
            st.markdown("### Stage 1 — Audio Extraction")
            bar1 = st.progress(0.0)
            status1 = st.empty()

            def cb1(frac, msg):
                bar1.progress(min(frac, 1.0))
                status1.caption(msg)

            try:
                extract_audio(processing_path, audio_path, cb1)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
            finally:
                # Compressed copy is no longer needed — free disk space before Whisper
                if processing_path == compressed_path and os.path.exists(compressed_path):
                    os.remove(compressed_path)

            # ── Stage 2: Transcription ─────────────────────
            st.markdown("### Stage 2 — Transcription")
            bar2 = st.progress(0.0)
            status2 = st.empty()

            def cb2(frac, msg):
                bar2.progress(min(frac, 1.0))
                status2.caption(msg)

            try:
                segments = transcribe_audio(audio_path, whisper_model_size, cb2, language=whisper_language_code)
            except Exception as e:
                st.error(f"Transcription error: {e}")
                st.stop()
            finally:
                # WAV is no longer needed — free disk space before assembly
                if os.path.exists(audio_path):
                    os.remove(audio_path)

            with st.expander("📄 View raw transcript segments"):
                st.json(segments[:20])
                if len(segments) > 20:
                    st.caption(f"…and {len(segments) - 20} more segments.")

            # ── Stage 3: LLM Segment Selection ────────────
            st.markdown("### Stage 3 — AI Segment Analysis")
            bar3 = st.progress(0.0)
            status3 = st.empty()

            def cb3(frac, msg):
                bar3.progress(min(frac, 1.0))
                status3.caption(msg)

            try:
                selected_segments = select_segments_with_claude(
                    segments, target_duration, anthropic_api_key, cb3,
                    user_instructions=st.session_state.user_instruction,
                )
            except json.JSONDecodeError as e:
                st.error(f"Claude returned invalid JSON: {e}")
                st.stop()
            except anthropic.AuthenticationError:
                st.error("❌ Invalid Anthropic API key. Please check your key in the sidebar.")
                st.stop()
            except Exception as e:
                st.error(f"Claude API error: {e}")
                st.stop()

            # Enforce duration hard cap in code — Claude's selection is a best-effort estimate
            selected_segments = enforce_duration(selected_segments, target_duration)

            total_selected = sum(s["end"] - s["start"] for s in selected_segments)
            st.markdown(
                f"**{len(selected_segments)} segments selected** · "
                f"Total duration: **{total_selected:.0f}s** "
                f"(target: {target_duration}s)"
            )
            st.table(format_segments_table(selected_segments))

            # ── Stage 4: Video Cutting ─────────────────────
            st.markdown("### Stage 4 — Video Assembly")
            bar4 = st.progress(0.0)
            status4 = st.empty()

            def cb4(frac, msg):
                bar4.progress(min(frac, 1.0))
                status4.caption(msg)

            try:
                build_highlight_reel(original_path, selected_segments, output_path, cb4)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

            # ── Download ───────────────────────────────────
            st.markdown("---")
            st.success("🎉 Your highlight reel is ready!")

            size_mb = os.path.getsize(output_path) / 1_048_576
            base_name = Path(uploaded_file.name).stem

            with open(output_path, "rb") as f:
                st.video(f.read())

            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Highlight Reel (summary.mp4)",
                    data=f,
                    file_name=f"{base_name}_summary.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    type="primary",
                )

            st.caption(f"Output size: {size_mb:.1f} MB · Duration: ~{total_selected:.0f}s")
