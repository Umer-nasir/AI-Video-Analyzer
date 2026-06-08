import base64
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
import streamlit as st

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


APP_TITLE = "Video Analysis using AI"
DEFAULT_MODEL = "gpt-4.1-mini"


@dataclass
class FrameSample:
    index: int
    timestamp: float
    image_bytes: bytes


USE_CASE_PROMPTS = {
    "Scene Description": {
        "frame": "Describe what is happening in this video frame in one concise sentence.",
        "summary": (
            "Write a short, meaningful summary of the video based on these timestamped "
            "frame descriptions. Keep it clear and useful for a general viewer."
        ),
    },
    "Sports Commentator": {
        "frame": (
            "Act as a sports commentator. Describe the visible action in this frame "
            "with energetic but accurate commentary."
        ),
        "summary": (
            "Turn these timestamped observations into a short sports-style commentary. "
            "Mention changes in action over time and avoid inventing details that are not visible."
        ),
    },
    "Sports Coach": {
        "frame": (
            "Act as a sports coach. Describe the athlete's visible posture, movement, "
            "positioning, or technique in this frame. Include one practical coaching note if possible."
        ),
        "summary": (
            "Write concise coaching feedback based on these timestamped observations. "
            "Separate strengths from improvement suggestions and avoid overclaiming."
        ),
    },
}


def save_upload_to_temp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def get_ffmpeg_executable() -> str:
    if imageio_ffmpeg is None:
        raise RuntimeError("The imageio-ffmpeg package is not installed.")

    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError:
        package_dir = Path(imageio_ffmpeg.__file__).parent
        candidates = sorted((package_dir / "binaries").glob("ffmpeg*.exe"))
        if candidates:
            return str(candidates[0])
        raise


def extract_frames(video_path: str, interval_seconds: int, max_frames: int) -> list[FrameSample]:
    with tempfile.TemporaryDirectory() as frame_dir:
        output_pattern = str(Path(frame_dir) / "frame_%03d.jpg")
        command = [
            get_ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vf",
            f"fps=1/{interval_seconds}",
            "-frames:v",
            str(max_frames),
            "-q:v",
            "3",
            output_pattern,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except OSError as exc:
            raise RuntimeError(
                "FFmpeg was found, but Windows blocked it from running. "
                "Allow the FFmpeg executable or run this project in an environment where FFmpeg is permitted."
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise RuntimeError(f"FFmpeg could not extract frames: {detail}") from exc

        samples: list[FrameSample] = []
        for index, frame_path in enumerate(sorted(Path(frame_dir).glob("frame_*.jpg")), start=1):
            samples.append(
                FrameSample(
                    index=index,
                    timestamp=(index - 1) * interval_seconds,
                    image_bytes=frame_path.read_bytes(),
                )
            )
        return samples


def image_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def call_openai_chat(model: str, messages: list[dict], max_tokens: int) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set the `OPENAI_API_KEY` environment variable before running analysis.")

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        },
        timeout=90,
    )
    if not response.ok:
        detail = response.text[:500]
        raise RuntimeError(f"OpenAI request failed with HTTP {response.status_code}: {detail}")

    payload = response.json()
    return payload["choices"][0]["message"]["content"].strip()


def analyze_frame(model: str, sample: FrameSample, prompt: str) -> str:
    return call_openai_chat(
        model,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(sample.image_bytes)}},
                ],
            }
        ],
        120,
    )


def summarize_video(model: str, descriptions: Iterable[str], prompt: str) -> str:
    observation_block = "\n".join(descriptions)
    return call_openai_chat(
        model,
        [
            {
                "role": "system",
                "content": "You summarize video observations faithfully and concisely.",
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nObservations:\n{observation_block}",
            },
        ],
        300,
    )


def render_setup_help(error: Exception) -> None:
    st.warning(str(error))
    st.info(
        "Install dependencies with `python -m pip install -r requirements.txt`, set "
        "`OPENAI_API_KEY`, and make sure FFmpeg is allowed to run on this computer."
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="video_camera", layout="wide")
    st.title(APP_TITLE)
    st.caption("Upload an .mp4 video, sample frames, and generate an AI summary.")

    with st.sidebar:
        st.header("Analysis")
        use_case = st.selectbox("Use case", list(USE_CASE_PROMPTS.keys()))
        interval_seconds = st.slider("Frame interval", 1, 10, 3, help="Extract one frame every N seconds.")
        max_frames = st.slider("Maximum frames", 1, 20, 8, help="Lower values are faster and cheaper.")
        model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))

    uploaded_file = st.file_uploader("Upload an MP4 video", type=["mp4"])
    if uploaded_file is None:
        st.stop()

    st.video(uploaded_file)

    if not st.button("Extract frames and analyze", type="primary"):
        st.stop()

    temp_path = save_upload_to_temp(uploaded_file)
    try:
        with st.spinner("Extracting frames..."):
            frames = extract_frames(temp_path, interval_seconds, max_frames)

        if not frames:
            st.error("No frames could be extracted from this video.")
            st.stop()

        st.subheader("Sampled Frames")
        columns = st.columns(min(4, len(frames)))
        for idx, sample in enumerate(frames):
            with columns[idx % len(columns)]:
                st.image(sample.image_bytes, caption=f"{sample.timestamp:.1f}s", use_container_width=True)

        prompts = USE_CASE_PROMPTS[use_case]
        observations: list[str] = []
        progress = st.progress(0, text="Analyzing frames...")
        for idx, sample in enumerate(frames, start=1):
            description = analyze_frame(model, sample, prompts["frame"])
            observations.append(f"{sample.timestamp:.1f}s: {description}")
            progress.progress(idx / len(frames), text=f"Analyzed {idx} of {len(frames)} frames")

        with st.spinner("Writing summary..."):
            summary = summarize_video(model, observations, prompts["summary"])

        st.subheader("Video Summary")
        st.write(summary)

        with st.expander("Frame-by-frame notes", expanded=True):
            for observation in observations:
                st.markdown(f"- {observation}")
    except Exception as exc:
        render_setup_help(exc)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
