# Video Analysis using AI

A simple Streamlit app that accepts an `.mp4` video, extracts one frame every few seconds, sends those frames to an AI vision model, and generates a short summary.

## Features

- Upload a single `.mp4` video in the app.
- Extract frames at a configurable interval.
- Choose one of three use cases:
  - Scene Description
  - Sports Commentator
  - Sports Coach
- View sampled frames, frame-by-frame notes, and a final video summary.

## Setup

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

You can enter your API key directly in the app's sidebar. The app will attempt to **automatically detect** the LLM provider (OpenAI or Groq) based on your API key format, or you can manually override the provider in the sidebar. The default model is `gpt-4.1-mini`. You can override it in the sidebar or set:

```powershell
$env:OPENAI_MODEL="your-model-name"
```

Frame extraction uses FFmpeg through `imageio-ffmpeg`.
