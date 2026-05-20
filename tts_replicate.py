#!/usr/bin/env python3
"""
Text-to-Speech using Replicate's Inworld TTS 1.5 Max model.
Converts text to natural speech audio via the Replicate API.
"""
import os
import replicate

# Set API token from environment variable
# Export with: export REPLICATE_API_TOKEN=r8_xxxxx
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")

if not REPLICATE_API_TOKEN:
    raise RuntimeError(
        "REPLICATE_API_TOKEN environment variable is not set. "
        "Set it with: export REPLICATE_API_TOKEN=<your-token>"
    )

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN


def text_to_speech(text: str, voice_id: str = "Ashley"):
    """
    Generate speech audio from text using Replicate's Inworld TTS model.

    Args:
        text: The text to convert to speech.
        voice_id: Voice to use (default: "Ashley").

    Returns:
        The output from the Replicate API (audio URL or file reference).
    """
    output = replicate.run(
        "inworld/tts-1.5-max",
        input={
            "text": text,
            "voice_id": voice_id,
        },
    )
    return output


if __name__ == "__main__":
    result = text_to_speech(
        text="Welcome to the future of voice AI. "
             "Inworld's text-to-speech technology brings natural, "
             "expressive speech to any application.",
        voice_id="Ashley",
    )
    print(result)
