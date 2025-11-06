# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example: send pre-computed embeddings to Qwen3 via the OpenAI-compatible API.

This demonstrates how to pass custom embeddings that replace the
``<|vision_start|>`` placeholder in the prompt. The embeddings can be produced
by any external pipeline – they simply need to be serialized with
``torch.save`` and base64 encoded.

Before running this script, launch the vLLM server with Qwen3:

    vllm serve Qwen/Qwen3-4B-Thinking-2507 \
      --limit-mm-per-prompt '{"vision_embedding": 1}'

Then execute:

    python openai_chat_completion_qwen3_embedding.py
"""

import base64
import io
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

MODEL = "Qwen/Qwen3-4B-Thinking-2507"


EMBED_PATH = Path("/workspace/Code-LLaVA/vllm_inference/dummy_prompt_embeddings.npy")


def encode_tensor(tensor: torch.Tensor) -> str:
    """Serialize ``tensor`` with torch.save and return a base64 string."""

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


if __name__ == "__main__":
    if not EMBED_PATH.is_file():
        raise FileNotFoundError(
            f"Expected embedding file at {EMBED_PATH}. Update EMBED_PATH or "
            "place the .npy file there.")

    numpy_embedding = np.load(EMBED_PATH)
    if numpy_embedding.ndim == 3 and numpy_embedding.shape[0] == 1:
        numpy_embedding = numpy_embedding[0]
    if numpy_embedding.ndim != 2:
        raise ValueError(
            f"Expected embedding array with 2 dimensions after squeeze, got "
            f"shape {numpy_embedding.shape}")

    vision_embedding = torch.tensor(numpy_embedding, dtype=torch.float32).contiguous()

    embedding_b64 = encode_tensor(vision_embedding)

    chat_completion = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=1024,
        temperature=0.6,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<|fim_pad|>"
                        ),
                    },
                    {
                        "type": "embedding",
                        "embedding": {
                            "data": embedding_b64,
                            "encoding": "pt",
                        },
                    },
                ],
            }
        ],
    )

    print("Chat completion output:")
    print(chat_completion.choices[0].message.content)
