# Qwen3 Vision-Embedding Support (vLLM)

This note documents the recent changes that allow `Qwen/Qwen3-*` checkpoints to accept pre-computed embeddings through the OpenAI-compatible API.

---

## 1. Input Parsing & Chat API

### What’s new
- The OpenAI chat parser now accepts payloads of the form:
  ```json
  {
    "type": "embedding",
    "embedding": {
      "data": "<base64 torch.save tensor>"
    }
  }
  ```
- The parser tags these payloads with a new modality name, `vision_embedding`, and hands them to the multimodal processor.

### Constraints
- The per-prompt limit is set to **one** `vision_embedding` payload. Multiple embeddings in a single request (even across multiple messages) will raise:
  ```
  Found more '<|fim_pad|>' placeholders in input prompt than actual multimodal data items.
  ```

### File references
- `vllm/entrypoints/chat_utils.py`
- `tests/entrypoints/test_chat_utils.py` (regression test)

---

## 2. Multimodal Processor Pipeline

### Processor implementation
- `Qwen3VisionMultiModalProcessor` (with shared processing info/dummy builder) was introduced in `vllm/model_executor/models/qwen3.py`.
- If the payload is a base64 string (or dict/list), it is deserialized with `torch.load`, converted to a contiguous **2-D** tensor of shape `(num_tokens, hidden_size)`, and kept on CPU until batching. (For `Qwen/Qwen3-4B-Thinking-2507`, `hidden_size = 2560`.)

### Prompt placeholder handling
- **Input prompt**: you write **one** `<|fim_pad|>` token.
- **Prompt update**: `_get_prompt_updates` expands that single token to *N* repeats, where *N = seq_len* of the embedding.
- **Dummy inputs / profiling**: use the same `<|fim_pad|>` token so initialization and LoRA flows stay aligned.

---

## 3. Model Merging Logic

### `get_input_embeddings`
```python
inputs_embeds = self.model.get_input_embeddings(input_ids)
if multimodal_embeddings:
    placeholder_id = self._get_vision_placeholder_token_id()
    inputs_embeds = merge_multimodal_embeddings(
        input_ids,
        inputs_embeds,
        tuple(multimodal_embeddings),
        placeholder_id,
    )
```
- `merge_multimodal_embeddings` masks the positions of `<|fim_pad|>` within `input_ids` and performs an in-place `masked_scatter_`.
- Any mismatch between the number of placeholder positions and the rows in the flattened embedding raises an explicit `ValueError`.
- A debug warning logs the shape(s) of the tensors being merged:
  ```
  Qwen3 vision embeddings merged: num_items=1, shapes=[(119, 2560)]
  ```

### Placeholder ID
- If the HF config exposes `vision_start_token_id`, it is used.
- Otherwise a fallback is captured at model init (`<|fim_pad|>` via `cached_tokenizer_from_config`) and reused.

---

## 4. Usage Example

1. **Start the server**

   ```bash
   vllm serve Qwen/Qwen3-4B-Thinking-2507 \
       --limit-mm-per-prompt '{"vision_embedding": 1}'
   ```

2. **Prepare the embedding**

   - Store your tensor as `(num_tokens, hidden_size)` with `torch.save`. Anything else (e.g., extra batch dims or mismatched hidden size) will raise during validation.
   - Example: `/workspace/Code-LLaVA/vllm_inference/combined_embeds.npy` can be loaded, squeezed, and saved back out as:

     ```python
     import numpy as np, torch, io, base64

     arr = np.load("/workspace/Code-LLaVA/vllm_inference/combined_embeds.npy")
     if arr.ndim == 3 and arr.shape[0] == 1:
         arr = arr[0]
     tensor = torch.tensor(arr, dtype=torch.float32).contiguous()
     buffer = io.BytesIO()
     torch.save(tensor, buffer)
     embedding_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
     ```

3. **Send a request (OpenAI client style)**

   ```python
   from openai import OpenAI

   client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

   response = client.chat.completions.create(
       model="Qwen/Qwen3-4B-Thinking-2507",
       messages=[{
           "role": "user",
           "content": [
               {
                   "type": "text",
                   "text": "Here is a <|fim_pad|> placeholder. Replace it with the embedding I am providing."
               },
               {
                   "type": "embedding",
                   "embedding": {
                       "data": embedding_b64,
                       "encoding": "pt"
                   }
               },
               {
                   "type": "text",
                   "text": "Describe what the embedding could represent."
               },
           ],
       }],
       max_completion_tokens=128,
   )
   print(response.choices[0].message.content)
   ```

4. **What to expect**

   - Server logs emit `Qwen3 vision embeddings merged: num_items=1, shapes=[(num_tokens, hidden_size)]`.
   - Output text comes from the merged hidden states (there is no vision tower).

5. **Important constraints**

   - Exactly **one** `<|fim_pad|>` in the prompt per request; it expands automatically to the embedding length.
   - Only **one** embedding payload per prompt.

For a runnable reference, see `examples/online_serving/openai_chat_completion_qwen3_embedding.py`, which performs all of the above end-to-end.

---

## 5. Testing

- Parser coverage ensures `<|fim_pad|>` expansion matches the payload length.
- Processor coverage verifies batching, splitting, and placeholder ranges.
- Existing Qwen2-VL style tests provide a reference for the expected flow; the Qwen3 tests mirror that behavior for embeddings.

Affected files:
- `tests/entrypoints/test_chat_utils.py`
- `tests/models/multimodal/processing/test_qwen3_vision_embedding.py`

---

## 6. Operational Notes

- Place only **one** `<|fim_pad|>` in the prompt; the processor expands it automatically.
- Embedding shape must be `(num_tokens, hidden_size)` with `num_tokens > 0`. A leading batch dimension is squeezed if it equals 1.
- The merge happens before the decoder runs, so the decoder consumes the combined tensor exactly as you supplied it.

With these changes you can now drive Qwen3 checkpoints with arbitrary external embeddings while retaining the familiar placeholder mechanics from Qwen2-VL.
