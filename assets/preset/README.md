# Preset voice bundle

`bundle.safetensors` contains the 9 named speakers from
[`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
packaged as extractable 2048-d bfloat16 rows for use with the `-Base` checkpoint.

Each row is lifted from `talker.model.codec_embedding.weight` at the speaker-ID
index from the CustomVoice `config.json`. Loaded via
[`voice_registry.VoiceRegistry._load_presets`](../../voice_registry.py) and
fed to `generate_voice_clone` as `VoiceClonePromptItem(ref_spk_embedding=row,
x_vector_only_mode=True)`.

## Contents

| file                  | size   | purpose                           |
|-----------------------|--------|-----------------------------------|
| `bundle.safetensors`  | ~37 KB | 9 rows × 2048 × bf16              |
| `bundle.json`         | ~1 KB  | speaker ID map + metadata         |

## Provenance

- **Source checkpoint:** `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- **Extraction method:** row indexing into `talker.model.codec_embedding.weight`
- **Validation:** A/B audio comparison at
  [`malaiwah/qwen3-tts-customvoice-ab-clips`](https://huggingface.co/datasets/malaiwah/qwen3-tts-customvoice-ab-clips)
- **Discussion:** [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice #45](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/discussions/45)
- **Mirror:** [`malaiwah/qwen3-tts-preset-voices`](https://huggingface.co/datasets/malaiwah/qwen3-tts-preset-voices)

## Rebuilding

See the docstring in `voice_registry.py` and the discussion #45 linked above
for the extraction snippet.  Inputs: the CustomVoice `model.safetensors`
shard and its `config.json` speaker-ID map.
