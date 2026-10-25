# Talisar's ComfyUI Node Pack

A small collection of ComfyUI custom nodes:

| Node | Category | What it does |
| --- | --- | --- |
| [Save Image (Civitai Metadata)](#save-image-civitai-metadata) | `image` | Saves images with A1111-style metadata (prompts, seed, sampler, model and LoRA hashes) that Civitai reads automatically. |
| [KoboldCpp Prompt](#koboldcpp-prompt) | `text` | Writes a prompt with a local LLM running in KoboldCpp, using your saved context/guidelines. |
| [Review & Edit Prompt](#review--edit-prompt) | `text` | Pauses the run so you can read and edit a prompt before it's used. |

More nodes will be added over time. No extra Python packages are needed.

## Installation

**ComfyUI-Manager:** *Manager → Install via Git URL* and paste:

```
https://github.com/Talisar/ComfyUI-Talisar-Pack
```

**Manual:** from your `ComfyUI/custom_nodes` folder:

```bash
git clone https://github.com/Talisar/ComfyUI-Talisar-Pack
```

Restart ComfyUI.

To update, run `git pull` inside the `ComfyUI-Talisar-Pack` folder (or use ComfyUI-Manager), then restart ComfyUI.

**Upgrading from ComfyUI-Civitai-Metadata-Saver:** this pack is the same repository under a new name, and node IDs are unchanged, so existing workflows keep working. In your existing checkout, optionally point git at the new address with `git remote set-url origin https://github.com/Talisar/ComfyUI-Talisar-Pack.git` (the old URL still redirects).

## Save Image (Civitai Metadata)

A drop-in replacement for ComfyUI's Save Image node that writes **A1111-style generation metadata** that Civitai reads automatically when you upload: prompt, negative prompt, seed, steps, sampler, scheduler, CFG, size, checkpoint and LoRAs. It includes **model/LoRA hashes**, so Civitai links your resources on its own.

Most older "A1111 format" savers only work when a KSampler is wired straight to a CLIPTextEncode. This node traces the actual graph from the image back to the sampler that made it. It also records the values that were really used at runtime, so it keeps working with complex workflows:

- prompts built from switches, string concatenation, primitives and preview nodes
- prompts written at runtime by LLM prompt-expander nodes (TextGenerate, Ollama, Gemini, ...)
- rgthree **Power Lora Loader**, standard LoRA loaders, Efficiency-style LoRA stacks
- rgthree Seed and other seed/primitive nodes
- images passed through preview/bridge nodes before saving
- `KSampler`, `KSamplerAdvanced`, `SamplerCustom`, `SamplerCustomAdvanced` (Flux guidance included)
- several samplers and save nodes in one workflow (each save node reports the branch that made its image)

The ComfyUI workflow is still embedded in PNGs, so saved images drag back into ComfyUI as usual.

### Usage

Add **Save Image (Civitai Metadata)** (category `image`) and connect your images to it, the same way as the regular Save Image node. That's all it needs: everything else is read from the workflow.

The `parameters` output contains the exact text written to the image. Connect it to a Show Text / Preview Any node to check it before uploading.

#### Inputs

| Input | Description |
| --- | --- |
| `images` | Images to save. |
| `filename_prefix` | Start of the filename. Accepts strftime dates (`%F`) and subfolders (`a/b`). |
| `filename_keys` | Comma-separated keys appended to the filename (see [Naming keys](#naming-keys)). |
| `foldername_prefix` | Folder inside the ComfyUI output directory to save into. Same syntax. |
| `foldername_keys` | Comma-separated keys appended to the folder name. |
| `delimiter` | Text placed between the prefix, keys and counter. The whole string is used (e.g. `_`, ` `, ` - `). |
| `counter_digits` | Zero-padded counter width. Continues from the highest number already in the folder; does not fill gaps. |
| `counter_position` | Counter at the `last` or `first` position of the filename. |
| `one_counter_per_folder` | On: one counter for the whole folder. Off: a separate counter for each distinct filename. |
| `file_format` | `png`, `jpg` or `webp`. JPG/WebP store the Civitai metadata in EXIF UserComment (no ComfyUI workflow). |
| `quality` | JPG/WebP quality. |
| `embed_workflow` | Also embed the ComfyUI workflow in PNGs. |
| `lora_tags_in_prompt` | Append `<lora:name:strength>` tags to the prompt, as A1111 does. |
| `hash_models` | Compute model/LoRA hashes so Civitai can link resources. |
| `positive_override` / `negative_override` | Optional: use this text instead of the traced prompt. |
| `extra_parameters` | Optional extra A1111 fields appended verbatim, e.g. `Shift: 4`. |

#### Naming keys

The filename and folder name are each built from the prefix followed by keys, joined with the delimiter. This uses the same conventions as [Save Image Extended](https://github.com/audioscavenger/save-image-extended-comfyui), so existing settings carry over.

| Key | Result |
| --- | --- |
| `%F`, `%Y-%m-%d`, `%H-%M-%S` ... | Date/time (strftime) |
| `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, `denoise` | Values traced for this image |
| `model` | Checkpoint / diffusion model name |
| `lora` / `loras` | First LoRA / all LoRAs joined with `+` |
| `123.widget_name` | Widget value from node #123 (the node's badge number) |
| `widget_name` | Widget value from the last node that has it |
| `./key` | A subfolder named by the key's value, e.g. `./lora`, `./%F` |
| `sub/` or `a/b` | Literal subfolder |
| anything else | Used literally |

**Examples**

| Settings | Saved as |
| --- | --- |
| folder prefix `MyChar`, folder keys `%F`, file prefix `MyChar`, delimiter ` ` | `output/MyChar 2026-10-23/MyChar 0001.png` |
| folder prefix `Civitai`, folder keys `./lora, ./%F`, file prefix `img`, file keys `seed`, delimiter `_` | `output/Civitai/MyLora/2026-10-23/img_123456789_0001.png` |
| file keys `model, seed, cfg`, counter position `first`, delimiter `-` | `output/0001-myModel-123456789-7.0.png` |

Folder names can't leave the ComfyUI output directory (`..` is ignored).

### What gets written

A PNG `parameters` text chunk (or EXIF UserComment for JPG/WebP) in the format Automatic1111 uses:

```
a photo of a cat on a swing
<lora:MyLora:0.8>
Negative prompt: blurry
Steps: 25, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 6, Seed: 123456789, Size: 1024x1024, Model hash: 14ba3e3b67, Model: myModel, Lora hashes: "MyLora: b71fef1e25", Hashes: {"model": "14ba3e3b67", "lora:MyLora": "b71fef1e25"}, Version: ComfyUI
```

Hashes are Civitai's AutoV2 format (the first 10 characters of the file's SHA-256). Hashing a large model the first time takes a while (about 30 s for a 13 GB file). Results are cached in `hash_cache.json` in the node folder and reused until the file changes.

### Civitai notes

- Civitai can only link a model or LoRA that is published on Civitai. Your own unpublished LoRAs show as *"could not be matched"*, but their names and tags are still in the metadata.
- If Civitai reads nothing from a PNG, try saving with `embed_workflow` off so only the A1111 metadata is present.

## KoboldCpp Prompt + Review & Edit Prompt

Two nodes (category `text`) for writing prompts with a local LLM running in [KoboldCpp](https://github.com/LostRuins/koboldcpp), then checking them before they're used:

```
KoboldCpp Prompt ──prompt──▶ Review & Edit Prompt ──text──▶ CLIP Text Encode ──▶ ... ──▶ Save Image (Civitai Metadata)
```

Start KoboldCpp with any text model first (Kobold Lite's default address is `http://localhost:5001`). No extra Python packages are needed.

### KoboldCpp Prompt

| Input | Description |
| --- | --- |
| `server_url` | KoboldCpp's address. |
| `context` | Your standing guidelines, sent as the system message on **every** request. KoboldCpp's API doesn't see Kobold Lite's Memory or chat, so put them here. Saved with the workflow. |
| `request` | What you want this time, e.g. `rooftop cafe at dusk, laughing`. |
| `required_phrase` | Optional exact text the prompt must contain. If the model leaves it out, it's inserted word for word at `phrase_position` (`start` / `end`). |
| `max_tokens`, `temperature` | Generation settings. Reasoning models need a higher `max_tokens`. |
| `seed` | With the same seed and inputs, ComfyUI reuses the last prompt instead of asking again. Set it to `fixed` to keep a prompt while you change image settings. |
| `clean_output` | Keep only the prompt: unwraps a ```` ``` ```` code block, drops "Here's your prompt:" lines, `**Prompt:**`-style labels, markdown bold and `<think>` reasoning. |
| `keep_conversation`, `history_turns` | Also send your last few requests and replies, so follow-ups like `same scene at dawn` work. Cleared when the context changes. Saved in `koboldcpp_conversation.json` in the node folder, so it survives a restart. |
| `randomize_slots` | On (default): picks a camera angle, expression, lighting, clothing colour and type, hairstyle and setting from the lists below, using `seed`, and adds `Use exactly these choices: ... Do not substitute other options.` to the request. Off: the request is sent as is. |
| `angle`, `expression`, `lighting`, `clothing_color`, `clothing_type`, `hairstyle`, `setting` | The options for each slot, one per line. Add `\|weight` to make one more or less likely (`neon\|2`, `cream\|0.5`, `\|0` turns it off). Leave a list empty to let the model choose that part. |

Under the inputs the node shows:

- **Status line:** the model KoboldCpp has loaded and its context size (green), or an orange warning when KoboldCpp isn't running or has no model loaded. It refreshes when the workflow opens, on every run, when you change `server_url`, and with the ↻ button.
- **Context files:** pick a saved context and click **Load** to copy it into the `context` box, or **Save…** to write the box to a `.txt` file. Files live in `ComfyUI/user/prompt_contexts/` and can be edited in any text editor. Loading is a one-time copy: later edits in the box stay in the workflow until you save them.

Outputs: `prompt` (cleaned text), `model` (the LLM that wrote it), `raw_reply` (the unedited reply) and `picks` (this run's slot choices on one line, e.g. `angle=... | expression=angry | lighting=neon | ...`; empty when `randomize_slots` is off).

**How picks avoid repeats:** the last 8 runs' picks are kept in `koboldcpp_recent_picks.json` in the node folder, so this works across restarts. Each slot is re-rolled (with the same seeded generator) if it matches the previous run, and the angle + expression + lighting + setting combination is never one of the last 8. Re-running a seed that's still in that list gives back exactly the same picks.

### Review & Edit Prompt

Pauses the run and opens an editor with the incoming text. Edit it, then click **Continue** (or Ctrl+Enter) to send your version on, or **Cancel run** to stop. If you reload the page while it's waiting, the editor opens again. The approved text is shown on the node afterwards.

| `pause` | Behavior |
| --- | --- |
| `when text changes` | Pauses for each new prompt. Rerunning with the same prompt (e.g. new image seed) reuses your edit without asking. |
| `every run` | Always pauses. |
| `off` | Passes the text straight through. |

The Civitai saver records the final, edited prompt that reaches the text encoder.

## Troubleshooting

Messages are logged to the ComfyUI console with the prefix `[Civitai Metadata]` (Save Image) or `[KoboldCpp Prompt]`. If something is missing or wrong for your workflow, [open an issue](https://github.com/Talisar/ComfyUI-Talisar-Pack/issues) with the workflow (or the saved PNG) attached.

## License

[MIT](LICENSE)
