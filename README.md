# Talisar's ComfyUI Node Pack

A small collection of ComfyUI custom nodes:

| Node | Category | What it does |
| --- | --- | --- |
| [Save Image (Civitai Metadata)](#save-image-civitai-metadata) | `image` | Saves images with A1111-style metadata (prompts, seed, sampler, model and LoRA hashes) that Civitai reads automatically. |
| [KoboldCpp Prompt](#koboldcpp-prompt) | `text` | Writes a prompt with a local LLM running in KoboldCpp, using your saved context/guidelines. |
| [Review & Edit Prompt](#review--edit-prompt) | `text` | Pauses the run so you can read and edit a prompt before it's used. |
| [Load Text by Index](#load-text--image-by-index-one-per-run) | `text` | Loads one `.txt` prompt from a folder per run, so big prompt batches don't pile up in RAM. |
| [Load Image by Index](#load-text--image-by-index-one-per-run) | `image` | Loads one image (plus its `.txt` caption) from a folder per run, for upscale / img2img passes. |
| [Load Image (with Parameters)](#upscale-only-runs-keep-the-original-metadata) | `image` | Core Load Image (drag & drop) plus a `parameters` output with the image's A1111 generation data. |
| [Image Filename (from loader)](#image-filename-from-loader) | `image` | Outputs the source filename of an image from any load-image node, e.g. for the save node's `filename_prefix`. |

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
- rgthree **Power Lora Loader**, **LoRA Loader Pixaroma**, standard LoRA loaders, Efficiency-style LoRA stacks
- rgthree Seed and other seed/primitive nodes
- images passed through preview/bridge nodes before saving
- `KSampler`, `KSamplerAdvanced`, `SamplerCustom`, `SamplerCustomAdvanced` (Flux guidance included)
- several samplers and save nodes in one workflow (each save node reports the branch that made its image)
- list runs (e.g. a list of prompts from a list loader): each image gets the prompt and seed of its own list item

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
| `source_parameters` | Optional: generation data read from the source image (see [Upscale-only runs](#upscale-only-runs-keep-the-original-metadata)). When not empty it is reused instead of tracing the workflow. |

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
| anything else | Used literally (an unknown `%` code no longer crashes the save) |

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
| `request` | What you want this time, e.g. `rooftop cafe at dusk, laughing`. Can contain [wildcards](#wildcards). |
| `required_phrase` | Optional exact text the prompt must contain. If the model leaves it out, it's inserted word for word at `phrase_position` (`start` / `end`). |
| `max_tokens`, `temperature` | Generation settings. Reasoning models need a higher `max_tokens`. |
| `seed` | With the same seed and inputs, ComfyUI reuses the last prompt instead of asking again. Set it to `fixed` to keep a prompt while you change image settings. |
| `clean_output` | Keep only the prompt: unwraps a ```` ``` ```` code block, drops "Here's your prompt:" lines, `**Prompt:**`-style labels, markdown bold and `<think>` reasoning. |
| `keep_conversation`, `history_turns` | Also send your last few requests and replies, so follow-ups like `same scene at dawn` work. Cleared when the context changes. Saved in `koboldcpp_conversation.json` in the node folder, so it survives a restart. |

Under the inputs the node shows:

- **Status line:** the model KoboldCpp has loaded and its context size (green), or an orange warning when KoboldCpp isn't running or has no model loaded. It refreshes when the workflow opens, on every run, when you change `server_url`, and with the ↻ button.
- **🎲 Random request:** sets the request to `__random__`. Each run then writes a new prompt based only on your guidelines, with the camera angle, expression, lighting, setting, clothing and hair picked for you (see [wildcards](#wildcards)). The full instruction it sends is in `wildcards/random.txt`.
- **Context files:** pick a saved context and click **Load** to copy it into the `context` box, or **Save…** to write the box to a `.txt` file. Files live in `ComfyUI/user/prompt_contexts/` and can be edited in any text editor. Loading is a one-time copy: later edits in the box stay in the workflow until you save them.

Outputs: `prompt` (cleaned text), `model` (the LLM that wrote it), `raw_reply` (the unedited reply) and `picks` (what each wildcard became, on one line, e.g. `angle=... | expression=angry | lighting=neon | ...`; empty when the request has no wildcards).

#### Wildcards

LLMs have favourites: ask Gemma for a "random" portrait every run and you keep getting the same lamp light, blouse and library. Wildcards let the node choose the details instead, and the model only writes the prose.

- `{red|black|navy}` picks one option. `2::red` makes an option twice as likely.
- `__setting__` picks a line from `setting.txt`. In the file, put one option per line. `neon|2` doubles an option's weight, `neon|0.5` halves it and `|0` turns it off. Lines starting with `#` are ignored. Wildcard files can contain wildcards themselves.
- Starter lists come with the node in its `wildcards` folder: `__angle__`, `__expression__`, `__lighting__`, `__clothing_color__`, `__clothing_type__`, `__hairstyle__` and `__setting__`. To change one, or add your own, put a `.txt` file with the same name in `ComfyUI/user/prompt_wildcards/`. Files there win over the starter files, and updates to the node won't overwrite them.
- `__random__` (what the 🎲 button puts in) uses all of them. You can also write your own, e.g. `Random portrait. Camera: __angle__. Expression: __expression__. Lighting: __lighting__, setting: __setting__, wearing __clothing_color__ __clothing_type__, hair __hairstyle__.`

When a request has wildcards, the node adds `Use the details above exactly as given; do not substitute other options.` to it.

Picks come from `seed`. A wildcard never gets the same value two runs in a row, and the whole set of picks never repeats one of the last 8 runs. Those runs are kept in `koboldcpp_recent_picks.json` in the node folder, so this works across restarts. Re-running a seed that's still in that list gives back the same picks.

A request without wildcards is sent exactly as typed.

### Review & Edit Prompt

Pauses the run and opens an editor with the incoming text. Edit it, then click **Continue** (or Ctrl+Enter) to send your version on, or **Cancel run** to stop. If you reload the page while it's waiting, the editor opens again. The approved text is shown on the node afterwards.

| `pause` | Behavior |
| --- | --- |
| `when text changes` | Pauses for each new prompt. Rerunning with the same prompt (e.g. new image seed) reuses your edit without asking. |
| `every run` | Always pauses. |
| `off` | Passes the text straight through. |

The Civitai saver records the final, edited prompt that reaches the text encoder.

## Load Text / Image by Index (one per run)

List loaders (e.g. *Load Texts from Folder Pixaroma*) send every file at once, and ComfyUI then runs each node over the whole list before moving on. Every intermediate image stays in RAM and nothing saves until the end. These nodes load **one file per queued run** instead, so each run goes through the whole workflow (generate, save, upscale, save) and frees its memory before the next one starts.

### Usage

1. Add **Load Text by Index (from folder)** or **Load Image by Index (from folder)** and point `folder` at your files.
2. Set `index` to the first file (1). Its *control after generate* must be **increment** (new nodes start on it; nodes added before this change may still say *randomize*).
3. Set ComfyUI's **Run** count to the number of files (the `total` output tells you) and queue.

To resume after a restart, set `index` to the next file number and queue the remaining runs.

#### Inputs

| Input | Description |
| --- | --- |
| `folder` | Empty: ComfyUI's output folder. A name: a folder inside output. Or a full path. Only files directly in the folder are used; subfolders (`_archive`, `_used`, ...) are skipped. |
| `index` | Which file to load, starting at 1. |
| `sort` | `natural name` (`2.txt` before `10.txt`), `name`, or `modified date` (oldest first). |
| `at_end` | `stop`: an index past the last file errors with `index 151 > 150 files, batch finished`, so extra queued runs fail fast. `wrap`: starts over at the first file. |
| `extensions` | Image node only. Default `png,jpg,jpeg,webp`. |

#### Outputs

| Node | Outputs |
| --- | --- |
| Load Text by Index | `text` (UTF-8, BOM removed), `filename` (without `.txt`), `index` (the one used), `total` (number of `.txt` files) |
| Load Image by Index | `image`, `mask` (like core Load Image: 1 where transparent, all zeros without alpha), `filename` (without extension), `caption` (the matching `<filename>.txt` if there is one, else empty), `index`, `total`, `parameters` (the image's A1111 generation data, empty if it has none) |

Images get their EXIF orientation applied and are converted to RGB. Each run prints a line such as `[LoadTextByIndex] 37/150: 037.txt` to the console.

## Upscale-only runs: keep the original metadata

A workflow that only loads an image and upscales it has no sampler, so Save Image (Civitai Metadata) has nothing to trace and the original prompt, seed, model and LoRAs would be lost. To carry them over:

1. Load the images with **Load Image by Index (from folder)** (a whole folder, one per run) or **Load Image (with Parameters)** (one image, drag & drop like core Load Image).
2. Wire its `parameters` output into the save node's `source_parameters` input.

The saved image keeps the original text as-is (prompt, seed, sampler, model + LoRA hashes), with these changes:

- `Size` is set to the size actually saved.
- If the image is bigger than the original, `Hires upscale: <factor>` is added (e.g. `2`, rounded to 2 decimals). With a downscale back to the input size, no tag is added.
- `extra_parameters` is added before `Version`.

`lora_tags_in_prompt` and `hash_models` don't apply in this mode (the tags and hashes come from the original). Filename keys such as `seed`, `steps`, `cfg` and `model` are read from the carried-over text. The `parameters` chunk is read from PNGs, and from EXIF UserComment in JPG/WebP (what this node and A1111 write). Images without it give an empty string, and the save node then traces the workflow as usual.

## Image Filename (from loader)

Gets the filename of a loaded image without swapping your loader. An IMAGE wire carries no filename, so the node follows its `image` input back through the workflow to the load node and reads that node's file widget.

1. Wire any image into **Image Filename (from loader)**: straight from the loader, or from later in the chain (after an upscale or resize also works).
2. Wire its `filename` output into the save node's `filename_prefix` (right-click the save node, convert the widget to an input if your ComfyUI needs it).

- `filename` is the name without extension, e.g. `vera_042` for `vera_042.png`.
- `image` passes the image through unchanged, so the node can sit inline or off to the side.
- Works with core Load Image, Load Image (with Parameters), Load Image by Index, and other loaders whose widget holds an image file name or path (`.png`, `.jpg`, `.webp`, ...).
- `fallback` is used (and a `[Image Filename]` warning logged) when nothing upstream names an image file, e.g. a generated image, or a list loader that only has a folder widget.

## Troubleshooting

Messages are logged to the ComfyUI console with the prefix `[Civitai Metadata]` (Save Image) or `[KoboldCpp Prompt]`. If something is missing or wrong for your workflow, [open an issue](https://github.com/Talisar/ComfyUI-Talisar-Pack/issues) with the workflow (or the saved PNG) attached.

## License

[MIT](LICENSE)
