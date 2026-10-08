# AI smart selection in this fork

Branch: feature/ai-smart-selection. The native ONNX manager and pinned exports come from
[upstream PR #1199](https://github.com/storytold/photocraft/pull/1199), preserving original
commits and authorship. This branch adds explicit BiRefNet background removal, interactive
SAM 2.1 object selection and BiRefNet candidates refined by SAM. Classical methods remain
available. There is no Python runtime, remote image upload, automatic weight download or
new document format.

This integration still needs full-model visual and hardware qualification. Synthetic ONNX
contract tests exercise the real CPU runtime; they do not demonstrate trained-model accuracy,
photographic/anime quality, consumer-hardware speed or RTX 2070 memory use.

## Build, install and activate

    cargo run --release -p photocraft --features local-ml -- photo.png
    # Optional CUDA execution-provider support:
    cargo run --release -p photocraft --features local-ml-cuda -- photo.png

Open Edit > Preferences > Integrations and explicitly download each required model. Downloads
have exact byte sizes and SHA-256 checks, immutable revisions, staged files and atomic commits.
Missing/corrupt weights produce errors; inference never starts a download or silently chooses
the classical algorithm. Weights and per-model license/source files stay in the application
model cache, separate from image files. Offline inference works after installation.

| Action | Required models | Menu |
|---|---|---|
| Remove Background | BiRefNet HR Matting | Layer > AI Remove Background… |
| Object Selection | SAM 2.1 Large | Select > AI Object Selection… |
| Select Subject | Both | Select > AI Select Subject… |

Compute a preview and inspect it before Apply. Apply creates one native undo step; Cancel
cancels the window's job and discards the preview. Previews bind to the original document
snapshot and reject stale results after edits, Undo, switching documents or reopening.

Background removal defaults to a non-destructive layer mask. Original pixel alpha remains,
and existing mask coverage is multiplied into the AI result. Feather, shift and radius use
native mask refinement. New Layer preserves the source and hides it. Decontamination forces
a new pixel copy rather than recolouring the source.

Object Selection accepts clicks (positive), Alt-clicks (negative), dragged boxes and up to
64 accumulated points. Each gesture recomputes the preview. SAM embeddings are reused for
the same normalized source tensor, including the existing screen/canvas zoom/pan transform.
The selected mode can replace, add, subtract or intersect native selection coverage.

Select Subject generates up to eight disconnected foreground candidates from learned
BiRefNet alpha, ranks by alpha confidence, area and distance from the image centre, then
passes a candidate box and interior positive point to SAM. Numbered canvas rectangles and
a dropdown allow a different candidate. All prominent subjects combines candidates scoring
at least one quarter of the highest prominence score. Recompute after changing a choice.
Scores are heuristics, not calibrated object-class confidence; detection is not human-only.

## Models and devices

| Model id | Export revision | Input | Download | License |
|---|---|---|---:|---|
| birefnet-hr-matting | PinkPixel/birefnet-hr-matting-onnx, 792518b9d4dfe9793891712677de25e554f948c6 | 2048 × 2048 RGB FP32 | 932,150,975 bytes | MIT |
| sam2.1-large | onnx-community/sam2.1-hiera-large-ONNX, 3c23431f721e69cae82dbfd0c28fd692cc714021 | 1024 × 1024 RGB FP32 | 911,109,265 bytes | Apache-2.0 |

Hashes and artifact contracts are in crates/ml/src/catalog.rs. ONNX Runtime is 1.28 via
ort 2.0.0-rc.13; redistribution notices are in crates/ml/licenses/. No weights are committed.

CPU is the default. Auto tries CUDA when compiled and falls back to CPU on provider/session
creation failure. Explicit CUDA fails with an actionable error instead of changing devices.
An inference failure after successful loading does not silently retry. Preferences exposes
requested/active provider and a Release model memory action. CUDA builds require suitable
ONNX Runtime GPU libraries, CUDA/cuDNN and drivers; the feature does not install them.

Only BiRefNet or the SAM encoder/decoder pair remains resident. Switching models/providers or
models.release drops sessions and embeddings. The SAM cache retains one image, keyed by
the complete normalized tensor hash, with each feature tensor capped at eight million values.
Only one preview is retained per session. Canvases are limited to 64 MP, decontamination to
16 MP and candidate grids to 256 × 256. The CUDA arena has a 6 GB limit, which does not cap
all CUDA allocations or guarantee that FP32 models fit an 8 GB card.

## Engine, CLI and MCP

| Command | Parameters / result |
|---|---|
| models.list | Availability, installations, sizes, revisions, licenses, device status |
| models.download / models.remove | Model id; no document required |
| models.device | Empty object queries; provider cpu, auto or cuda configures |
| models.release | Release resident sessions/embeddings |
| layer.aiRemoveBackground | layer?, sampleAllLayers=false, quality=high and common options |
| select.aiObject | rect=[x,y,width,height]?, points=[{position:[x,y],positive:true}]? and common options |
| select.aiSubject | sampleAllLayers=true, candidate=0..7? or allSubjects=true; returns candidates |
| ai.previewInfo | ready, stale, document, output, candidates |
| ai.apply / ai.discard | Apply unchanged-source preview / discard without editing |
| jobs.list / jobs.cancel | Existing asynchronous status/result/error and cancellation |

Common options: preview=false, output=selection/layerMask/newLayer,
mode=replace/add/subtract/intersect, edgeFeather=0..128, edgeShift=-100..100,
edgeRadius=0..64, decontaminate=false, optional layer and sampleAllLayers.
Remove Background defaults to layerMask, the selection actions to selection.
Unknown parameters and incompatible options return structured engine errors. Points must be
finite document coordinates inside the canvas; reversed boxes are normalized and clipped.

Trusted CLI example:

    cargo build --release -p photocraft-cli --features local-ml
    export PHOTOCRAFT_MODEL_DIR="/absolute/path/to/model-cache"
    target/release/photocraft-cli run --new '{"width":1,"height":1}' \
      --cmd models.download --params '{"id":"birefnet-hr-matting"}'
    # Repeat explicit installation for sam2.1-large when needed.
    target/release/photocraft-cli run photo.png \
      --cmd layer.aiRemoveBackground --params '{"edgeFeather":1,"output":"layerMask"}' \
      --out cutout.psd
    target/release/photocraft-cli run photo.png \
      --cmd select.aiObject \
      --params '{"rect":[40,20,500,700],"points":[{"position":[200,160]},{"position":[60,40],"positive":false}]}' \
      --out selected.pcraft
    target/release/photocraft-cli run photo.png \
      --cmd select.aiSubject --params '{"allSubjects":true}' --out subjects.pcraft

MCP uses the existing authenticated loopback desktop bridge with a native backend configured.
The default untrusted headless MCP session intentionally has no model/cache/network authority.
Existing tokens and read/write-root restrictions remain enforced; no new remote-control server
was added. See [control-protocol.md](control-protocol.md) for bridge activation and root grants.

Example MCP tool calls (use the actual returned job number):

    {"name":"command_run","arguments":{"id":"models.list"}}
    {"name":"command_run","arguments":{"id":"select.aiSubject","params":{"preview":true},"wait":false}}
    {"name":"jobs_list","arguments":{}}
    {"name":"jobs_cancel","arguments":{"job":1}}
    {"name":"command_run","arguments":{"id":"ai.previewInfo"}}
    {"name":"command_run","arguments":{"id":"ai.apply"}}
    {"name":"doc_save","arguments":{"path":"cutout.psd"}}

Cancellation is optional; await successful job completion before Apply. Save requires write
authority for the relative PSD path. The control equivalent is engine.execute with command,
params and wait=false. No arbitrary model URLs/paths are accepted by engine commands.

## Files changed

| Area | Main files |
|---|---|
| Shared ML | crates/ml/src/{catalog,native,image,prompts,candidates}.rs, licenses and manifests |
| Native engine | model_cmds.rs, ai_smart_cmds.rs and tests, commands.rs, Session and jobs |
| UI | ai_ui.rs, menus, menu_catalog, canvas, retouch_ui, state, preferences and i18n |
| Host activation | Desktop/CLI main.rs and feature declarations |
| MCP verification | crates/automation/tests/mcp.rs |
| Documentation | This guide, local-models guides, provenance and generated parity |

Pre-existing style warnings in text/cjk.rs and plugins/manifest.rs were simplified for
Rust 1.95 Clippy without changing language selection or plugin validation behavior.

## Remaining validation and limitations

- High is the only supported BiRefNet preset. Its fixed 2048-pixel export cannot implement
  arbitrary inference resolution, Fast or Balanced; unsupported qualities are rejected.
  Additional reviewed variants need quality/latency benchmarks before exposure.
- The pinned SAM decoder requires a box; click-only prompts use the full-image box. It has
  no prior-mask-logit input, so refinement re-decodes accumulated points/box, not native mask
  feedback or video memory.
- Candidates depend on BiRefNet foreground alpha. Touching subjects may merge, tiny subjects
  may be discarded and missed foreground cannot be recovered by ranking. Use candidate
  review or manual SAM points for ambiguous images.
- Model output is float, converted into native 8-bit selection/mask coverage. Original
  8/16/32-bit pixels retain their format/profile. Model input is CMS-converted to sRGB and HDR
  is clipped for these exports. Large inputs are resized, not tiled; fine-detail guarantees
  have not been tested. Per-inference hash verification adds I/O cost.
- Windows CUDA execution, RTX 2070/8 GB memory and speed, packaging, full-model photographs,
  anime illustrations and labelled visual regression fixtures still need qualification.
- New Spanish labels are translated. Other locale catalogs contain English entries for the
  new labels pending native-language review.
- The M3 timings/screenshots in local-models-validation.md are inherited from PR #1199, not
  measurements repeated here. Full learned-model smoke tests require explicit installation.

The compilation environment reset during development, so interrupted checks are not counted
as passing. Final verification below records only completed runs on the delivered code.
An upstream draft PR is deferred until model/hardware qualification and coordination with
the existing upstream contribution are ready.

## Final verification

Pending final runs; this section will be updated with actual results before delivery.

