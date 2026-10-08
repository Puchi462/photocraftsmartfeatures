//! Revision-bound local AI previews and native selection/mask application.
use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};
use photocraft_algo::matting::{self, RefineParams};
use photocraft_algo::selection::{Region, SelectionMode, combine_region};
use photocraft_doc::{DocId, Document, Layer, LayerContent, LayerId, LayerMask};
use photocraft_ml::{ModelId, ObjectPrompt, PointPrompt, Prompt, candidates::Candidate};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
enum Output {
    #[default]
    Selection,
    LayerMask,
    NewLayer,
}
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Mode {
    #[default]
    Replace,
    Add,
    Subtract,
    Intersect,
}
impl Mode {
    fn native(self) -> SelectionMode {
        match self {
            Self::Replace => SelectionMode::Replace,
            Self::Add => SelectionMode::Add,
            Self::Subtract => SelectionMode::Subtract,
            Self::Intersect => SelectionMode::Intersect,
        }
    }
}
#[derive(Clone, Default, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
struct Request {
    layer: Option<u64>,
    sample_all_layers: bool,
    preview: bool,
    output: Option<Output>,
    mode: Mode,
    edge_feather: f32,
    edge_shift: f32,
    edge_radius: f32,
    decontaminate: bool,
    quality: Option<String>,
    rect: Option<[f64; 4]>,
    points: Vec<InputPoint>,
    all_subjects: bool,
    candidate: Option<usize>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputPoint {
    position: [f64; 2],
    #[serde(default = "positive")]
    positive: bool,
}
fn positive() -> bool {
    true
}
fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}
impl Request {
    fn parse(cmd: &str, params: &Value, default: Output) -> Result<Self> {
        let mut r: Self = serde_json::from_value(if params.is_null() { json!({}) } else { params.clone() }).map_err(|e| bad(cmd, e.to_string()))?;
        r.output.get_or_insert(default);
        for (name, v, lo, hi) in
            [("edgeFeather", r.edge_feather, 0.0, 128.0), ("edgeShift", r.edge_shift, -100.0, 100.0), ("edgeRadius", r.edge_radius, 0.0, 64.0)]
        {
            if !v.is_finite() || !(lo..=hi).contains(&v) {
                return Err(bad(cmd, format!("{name} must be finite in {lo}..{hi}")));
            }
        }
        if r.quality.as_deref().is_some_and(|q| q != "high") {
            return Err(bad(cmd, "the reviewed BiRefNet HR export has a fixed 2048px input; only high quality is supported"));
        }
        if r.points.len() > ObjectPrompt::MAX_POINTS {
            return Err(bad(cmd, "at most 64 positive/negative points are supported"));
        }
        if r.candidate.is_some_and(|i| i >= 8) || (r.all_subjects && r.candidate.is_some()) {
            return Err(bad(cmd, "candidate must be in 0..8 and cannot be combined with allSubjects"));
        }
        if r.decontaminate {
            r.output = Some(Output::NewLayer);
        }
        Ok(r)
    }
    fn refine(&self) -> RefineParams {
        RefineParams { radius: self.edge_radius, feather: self.edge_feather, shift_edge: self.edge_shift, ..Default::default() }
    }
}
#[derive(Clone)]
pub(crate) struct AiPreview {
    source: Arc<Document>,
    display: Arc<Document>,
    layer: Option<LayerId>,
    request: Request,
    region: Option<Region>,
    candidates: Vec<Candidate>,
    label: &'static str,
    serial: u64,
}
impl Session {
    pub fn ai_preview_document(&self, id: DocId) -> Option<(Arc<Document>, u64)> {
        let p = self.ai_preview.as_ref()?;
        let current = self.documents().iter().find(|d| d.doc.id == id)?;
        if !Arc::ptr_eq(&current.doc, &p.source) {
            return None;
        }
        Some((p.display.clone(), (1 << 57) | p.serial))
    }
    pub fn ai_preview_candidates(&self) -> &[Candidate] {
        self.ai_preview.as_ref().map(|p| p.candidates.as_slice()).unwrap_or(&[])
    }
}
fn enabled(s: &Session) -> std::result::Result<(), String> {
    s.active().ok_or("no document open")?;
    if s.model_backend.is_none() {
        Err("local AI is unavailable; build with local-ml and install models in Preferences › Integrations".into())
    } else {
        Ok(())
    }
}
fn validate_output(doc: &Document, layer: Option<LayerId>, r: &Request) -> Result<()> {
    if matches!(r.output, Some(Output::LayerMask | Output::NewLayer)) {
        let id = layer.ok_or_else(|| EngineError::Other("choose an active pixel layer for mask or new-layer output".into()))?;
        crate::cutout_cmds::check(doc, doc.layer(id).ok_or(EngineError::NoLayer(id))?).map_err(EngineError::Other)?;
    }
    Ok(())
}
fn apply_region(doc: &mut Document, active: &mut Option<LayerId>, layer: Option<LayerId>, r: &Request, region: Option<&Region>) -> Result<Value> {
    validate_output(doc, layer, r)?;
    let output = r.output.unwrap_or_default();
    if matches!(output, Output::Selection) {
        doc.selection = combine_region(doc.selection.as_ref(), region, r.mode.native());
        return Ok(json!({"output":output,"selected":doc.selection.is_some()}));
    }
    let id = layer.ok_or_else(|| EngineError::Other("no active layer".into()))?;
    let region = region.ok_or_else(|| EngineError::Other("no foreground found; try Object Selection with a box or points".into()))?;
    let mut coverage = region.clone();
    let source = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    if let Some(mask) = &source.mask {
        let w = coverage.bbox.width() as usize;
        for (y, row) in coverage.mask.chunks_mut(w.max(1)).enumerate() {
            for (x, value) in row.iter_mut().enumerate() {
                *value = (*value as f32 * mask.value(coverage.bbox.x0 + x as i32, coverage.bbox.y0 + y as i32)).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    let mask = LayerMask { surface: matting::region_surface(&coverage), ..LayerMask::reveal_all() };
    let applied = if matches!(output, Output::NewLayer) {
        let pixels = source.surface().ok_or_else(|| EngineError::Other("new-layer output needs a pixel layer".into()))?;
        if r.decontaminate && region.mask.len() > 16_000_000 {
            return Err(EngineError::Other("decontamination is limited to 16 megapixels; reduce the canvas or turn it off".into()));
        }
        let pixels = if r.decontaminate { matting::decontaminate(pixels, region, r.edge_radius.max(4.0), 100.0) } else { pixels.clone() };
        let mut copy = Layer::new(format!("{} AI", source.name), LayerContent::Raster(pixels));
        copy.opacity = source.opacity;
        copy.blend = source.blend;
        copy.mask = Some(mask);
        let new_id = doc.insert_above(Some(id), copy);
        if let Some(source) = doc.layer_mut(id) {
            source.visible = false;
        }
        *active = Some(new_id);
        new_id
    } else {
        crate::extra_cmds::background_to_layer_for_mask(doc, id);
        doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.mask = Some(mask);
        id
    };
    doc.selection = None;
    Ok(json!({"output":output,"layer":applied.0,"selected":false}))
}
fn object_prompt(r: &Request, canvas: photocraft_geom::Rect) -> Result<ObjectPrompt> {
    let cmd = "select.aiObject";
    let (w, h) = (canvas.width() as f64, canvas.height() as f64);
    if w == 0.0 || h == 0.0 {
        return Err(bad(cmd, "the canvas is empty"));
    }
    let norm = |x: f64, y: f64| -> Result<[f32; 2]> {
        if !x.is_finite() || !y.is_finite() || x < canvas.x0 as f64 || x > canvas.x1 as f64 || y < canvas.y0 as f64 || y > canvas.y1 as f64 {
            return Err(bad(cmd, "points must be finite document coordinates inside the canvas"));
        }
        Ok([((x - canvas.x0 as f64) / w) as f32, ((y - canvas.y0 as f64) / h) as f32])
    };
    let bounds = if let Some([x, y, rw, rh]) = r.rect {
        let (x1, y1) = (x + rw, y + rh);
        if ![x, y, x1, y1].iter().all(|v| v.is_finite()) {
            return Err(bad(cmd, "box coordinates must be finite"));
        }
        let a = norm(x.min(x1).clamp(canvas.x0 as f64, canvas.x1 as f64), y.min(y1).clamp(canvas.y0 as f64, canvas.y1 as f64))?;
        let b = norm(x.max(x1).clamp(canvas.x0 as f64, canvas.x1 as f64), y.max(y1).clamp(canvas.y0 as f64, canvas.y1 as f64))?;
        [a[0], a[1], b[0], b[1]]
    } else {
        if !r.points.iter().any(|p| p.positive) {
            return Err(bad(cmd, "provide a box or at least one positive point"));
        }
        // This pinned export requires a box. Point-only requests use the entire image.
        [0.0, 0.0, 1.0, 1.0]
    };
    let points =
        r.points.iter().map(|p| Ok(PointPrompt { position: norm(p.position[0], p.position[1])?, positive: p.positive })).collect::<Result<Vec<_>>>()?;
    let p = ObjectPrompt { bounds, points };
    p.validate().map_err(crate::model_cmds::error)?;
    Ok(p)
}
enum Pipeline {
    Single(ModelId, Prompt),
    Subject,
}
fn remove_background(s: &mut Session, p: &Value) -> Result<Value> {
    let r = Request::parse("layer.aiRemoveBackground", p, Output::LayerMask)?;
    if r.rect.is_some() || !r.points.is_empty() || r.all_subjects || r.candidate.is_some() {
        return Err(bad("layer.aiRemoveBackground", "prompts and candidates belong to the AI selection commands"));
    }
    start(s, r, Pipeline::Single(ModelId::BiRefNet, Prompt::Subject), "AI Remove Background")
}
fn object_selection(s: &mut Session, p: &Value) -> Result<Value> {
    let r = Request::parse("select.aiObject", p, Output::Selection)?;
    if r.all_subjects || r.candidate.is_some() {
        return Err(bad("select.aiObject", "automatic candidates belong to select.aiSubject"));
    }
    let prompt = object_prompt(&r, s.active().ok_or(EngineError::NoDocument)?.doc.bounds())?;
    start(s, r, Pipeline::Single(ModelId::Sam2, Prompt::Object(prompt)), "AI Object Selection")
}
fn subject_selection(s: &mut Session, p: &Value) -> Result<Value> {
    let mut r = Request::parse("select.aiSubject", p, Output::Selection)?;
    if p.get("sampleAllLayers").is_none() {
        r.sample_all_layers = true;
    }
    if r.rect.is_some() || !r.points.is_empty() {
        return Err(bad("select.aiSubject", "manual prompts belong to select.aiObject"));
    }
    start(s, r, Pipeline::Subject, "AI Select Subject")
}
fn automatic_subject(
    backend: &dyn photocraft_ml::InferenceBackend,
    sampler: &dyn photocraft_algo::segment::Sampler,
    doc: &Document,
    r: &Request,
    ctx: &crate::jobs::JobCtx,
) -> Result<(Option<Region>, Vec<Candidate>)> {
    let matte = crate::model_cmds::infer_mask(backend, ModelId::BiRefNet, sampler, doc, Prompt::Subject, ctx)?;
    let candidates = ctx
        .stage(0.35, 0.4, "Ranking foreground candidates", |ctl| photocraft_ml::candidates::foreground_candidates(&matte, ctl))
        .map_err(crate::model_cmds::error)?;
    let best =
        candidates.first().ok_or_else(|| EngineError::Other("BiRefNet found no foreground candidate; use AI Object Selection with manual prompts".into()))?;
    let chosen = r.candidate.unwrap_or(0);
    if chosen >= candidates.len() {
        return Err(bad("select.aiSubject", format!("candidate {chosen} is unavailable; this image has {} candidates", candidates.len())));
    }
    let mut merged: Option<photocraft_ml::AlphaMask> = None;
    for (i, c) in candidates.iter().enumerate() {
        if (!r.all_subjects && i != chosen) || (r.all_subjects && c.score < best.score * 0.25) {
            continue;
        }
        ctx.check()?;
        let prompt = Prompt::Object(ObjectPrompt { bounds: c.bounds, points: vec![PointPrompt { position: c.seed, positive: true }] });
        let mask = crate::model_cmds::infer_mask(backend, ModelId::Sam2, sampler, doc, prompt, ctx)?;
        if let Some(current) = &mut merged {
            if current.width != mask.width || current.height != mask.height || current.kind != mask.kind {
                return Err(EngineError::Other("SAM returned incompatible candidate masks".into()));
            }
            for (a, b) in current.values.iter_mut().zip(&mask.values) {
                *a = a.max(*b);
            }
        } else {
            merged = Some(mask);
        }
    }
    let region = merged.as_ref().map(|m| crate::model_cmds::mask_region(m, doc.bounds(), ctx)).transpose()?.flatten();
    Ok((region, candidates))
}
fn start(s: &mut Session, r: Request, pipeline: Pipeline, label: &'static str) -> Result<Value> {
    let backend = crate::model_cmds::backend(s)?;
    let current = s.active().ok_or(EngineError::NoDocument)?;
    let source = current.doc.clone();
    let layer = r.layer.map(LayerId).or(current.active_layer);
    validate_output(&source, layer, &r)?;
    let worker = r.clone();
    let worker_source = source.clone();
    crate::jobs::run(
        s,
        label,
        true,
        move |ctx| {
            crate::smartselect_cmds::with_doc_sampler(&worker_source, layer, worker.sample_all_layers, |sampler, doc| {
                let (raw, candidates) = match pipeline {
                    Pipeline::Single(model, prompt) => (crate::model_cmds::infer_region(backend.as_ref(), model, sampler, doc, prompt, ctx)?, vec![]),
                    Pipeline::Subject => automatic_subject(backend.as_ref(), sampler, doc, &worker, ctx)?,
                };
                if worker.refine() == RefineParams::default() {
                    return Ok((raw, candidates));
                }
                ctx.check()?;
                let refined =
                    raw.as_ref().and_then(|region| matting::refine_mask(sampler, &matting::region_reader(region), region.bbox, doc.bounds(), &worker.refine()));
                ctx.check()?;
                Ok((refined, candidates))
            })
        },
        move |s, result| finish(s, source, layer, r, result, label),
    )
}
fn finish(
    s: &mut Session,
    source: Arc<Document>,
    layer: Option<LayerId>,
    r: Request,
    result: (Option<Region>, Vec<Candidate>),
    label: &'static str,
) -> Result<Value> {
    let (region, candidates) = result;
    if !s.active().is_some_and(|d| Arc::ptr_eq(&d.doc, &source)) {
        return Err(EngineError::Other("the source document changed; recompute the AI result".into()));
    }
    if r.preview {
        let mut shown = (*source).clone();
        let mut active = layer;
        let result = apply_region(&mut shown, &mut active, layer, &r, region.as_ref())?;
        s.ai_preview_serial = s.ai_preview_serial.saturating_add(1);
        let response = json!({"preview":true,"result":result,"candidates":candidates});
        s.ai_preview = Some(AiPreview { source, display: Arc::new(shown), layer, request: r, region, candidates, label, serial: s.ai_preview_serial });
        return Ok(response);
    }
    let mut result = s.edit(label, |doc, active| apply_region(doc, active, layer, &r, region.as_ref()))?;
    s.ai_preview = None;
    result["candidates"] = json!(candidates);
    Ok(result)
}
fn apply_preview(s: &mut Session, _: &Value) -> Result<Value> {
    let p = s.ai_preview.clone().ok_or_else(|| EngineError::Other("compute an AI preview first".into()))?;
    let mut r = p.request;
    r.preview = false;
    finish(s, p.source, p.layer, r, (p.region, p.candidates), p.label)
}
pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "layer.aiRemoveBackground",
            label: "AI Remove Background…",
            menu: &["Layer"],
            shortcut: None,
            params: r#"{"layer":id?,"sampleAllLayers":bool=false,"preview":bool=false,"quality":"high","output":"selection|layerMask|newLayer"="layerMask","edgeFeather":0..128=0,"edgeShift":-100..100=0,"edgeRadius":0..64=0,"decontaminate":bool=false} (BiRefNet HR; originals preserved)"#,
            enabled,
            run: remove_background,
            journal: true,
        },
        CommandSpec {
            id: "select.aiObject",
            label: "AI Object Selection…",
            menu: &["Select"],
            shortcut: None,
            params: r#"{"rect":[x,y,width,height]?,"points":[{"position":[x,y],"positive":bool=true}]?,"sampleAllLayers":bool=false,"preview":bool=false,"mode":"replace|add|subtract|intersect"="replace","output":"selection|layerMask|newLayer"="selection","edgeFeather":0..128=0,"edgeShift":-100..100=0,"edgeRadius":0..64=0} (SAM 2.1; document pixels; max 64 points; omitted box uses the full image)"#,
            enabled,
            run: object_selection,
            journal: true,
        },
        CommandSpec {
            id: "select.aiSubject",
            label: "AI Select Subject…",
            menu: &["Select"],
            shortcut: None,
            params: r#"{"sampleAllLayers":bool=true,"preview":bool=false,"allSubjects":bool=false,"candidate":0..7?,"mode":"replace|add|subtract|intersect"="replace","output":"selection|layerMask|newLayer"="selection","edgeFeather":0..128=0,"edgeShift":-100..100=0,"edgeRadius":0..64=0} (BiRefNet candidates ranked by prominence, refined by SAM 2.1; candidates returned for review)"#,
            enabled,
            run: subject_selection,
            journal: true,
        },
        CommandSpec {
            id: "ai.apply",
            label: "Apply AI Preview",
            menu: &[],
            shortcut: None,
            params: "{} (one undo step, only if the source is unchanged)",
            enabled: |s| if s.ai_preview.is_some() { Ok(()) } else { Err("no AI preview".into()) },
            run: apply_preview,
            journal: true,
        },
        CommandSpec {
            id: "ai.discard",
            label: "Discard AI Preview",
            menu: &[],
            shortcut: None,
            params: "{} (discard without editing)",
            enabled: |_| Ok(()),
            run: |s, _| {
                s.ai_preview = None;
                Ok(json!({"discarded":true}))
            },
            journal: false,
        },
        CommandSpec {
            id: "ai.previewInfo",
            label: "AI Preview Status",
            menu: &[],
            shortcut: None,
            params: "{} → {ready,stale,document,output,candidates}",
            enabled: |_| Ok(()),
            run: |s, _| {
                Ok(match &s.ai_preview {
                    Some(p) => {
                        json!({"ready":true,"stale":!s.active().is_some_and(|d|Arc::ptr_eq(&d.doc,&p.source)),"document":p.source.id.0,"output":p.request.output,"candidates":p.candidates})
                    }
                    None => json!({"ready":false}),
                })
            },
            journal: false,
        },
    ]
}
#[cfg(test)]
mod tests;
