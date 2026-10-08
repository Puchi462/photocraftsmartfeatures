//! Parameter window and prompt gestures. The engine owns inference, previews and history.
use crate::{PhotocraftApp, theme::Tokens};
use egui::RichText;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AiUi {
    pub open: bool,
    pub command: String,
    pub params: Value,
    pub computed: Option<Value>,
    pub pending: Option<u64>,
    pub document: Option<u64>,
}
fn same_document(app: &PhotocraftApp, state: &AiUi) -> bool {
    state.document == app.session.active().map(|d| d.doc.id.0)
}
pub fn menu(app: &mut PhotocraftApp, id: &str, params: &Value) -> Option<Result<Value, String>> {
    if !matches!(id, "layer.aiRemoveBackground" | "select.aiObject" | "select.aiSubject") || params.as_object().is_some_and(|p| !p.is_empty()) {
        return None;
    }
    let mut previous = app.ui.ai.clone();
    discard(app, &mut previous);
    app.ui.ai = AiUi {
        open: true,
        command: id.into(),
        document: app.session.active().map(|d| d.doc.id.0),
        params: json!({"output":if id=="layer.aiRemoveBackground"{"layerMask"}else{"selection"},"mode":"replace","sampleAllLayers":id=="select.aiSubject","edgeFeather":0.0,"edgeShift":0.0,"edgeRadius":0.0,"decontaminate":false}),
        ..Default::default()
    };
    if id == "select.aiObject" {
        app.ui.tool = crate::state::Tool::ObjectSelection;
        app.ui.ai.params["points"] = json!([]);
    }
    Some(Ok(json!({"window":"ai"})))
}
fn compute(app: &mut PhotocraftApp, state: &mut AiUi) {
    if !same_document(app, state) {
        return;
    }
    let _ = app.run("ai.discard", json!({}));
    let mut params = state.params.clone();
    params["preview"] = json!(true);
    match app.run(&state.command, params) {
        Ok(result) => {
            state.computed = Some(state.params.clone());
            state.pending = result.get("job").and_then(Value::as_u64);
        }
        Err(_) => state.computed = None,
    }
}
fn discard(app: &mut PhotocraftApp, state: &mut AiUi) {
    if let Some(job) = state.pending.take() {
        let _ = app.session.execute("jobs.cancel", json!({"job":job}));
    }
    let _ = app.run("ai.discard", json!({}));
    state.open = false;
}
pub fn object_gesture(app: &mut PhotocraftApp, start: [f64; 2], end: [f64; 2], modifiers: egui::Modifiers) -> bool {
    if !app.ui.ai.open || app.ui.ai.command != "select.aiObject" {
        return false;
    }
    if app.session.has_jobs() || !same_document(app, &app.ui.ai) {
        return true;
    }
    let mut state = app.ui.ai.clone();
    let (w, h) = ((end[0] - start[0]).abs(), (end[1] - start[1]).abs());
    if w < 2.0 && h < 2.0 {
        if let Some(points) = state.params["points"].as_array_mut() {
            if points.len() >= 64 {
                return true;
            }
            points.push(json!({"position":end,"positive":!modifiers.alt}));
        }
    } else {
        state.params["rect"] = json!([start[0].min(end[0]), start[1].min(end[1]), w, h]);
    }
    compute(app, &mut state);
    app.ui.ai = state;
    true
}
pub fn markers(app: &PhotocraftApp, painter: &egui::Painter, xf: &crate::canvas::ViewXform) {
    if !app.ui.ai.open || !same_document(app, &app.ui.ai) {
        return;
    }
    let t = Tokens::get(painter.ctx());
    if app.ui.ai.command == "select.aiSubject" {
        if let Some((doc, _)) = app.session.active().and_then(|d| app.session.ai_preview_document(d.doc.id)) {
            let b = doc.bounds();
            for (index, c) in app.session.ai_preview_candidates().iter().enumerate() {
                let [x0, y0, x1, y1] = c.bounds;
                let rect = egui::Rect::from_two_pos(
                    xf.to_screen(b.x0 as f32 + x0 * b.width() as f32, b.y0 as f32 + y0 * b.height() as f32),
                    xf.to_screen(b.x0 as f32 + x1 * b.width() as f32, b.y0 as f32 + y1 * b.height() as f32),
                );
                painter.rect_stroke(rect, 0.0, egui::Stroke::new(1.0, t.accent), egui::StrokeKind::Inside);
                painter.text(
                    rect.left_top(),
                    egui::Align2::LEFT_TOP,
                    format!("{} {}", tl!("Candidate"), index + 1),
                    egui::FontId::proportional(12.0),
                    t.accent,
                );
            }
        }
        return;
    }
    if app.ui.ai.command != "select.aiObject" {
        return;
    }
    for p in app.ui.ai.params["points"].as_array().into_iter().flatten().take(64) {
        let Some(x) = p["position"].get(0).and_then(Value::as_f64) else { continue };
        let Some(y) = p["position"].get(1).and_then(Value::as_f64) else { continue };
        let pos = xf.to_screen(x as f32, y as f32);
        painter.circle_filled(pos, 6.0, t.field);
        painter.circle_stroke(pos, 6.0, egui::Stroke::new(1.0, t.accent));
        painter.text(
            pos,
            egui::Align2::CENTER_CENTER,
            if p["positive"].as_bool().unwrap_or(true) { "+" } else { "−" },
            egui::FontId::proportional(12.0),
            t.text,
        );
    }
}
pub fn show(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if !app.ui.ai.open {
        return;
    }
    let mut state = app.ui.ai.clone();
    let mut open = true;
    let t = Tokens::get(ctx);
    let object = state.command == "select.aiObject";
    let subject = state.command == "select.aiSubject";
    let same = same_document(app, &state);
    let title = if object {
        "AI Object Selection…"
    } else if subject {
        "AI Select Subject…"
    } else {
        "AI Remove Background…"
    };
    egui::Window::new(tl!(title)).id(egui::Id::new("ai-smart-window")).open(&mut open).default_width(390.0).show(ctx, |ui| {
        ui.label(
            RichText::new(tl!(if object {
                "SAM 2.1 Large"
            } else if subject {
                "BiRefNet candidates + SAM 2.1"
            } else {
                "BiRefNet HR Matting"
            }))
            .color(t.text_dim),
        );
        if !same {
            ui.label(tl!("Return to the source document or reopen this AI action."));
        }
        let info = app.session.execute("ai.previewInfo", json!({})).unwrap_or(Value::Null);
        if subject {
            ui.label(tl!("Candidate scores estimate prominence, not object-class confidence."));
            let mut all = state.params["allSubjects"].as_bool().unwrap_or(false);
            crate::widgets::checkbox(ui, &mut all, tl!("All prominent subjects"));
            state.params["allSubjects"] = json!(all);
            if all {
                if let Some(fields) = state.params.as_object_mut() {
                    fields.remove("candidate");
                }
            } else if let Some(candidates) = info["candidates"].as_array() {
                let mut chosen = state.params["candidate"].as_u64().unwrap_or(0) as usize;
                egui::ComboBox::from_id_salt("ai-candidate").selected_text(format!("{} {}", tl!("Candidate"), chosen + 1)).show_ui(ui, |ui| {
                    for (index, candidate) in candidates.iter().enumerate().take(8) {
                        let score = candidate["score"].as_f64().unwrap_or(0.0);
                        ui.selectable_value(&mut chosen, index, format!("{} {} ({score:.3})", tl!("Candidate"), index + 1));
                    }
                });
                if chosen != state.params["candidate"].as_u64().unwrap_or(0) as usize {
                    state.params["candidate"] = json!(chosen);
                }
            }
        }
        if object {
            ui.label(tl!("Click an object, Alt-click to exclude, or drag a box. Apply confirms the preview."));
            if crate::widgets::secondary_button(ui, tl!("Reset prompts"), 0.0).clicked() {
                state.params["points"] = json!([]);
                if let Some(fields) = state.params.as_object_mut() {
                    fields.remove("rect");
                }
                state.computed = None;
                let _ = app.run("ai.discard", json!({}));
            }
        }
        let mut all = state.params["sampleAllLayers"].as_bool().unwrap_or(false);
        crate::widgets::checkbox(ui, &mut all, tl!("Sample All Layers"));
        state.params["sampleAllLayers"] = json!(all);
        for (key, label, lo, hi) in [("edgeFeather", "Feather", 0.0, 128.0), ("edgeShift", "Shift Edge", -100.0, 100.0), ("edgeRadius", "Radius", 0.0, 64.0)] {
            ui.horizontal(|ui| {
                ui.label(tl!(label));
                let mut value = state.params[key].as_f64().unwrap_or(0.0);
                ui.add(egui::DragValue::new(&mut value).range(lo..=hi).speed(0.25));
                state.params[key] = json!(value);
            });
        }
        let mut decontaminate = state.params["decontaminate"].as_bool().unwrap_or(false);
        crate::widgets::checkbox(ui, &mut decontaminate, tl!("Decontaminate Colors"));
        state.params["decontaminate"] = json!(decontaminate);
        let mut output = state.params["output"].as_str().unwrap_or("layerMask").to_string();
        ui.horizontal(|ui| {
            ui.label(tl!("Output"));
            egui::ComboBox::from_id_salt("ai-output")
                .selected_text(tl!(match output.as_str() {
                    "selection" => "Selection",
                    "newLayer" => "New Layer",
                    _ => "Layer Mask",
                }))
                .show_ui(ui, |ui| {
                    for (key, label) in [("selection", "Selection"), ("layerMask", "Layer Mask"), ("newLayer", "New Layer")] {
                        ui.selectable_value(&mut output, key.into(), tl!(label));
                    }
                });
        });
        state.params["output"] = json!(output);
        if object || subject {
            let mut mode = state.params["mode"].as_str().unwrap_or("replace").to_string();
            egui::ComboBox::from_id_salt("ai-selection-mode").selected_text(tl!(mode.as_str())).show_ui(ui, |ui| {
                for key in ["replace", "add", "subtract", "intersect"] {
                    ui.selectable_value(&mut mode, key.into(), tl!(key));
                }
            });
            state.params["mode"] = json!(mode);
        }
        let ready = !app.session.has_jobs()
            && same
            && info["ready"].as_bool().unwrap_or(false)
            && !info["stale"].as_bool().unwrap_or(true)
            && state.computed.as_ref() == Some(&state.params);
        ui.separator();
        ui.horizontal(|ui| {
            if ui.add_enabled_ui(!app.session.has_jobs() && same, |ui| crate::widgets::secondary_button(ui, tl!("Compute AI Preview"), 0.0).clicked()).inner {
                compute(app, &mut state);
            }
            if ui.add_enabled_ui(ready, |ui| crate::widgets::secondary_button(ui, tl!("Apply"), 0.0).clicked()).inner && app.run("ai.apply", json!({})).is_ok()
            {
                state.open = false;
                state.pending = None;
            }
            if crate::widgets::secondary_button(ui, tl!("Cancel"), 0.0).clicked() {
                discard(app, &mut state);
            }
        });
        if app.ui.status_error {
            ui.label(RichText::new(&app.ui.status).color(t.text));
        }
    });
    if !open {
        discard(app, &mut state);
    }
    app.ui.ai = state;
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn menu_window_is_serialized_and_explicit_params_run_directly() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        assert!(menu(&mut app, "layer.aiRemoveBackground", &json!({})).unwrap().is_ok());
        assert_eq!(app.ui.ai.params["output"], "layerMask");
        assert!(menu(&mut app, "layer.aiRemoveBackground", &json!({"preview":true})).is_none());
        assert_eq!(serde_json::to_value(&app.ui).unwrap()["ai"]["open"], true);
    }
    #[test]
    fn prompts_cannot_leak_into_another_document() {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width":20,"height":12})).unwrap();
        let mut app = PhotocraftApp::new(s, crate::Services::default());
        menu(&mut app, "select.aiObject", &json!({})).unwrap().unwrap();
        let source = app.ui.ai.document;
        app.session.execute("file.new", json!({"width":20,"height":12})).unwrap();
        assert_ne!(source, app.session.active().map(|d| d.doc.id.0));
        assert!(object_gesture(&mut app, [4.0, 4.0], [4.0, 4.0], egui::Modifiers::NONE));
        assert!(app.ui.ai.params["points"].as_array().unwrap().is_empty());
        assert!(app.ui.ai.computed.is_none());
        menu(&mut app, "select.aiSubject", &json!({})).unwrap().unwrap();
        assert_eq!(app.ui.ai.params["sampleAllLayers"], true);
    }
}
