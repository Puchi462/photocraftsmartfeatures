use super::*;
use photocraft_algo::segment::RgbImage;
use photocraft_ml::{AlphaMask, InferenceBackend, MaskKind, ModelStatus};
use photocraft_raster::Interrupt;
struct Fake;
impl InferenceBackend for Fake {
    fn status(&self) -> Vec<ModelStatus> {
        vec![]
    }
    fn download(&self, _: ModelId, _: &Interrupt<'_>) -> photocraft_ml::Result<()> {
        Ok(())
    }
    fn remove(&self, _: ModelId, _: &Interrupt<'_>) -> photocraft_ml::Result<()> {
        Ok(())
    }
    fn infer(&self, _: ModelId, _: &RgbImage, _: Prompt, ctl: &Interrupt<'_>) -> photocraft_ml::Result<AlphaMask> {
        if ctl.cancelled() {
            return Err(photocraft_ml::Error::Cancelled);
        }
        Ok(AlphaMask { width: 2, height: 1, values: vec![0.0, 1.0], kind: MaskKind::Alpha })
    }
}
fn session(depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width":12,"height":6,"depth":depth})).unwrap();
    s.model_backend = Some(Arc::new(Fake));
    s
}
#[test]
fn preview_apply_undo_redo_and_pixel_preservation_at_every_depth() {
    for depth in [8, 16, 32] {
        let mut s = session(depth);
        let before = s.active().unwrap().doc.clone();
        let steps = s.active().unwrap().history.past_len();
        s.execute("layer.aiRemoveBackground", json!({"preview":true})).unwrap();
        assert!(Arc::ptr_eq(&before, &s.active().unwrap().doc));
        assert_eq!(s.active().unwrap().history.past_len(), steps);
        assert!(s.ai_preview_document(before.id).unwrap().0.layers[0].mask.is_some());
        s.execute("ai.apply", json!({})).unwrap();
        assert_eq!(s.active().unwrap().history.past_len(), steps + 1);
        let after = s.active().unwrap().doc.clone();
        assert!(after.layers[0].mask.is_some());
        assert_eq!(before.layers[0].surface().unwrap().read_region(before.bounds()), after.layers[0].surface().unwrap().read_region(after.bounds()));
        s.execute("edit.undo", json!({})).unwrap();
        assert!(s.active().unwrap().doc.layers[0].mask.is_none());
        s.execute("edit.redo", json!({})).unwrap();
        assert!(s.active().unwrap().doc.layers[0].mask.is_some());
    }
}
#[test]
fn stale_and_discarded_previews_are_rejected() {
    let mut s = session(8);
    s.execute("layer.aiRemoveBackground", json!({"preview":true})).unwrap();
    s.execute("layer.new.layer", json!({"name":"other"})).unwrap();
    let before = s.active().unwrap().doc.clone();
    assert!(s.execute("ai.apply", json!({})).is_err());
    assert!(Arc::ptr_eq(&before, &s.active().unwrap().doc));
    assert!(s.ai_preview_document(before.id).is_none());
    s.execute("ai.discard", json!({})).unwrap();
    assert!(s.execute("ai.apply", json!({})).is_err());
}
#[test]
fn native_outputs_preserve_existing_mask_and_source() {
    let mut s = session(16);
    s.edit("existing mask", |doc, _| {
        let mut mask = LayerMask::reveal_all();
        mask.surface = photocraft_raster::Surface::with_default(photocraft_color::PixelFormat::GRAY8, &[0.5]);
        doc.layers[0].mask = Some(mask);
        Ok(())
    })
    .unwrap();
    s.execute("layer.aiRemoveBackground", json!({})).unwrap();
    assert!((s.active().unwrap().doc.layers[0].mask.as_ref().unwrap().value(11, 3) - 0.5).abs() < 0.01);
    let mut s = session(16);
    s.execute("layer.aiRemoveBackground", json!({"output":"selection"})).unwrap();
    assert!(s.active().unwrap().doc.selection.is_some());
    assert!(s.active().unwrap().doc.layers[0].mask.is_none());
    let mut s = session(16);
    s.edit("source appearance", |doc, _| {
        doc.layers[0].fill_opacity = 0.65;
        doc.layers[0].psd_id = Some(777);
        Ok(())
    })
    .unwrap();
    let source_id = s.active().unwrap().doc.layers[0].id;
    s.execute("layer.aiRemoveBackground", json!({"output":"newLayer"})).unwrap();
    assert_eq!(s.active().unwrap().doc.layers.len(), 2);
    assert!(!s.active().unwrap().doc.layers[0].visible);
    assert!(s.active().unwrap().doc.layer(s.active().unwrap().active_layer.unwrap()).unwrap().mask.is_some());
    let copy = s.active().unwrap().doc.layer(s.active().unwrap().active_layer.unwrap()).unwrap();
    assert_ne!(copy.id, source_id);
    assert_eq!(copy.fill_opacity, 0.65);
    assert!(copy.psd_id.is_none());
}
#[test]
fn native_masks_survive_psd_and_pcraft_save_reopen() {
    for depth in [8, 16, 32] {
        for (id, params) in [
            ("layer.aiRemoveBackground", json!({"output":"layerMask"})),
            ("select.aiObject", json!({"points":[{"position":[8,3]}],"output":"layerMask"})),
            ("select.aiSubject", json!({"output":"layerMask"})),
        ] {
            let mut s = session(depth);
            s.execute(id, params).unwrap();
            let doc = &s.active().unwrap().doc;
            let value = doc.layers[0].mask.as_ref().unwrap().value(5, 3);
            for extension in ["psd", "pcraft"] {
                let out = photocraft_io::export(doc, extension, &Default::default()).unwrap();
                let back = photocraft_io::import(&format!("roundtrip.{extension}"), &out.bytes).unwrap().document;
                assert!((back.layers[0].mask.as_ref().unwrap().value(5, 3) - value).abs() < 0.01, "{id} {extension} {depth}");
            }
        }
    }
}
#[test]
fn parameters_and_prompt_coordinates_are_bounded() {
    let canvas = photocraft_geom::Rect::new(0, 0, 120, 60);
    let r = Request::parse("select.aiObject", &json!({"rect":[100,50,-80,-40],"points":[{"position":[60,30],"positive":false}]}), Output::Selection).unwrap();
    let p = object_prompt(&r, canvas).unwrap();
    assert_eq!(p.bounds, [20.0 / 120.0, 10.0 / 60.0, 100.0 / 120.0, 50.0 / 60.0]);
    assert_eq!(p.points[0].position, [0.5, 0.5]);
    assert!(!p.points[0].positive);
    let mut s = session(8);
    let before = s.active().unwrap().doc.clone();
    for params in [json!({"quality":"fast"}), json!({"edgeFeather":-1}), json!({"edgeShift":101}), json!({"edgeRadius":65}), json!({"unknown":1})] {
        assert!(s.execute("layer.aiRemoveBackground", params).is_err());
        assert!(Arc::ptr_eq(&before, &s.active().unwrap().doc));
    }
    for params in [
        json!({}),
        json!({"points":[{"position":[5,2],"positive":false}]}),
        json!({"points":[{"position":[-1,2]}]}),
        json!({"rect":[1,1,0,1]}),
        json!({"points":vec![json!({"position":[1,1]});65]}),
    ] {
        assert!(s.execute("select.aiObject", params).is_err());
    }
    s.model_backend = None;
    assert!(s.execute("layer.aiRemoveBackground", json!({})).is_err());
}
#[test]
fn object_selection_combines_native_coverage() {
    let mut s = session(8);
    let params = json!({"points":[{"position":[8,3]}]});
    s.execute("select.aiObject", params.clone()).unwrap();
    let bounds = s.active().unwrap().doc.bounds();
    let original = s.active().unwrap().doc.selection.as_ref().unwrap().read_region(bounds);
    for mode in ["add", "intersect"] {
        s.execute("select.aiObject", json!({"points":[{"position":[8,3]}],"mode":mode})).unwrap();
        assert_eq!(s.active().unwrap().doc.selection.as_ref().unwrap().read_region(bounds), original);
    }
    s.execute("select.aiObject", json!({"points":[{"position":[8,3]}],"mode":"subtract"})).unwrap();
    assert!(s.active().unwrap().doc.selection.is_none());
}
struct CandidatesBackend {
    calls: std::sync::Mutex<Vec<(ModelId, Prompt)>>,
    empty: bool,
}
impl InferenceBackend for CandidatesBackend {
    fn status(&self) -> Vec<ModelStatus> {
        vec![]
    }
    fn download(&self, _: ModelId, _: &Interrupt<'_>) -> photocraft_ml::Result<()> {
        Ok(())
    }
    fn remove(&self, _: ModelId, _: &Interrupt<'_>) -> photocraft_ml::Result<()> {
        Ok(())
    }
    fn infer(&self, id: ModelId, _: &RgbImage, prompt: Prompt, _: &Interrupt<'_>) -> photocraft_ml::Result<AlphaMask> {
        self.calls.lock().unwrap().push((id, prompt));
        let mut values = vec![0.0; 32];
        if !self.empty {
            for y in 0..4 {
                for x in [1, 2, 5, 6] {
                    values[y * 8 + x] = 0.9;
                }
            }
        }
        Ok(AlphaMask { width: 8, height: 4, values, kind: MaskKind::Alpha })
    }
}
#[test]
fn subject_runs_birefnet_then_sam_and_allows_candidate_choice() {
    let mut s = session(16);
    let b = Arc::new(CandidatesBackend { calls: Default::default(), empty: false });
    s.model_backend = Some(b.clone());
    let before = s.active().unwrap().doc.clone();
    let result = s.execute("select.aiSubject", json!({"preview":true})).unwrap();
    assert!(Arc::ptr_eq(&before, &s.active().unwrap().doc));
    assert_eq!(result["candidates"].as_array().unwrap().len(), 2);
    let seed = s.ai_preview_candidates()[0].seed;
    {
        let calls = b.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, ModelId::BiRefNet);
        assert_eq!(calls[1].0, ModelId::Sam2);
        assert!(matches!(&calls[1].1,Prompt::Object(p) if p.points.len()==1 && p.points[0].positive && p.points[0].position==seed));
    }
    s.execute("select.aiSubject", json!({"candidate":1,"preview":true})).unwrap();
    s.execute("ai.apply", json!({})).unwrap();
    assert!(s.active().unwrap().doc.selection.is_some());
    b.calls.lock().unwrap().clear();
    s.execute("select.aiSubject", json!({"allSubjects":true})).unwrap();
    assert_eq!(b.calls.lock().unwrap().iter().map(|c| c.0).collect::<Vec<_>>(), [ModelId::BiRefNet, ModelId::Sam2, ModelId::Sam2]);
    let before = s.active().unwrap().doc.clone();
    for p in [json!({"candidate":2}), json!({"candidate":8}), json!({"candidate":0,"allSubjects":true}), json!({"rect":[0,0,4,4]})] {
        assert!(s.execute("select.aiSubject", p).is_err());
        assert!(Arc::ptr_eq(&before, &s.active().unwrap().doc));
    }
    s.model_backend = Some(Arc::new(CandidatesBackend { calls: Default::default(), empty: true }));
    assert!(s.execute("select.aiSubject", json!({})).unwrap_err().to_string().contains("no foreground candidate"));
    assert!(Arc::ptr_eq(&before, &s.active().unwrap().doc));
}
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn cancelling_inference_preserves_pixels_history_and_preview() {
    struct Slow;
    impl InferenceBackend for Slow {
        fn status(&self) -> Vec<ModelStatus> {
            vec![]
        }
        fn download(&self, _: ModelId, _: &Interrupt<'_>) -> photocraft_ml::Result<()> {
            Ok(())
        }
        fn remove(&self, _: ModelId, _: &Interrupt<'_>) -> photocraft_ml::Result<()> {
            Ok(())
        }
        fn infer(&self, _: ModelId, _: &RgbImage, _: Prompt, ctl: &Interrupt<'_>) -> photocraft_ml::Result<AlphaMask> {
            let end = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while !ctl.cancelled() && std::time::Instant::now() < end {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Err(photocraft_ml::Error::Cancelled)
        }
    }
    let mut s = session(8);
    s.model_backend = Some(Arc::new(Slow));
    let before = s.active().unwrap().doc.clone();
    let steps = s.active().unwrap().history.past_len();
    let crate::jobs::Started::Job(id) = s.start("layer.aiRemoveBackground", json!({"preview":true})).unwrap() else { panic!("background job expected") };
    assert!(s.execute("ai.previewInfo", json!({})).is_ok());
    assert!(s.execute("ai.discard", json!({})).is_ok());
    assert!(s.cancel_job(id));
    assert!(matches!(s.wait_job(id), Err(EngineError::Cancelled)));
    assert!(Arc::ptr_eq(&before, &s.active().unwrap().doc));
    assert_eq!(s.active().unwrap().history.past_len(), steps);
    assert!(s.ai_preview.is_none());
}
