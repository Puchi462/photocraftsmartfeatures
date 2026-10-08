//! Candidates from learned foreground alpha; scores are prominence heuristics, not classes.
use crate::{AlphaMask, Error, MaskKind, Result, check};
use photocraft_raster::Interrupt;
use serde::Serialize;
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub bounds: [f32; 4],
    pub seed: [f32; 2],
    pub score: f32,
    pub matte_confidence: f32,
    pub area_fraction: f32,
}
pub fn foreground_candidates(mask: &AlphaMask, ctl: &Interrupt<'_>) -> Result<Vec<Candidate>> {
    mask.validate()?;
    if mask.kind != MaskKind::Alpha {
        return Err(Error::Input("candidate generation needs foreground alpha, not SAM logits".into()));
    }
    let (w, h) = (mask.width.min(256), mask.height.min(256));
    let n = w * h;
    let mut alpha = Vec::with_capacity(n);
    for y in 0..h {
        check(ctl)?;
        for x in 0..w {
            alpha.push(mask.sample((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32).clamp(0.0, 1.0));
        }
    }
    let mut seen = vec![false; n];
    let mut queue = Vec::with_capacity(n);
    let mut candidates = Vec::new();
    for index in 0..n {
        check(ctl)?;
        if seen.get(index).copied().unwrap_or(true) || alpha.get(index).copied().unwrap_or(0.0) < 0.5 {
            continue;
        }
        queue.clear();
        queue.push(index);
        if let Some(v) = seen.get_mut(index) {
            *v = true;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
        let (mut total, mut confidence, mut cx, mut cy) = (0usize, 0.0f32, 0.0f32, 0.0f32);
        let (mut best, mut seed) = (-1.0f32, [0.5, 0.5]);
        while let Some(i) = queue.pop() {
            if total % 1024 == 0 {
                check(ctl)?;
            }
            let (x, y) = (i % w, i / w);
            let a = alpha.get(i).copied().unwrap_or(0.0);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
            total += 1;
            confidence += a;
            cx += x as f32 + 0.5;
            cy += y as f32 + 0.5;
            if a > best {
                best = a;
                seed = [(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32];
            }
            for neighbor in [(x > 0).then(|| i - 1), (x + 1 < w).then_some(i + 1), (y > 0).then(|| i - w), (y + 1 < h).then_some(i + w)].into_iter().flatten() {
                if !seen.get(neighbor).copied().unwrap_or(true) && alpha.get(neighbor).copied().unwrap_or(0.0) >= 0.5 {
                    if let Some(v) = seen.get_mut(neighbor) {
                        *v = true;
                    }
                    queue.push(neighbor);
                }
            }
        }
        if total < n.div_ceil(2000).max(1) {
            continue;
        }
        let area = total as f32 / n as f32;
        let confidence = confidence / total as f32;
        let distance = ((cx / total as f32 / w as f32 - 0.5).powi(2) + (cy / total as f32 / h as f32 - 0.5).powi(2)).sqrt() * std::f32::consts::SQRT_2;
        candidates.push(Candidate {
            bounds: [
                x0.saturating_sub(1) as f32 / w as f32,
                y0.saturating_sub(1) as f32 / h as f32,
                (x1 + 1).min(w) as f32 / w as f32,
                (y1 + 1).min(h) as f32 / h as f32,
            ],
            seed,
            score: confidence * area.sqrt() * (1.0 - 0.35 * distance.clamp(0.0, 1.0)),
            matte_confidence: confidence,
            area_fraction: area,
        });
    }
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    candidates.truncate(8);
    Ok(candidates)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rank_and_seed_and_cancel() {
        let mut m = AlphaMask { width: 12, height: 8, values: vec![0.0; 96], kind: MaskKind::Alpha };
        for y in 1..7 {
            for x in 3..8 {
                m.values[y * 12 + x] = 0.9;
            }
        }
        m.values[10] = 0.75;
        let c = foreground_candidates(&m, &Interrupt::NONE).unwrap();
        assert_eq!(c.len(), 2);
        assert!(c[0].score > c[1].score);
        assert!(m.sample(c[0].seed[0], c[0].seed[1]) > 0.5);
        assert!(foreground_candidates(&m, &Interrupt::cancel_only(&|| true)).is_err());
    }
    #[test]
    fn transparent_alpha_has_no_candidates_and_logits_are_rejected() {
        let mut m = AlphaMask { width: 2, height: 2, values: vec![0.0; 4], kind: MaskKind::Alpha };
        assert!(foreground_candidates(&m, &Interrupt::NONE).unwrap().is_empty());
        m.kind = MaskKind::Logits;
        assert!(foreground_candidates(&m, &Interrupt::NONE).is_err());
    }
}
