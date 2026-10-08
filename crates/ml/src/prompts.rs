//! Bounded SAM prompts in image-normalized coordinates. The pinned decoder always accepts a
//! box alongside points; for a click-only interaction the host uses the full-image box.
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointPrompt {
    pub position: [f32; 2],
    pub positive: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObjectPrompt {
    pub bounds: [f32; 4],
    pub points: Vec<PointPrompt>,
}

impl ObjectPrompt {
    pub const MAX_POINTS: usize = 64;

    pub fn validate(&self) -> Result<()> {
        let [x0, y0, x1, y1] = self.bounds;
        let valid = |v: f32| v.is_finite() && (0.0..=1.0).contains(&v);
        if !self.bounds.iter().all(|v| valid(*v)) || x0 >= x1 || y0 >= y1 {
            return Err(Error::Input("SAM box must be a non-empty normalized image rectangle".into()));
        }
        if self.points.len() > Self::MAX_POINTS || self.points.iter().any(|p| !p.position.iter().all(|v| valid(*v))) {
            return Err(Error::Input("SAM supports up to 64 finite points inside the image".into()));
        }
        Ok(())
    }

    #[cfg(any(feature = "onnx", test))]
    pub(crate) fn tensors(&self) -> Result<(Vec<f32>, Vec<i64>)> {
        self.validate()?;
        if self.points.is_empty() {
            return Ok((vec![0.0, 0.0], vec![-1]));
        }
        let coordinates = self.points.iter().flat_map(|p| p.position.map(|v| v * 1024.0)).collect();
        let labels = self.points.iter().map(|p| i64::from(p.positive)).collect();
        Ok((coordinates, labels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn positive_negative_and_box_prompts_use_the_same_image_space() {
        let p = ObjectPrompt {
            bounds: [0.0, 0.0, 1.0, 1.0],
            points: vec![PointPrompt { position: [0.25, 0.75], positive: true }, PointPrompt { position: [1.0, 0.0], positive: false }],
        };
        assert_eq!(p.tensors().unwrap(), (vec![256.0, 768.0, 1024.0, 0.0], vec![1, 0]));
        assert_eq!(ObjectPrompt { points: vec![], ..p }.tensors().unwrap(), (vec![0.0, 0.0], vec![-1]));
    }
    #[test]
    fn invalid_and_unbounded_prompts_are_rejected() {
        let mut p = ObjectPrompt { bounds: [0.0, 0.0, 1.0, 1.0], points: vec![PointPrompt { position: [f32::NAN, 0.0], positive: true }] };
        assert!(p.validate().is_err());
        p.points = vec![PointPrompt { position: [0.5, 0.5], positive: true }; 65];
        assert!(p.validate().is_err());
        p.points.clear();
        p.bounds = [1.0, 0.0, 0.0, 1.0];
        assert!(p.validate().is_err());
    }
}
