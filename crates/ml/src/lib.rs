//! Optional, local image segmentation. The default build contains only the model catalogue and
//! the backend seam; `onnx` adds native CPU inference and explicit, verified downloads.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod candidates;
pub mod catalog;
mod image;
mod prompts;
pub use catalog::{MODELS, ModelId, ModelInfo};
pub use image::{AlphaMask, MaskKind, normalize};
pub use prompts::{ObjectPrompt, PointPrompt};

#[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
mod native;
#[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
pub use native::NativeBackend;

#[cfg(all(test, feature = "onnx", not(target_arch = "wasm32")))]
mod test_models;

use photocraft_algo::segment::RgbImage;
use photocraft_raster::Interrupt;
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cancelled")]
    Cancelled,
    #[error("invalid model input: {0}")]
    Input(String),
    #[error("local model unavailable: {0}")]
    Unavailable(String),
    #[error("model download: {0}")]
    Download(String),
    #[error("model inference: {0}")]
    Inference(String),
    #[error("model storage: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn check(ctl: &Interrupt<'_>) -> Result<()> {
    ctl.check().map_err(|_| Error::Cancelled)
}

/// Coordinates relative to the full input image, in 0..=1. A box guides SAM to one object;
/// automatic foreground matting uses `Subject`.
#[derive(Clone, Debug, PartialEq)]
pub enum Prompt {
    Subject,
    Box([f32; 4]),
    Object(ObjectPrompt),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    #[default]
    Cpu,
    Auto,
    Cuda,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeviceStatus {
    pub requested: Provider,
    pub active: Provider,
    pub cuda_compiled: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    pub id: ModelId,
    pub installed: bool,
    pub busy: bool,
}

/// No document or UI types cross this seam. Tests can inject a deterministic backend; the web
/// and builds without `onnx` keep the existing classical algorithms and never access the network.
pub trait InferenceBackend: Send + Sync {
    fn status(&self) -> Vec<ModelStatus>;
    fn download(&self, id: ModelId, ctl: &Interrupt<'_>) -> Result<()>;
    fn remove(&self, id: ModelId, ctl: &Interrupt<'_>) -> Result<()>;
    fn infer(&self, id: ModelId, image: &RgbImage, prompt: Prompt, ctl: &Interrupt<'_>) -> Result<AlphaMask>;
    fn device(&self) -> DeviceStatus {
        DeviceStatus { requested: Provider::Cpu, active: Provider::Cpu, cuda_compiled: false }
    }
    fn configure(&self, provider: Provider) -> Result<()> {
        if provider == Provider::Cuda { Err(Error::Unavailable("this backend has no CUDA provider".into())) } else { Ok(()) }
    }
    fn release(&self) -> Result<()> {
        Ok(())
    }
}
