use burn::backend::{Autodiff, Wgpu};

pub mod get;
pub type MyBackend = Wgpu<f32, i32>;
pub type MyAutodiffBackend = Autodiff<MyBackend>;
