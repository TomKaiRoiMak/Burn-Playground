use crate::{
    data::{NumsBatcher, NumsItem},
    training::TrainingConfig,
};
use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    prelude::Backend,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::cast::ToElement,
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: NumsItem) -> String {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let mut model = config.model_config.init::<B>(&device);
    model = model
        .load_file(&format!("{artifact_dir}/model"), &recorder, &device)
        .expect("Model should be loaded successfully");

    let label = item.nums_labels;
    let batcher = NumsBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.march(batch.nums_nums);
    // let predicted = (output.clone().into_scalar().to_f32() + 0.5).floor().to_i32();
    let predicted = output.clone().into_scalar().to_f32();
    println!("Predicted {} Expected {}", predicted, label);
    predicted.to_string()
    // println!("Output tensor: {:?}", output);
}
