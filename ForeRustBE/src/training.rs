use std::fs::{create_dir_all, remove_dir_all};

use crate::{
    data::{get_dataset, get_valid_dataset, NumsBatch, NumsBatcher},
    model::{Model, ModelConfig},
};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    nn::loss::{MseLoss, Reduction},
    optim::AdamConfig,
    prelude::Backend,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{backend::AutodiffBackend, Tensor},
    train::{
        metric::LossMetric, LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
};

impl<B: Backend> Model<B> {
    pub fn march_classification(
        &self,
        nums_nums: Tensor<B, 2>,
        nums_labels: Tensor<B, 2>,
    ) -> RegressionOutput<B> {
        let output = self.march(nums_nums);
        let loss = MseLoss::new().forward(output.clone(), nums_labels.clone(), Reduction::Auto);
        RegressionOutput {
            loss,
            output,
            targets: nums_labels,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<NumsBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: NumsBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.march_classification(batch.nums_nums, batch.nums_labels);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}
impl<B: Backend> ValidStep<NumsBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: NumsBatch<B>) -> RegressionOutput<B> {
        self.march_classification(batch.nums_nums, batch.nums_labels)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model_config: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 1000)]
    pub nums_epochs: usize,
    #[config(default = 1)]
    pub batch_size: usize,
    #[config(default = 10)]
    pub thread_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-6)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    remove_dir_all(artifact_dir).expect("Error");
    create_dir_all(artifact_dir).expect("Error");
}

pub fn try_train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) {
    // create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);
    let batcher_train = NumsBatcher::<B>::new(device.clone());
    let batcher_valid = NumsBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(config.thread_workers)
        .shuffle(config.seed)
        .build(get_dataset());
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.thread_workers)
        .shuffle(config.seed)
        .build(get_valid_dataset());

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let mut model = config.model_config.init::<B>(&device);
    model = model
        .load_file(&format!("{artifact_dir}/model"), &recorder, &device)
        .unwrap_or_else(|_| config.model_config.init::<B>(&device));

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(NamedMpkFileRecorder::<FullPrecisionSettings>::new())
        .devices(vec![device.clone()])
        .num_epochs(config.nums_epochs)
        .summary()
        .build(model, config.optimizer.init(), config.learning_rate);
    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .save_file(
            format!("{artifact_dir}/model"),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
        )
        .expect("Trained model should be saved successfully");
}
