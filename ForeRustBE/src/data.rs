use burn::{
    data::{dataloader::batcher::Batcher, dataset::InMemDataset},
    prelude::Backend,
    tensor::{cast::ToElement, Tensor, TensorData},
};

#[derive(Clone)]
pub struct NumsBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> NumsBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct NumsBatch<B: Backend> {
    pub nums_nums: Tensor<B, 2>,
    pub nums_labels: Tensor<B, 2>,
}

impl<B: Backend> Batcher<NumsItem, NumsBatch<B>> for NumsBatcher<B> {
    fn batch(&self, items: Vec<NumsItem>) -> NumsBatch<B> {
        let nums_nums: Vec<f32> = items.iter().map(|item| item.nums_nums.to_f32()).collect();
        let nums_labels: Vec<f32> = items.iter().map(|item| item.nums_labels.to_f32()).collect();

        let nums_nums_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(nums_nums, vec![items.len(), 1]),
            &self.device,
        );
        let nums_labels_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(nums_labels, vec![items.len(), 1]),
            &self.device,
        );

        NumsBatch {
            nums_nums: nums_nums_tensor,
            nums_labels: nums_labels_tensor,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NumsItem {
    pub nums_nums: u16,
    pub nums_labels: u16,
}

pub fn get_dataset() -> InMemDataset<NumsItem> {
    let data: Vec<NumsItem> = (0..1000)
        .map(|i| NumsItem {
            nums_nums: i,
            nums_labels: i + 1,
        })
        .collect();

    InMemDataset::new(data)
}
pub fn get_valid_dataset() -> InMemDataset<NumsItem> {
    let data: Vec<NumsItem> = (1000..2000)
        .map(|i| NumsItem {
            nums_nums: i,
            nums_labels: i + 1,
        })
        .collect();

    InMemDataset::new(data)
}
