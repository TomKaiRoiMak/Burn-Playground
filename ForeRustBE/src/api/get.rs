use crate::{data::NumsItem, inference::infer};
use axum::{extract::Query, Json};
use burn::{backend::wgpu::WgpuDevice, prelude::Backend};
use serde::{Deserialize, Serialize};

use super::MyBackend;


#[derive(Deserialize, Serialize, Debug)]
pub struct RankObjectQuery {
    number: u16,
}

pub async fn test() {
    println!("Ready to Go!")
}

pub async fn get_forecast(Query(query): Query<RankObjectQuery>) -> Json<String> {
    let device = WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    let number = query.number;
    let item = NumsItem {
        nums_nums: number,
        nums_labels: number + 1,
    };
    let test = infer::<MyBackend>(artifact_dir, device.clone(), item);
    Json(test)
}
