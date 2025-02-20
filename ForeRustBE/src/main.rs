use std::net::SocketAddr;

use api::get::{get_forecast, test};
use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Extension, Router,
};
use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    data::dataset::Dataset,
    optim::AdamConfig,
};
use data::{get_dataset, get_valid_dataset};
use inference::infer;
use model::ModelConfig;
use tower_http::{
    cors::{Any, CorsLayer},
    limit::RequestBodyLimitLayer,
};
use training::{try_train, TrainingConfig};
mod api;
mod data;
mod inference;
mod model;
mod training;




#[tokio::main]
async fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    try_train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(1024, 1024, 1, 1), AdamConfig::new()),
        device.clone(),
    );
    // infer::<MyBackend>(artifact_dir, device, _);
    // let dataset = get_valid_dataset();
    // for i in 0..1000 {
    //     let item = dataset.get(i).unwrap();
    //     infer::<MyBackend>(artifact_dir, device.clone(), item.clone());
    // }

    // let cors = CorsLayer::new()
    //     .allow_origin(Any)
    //     .allow_methods(Any)
    //     .allow_headers(Any);
    // test().await;
    // let app = Router::new()
    //     .route("/test", get(get_forecast))
    //     .layer(DefaultBodyLimit::disable())
    //     .layer(RequestBodyLimitLayer::new(
    //         250 * 1024 * 1024, /* 250mb */
    //     ))
    //     .layer(tower_http::trace::TraceLayer::new_for_http())
    //     .layer(cors);


    // // .layer(Extension(conn));

    // let addr = SocketAddr::from(([0, 0, 0, 0], 4000));
    // println!("Listening on {}", addr);
    // let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    // axum::serve(listener, app).await.unwrap();
}
