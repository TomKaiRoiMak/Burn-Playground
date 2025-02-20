use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::Tensor,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear_hidden: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    hidden_size_1: usize,
    hidden_size_2: usize,
    input: usize,
    output: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear1: LinearConfig::new(self.input, self.hidden_size_1).init(device),
            linear_hidden: LinearConfig::new(self.hidden_size_1, self.hidden_size_2).init(device),
            linear2: LinearConfig::new(self.hidden_size_2, self.output).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn march(&self, to_march: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(to_march);
        let x = self.activation.forward(x);
        let x = self.linear_hidden.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x)
    }
}
