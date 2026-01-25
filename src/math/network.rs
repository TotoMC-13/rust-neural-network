use crate::math::matrix::Matrix;
use crate::math::layer::Layer;

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Network {
        Network { layers }
    }

    pub fn feed_forward(&self, mut inputs: Matrix) -> Matrix {
        // Le paso a la siguiente capa la anterior como input
        for layer in &self.layers {
            inputs = layer.feed_forward(&inputs);
        }

        inputs
    }
}