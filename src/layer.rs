use crate::matrix::Matrix;

pub struct Layer {
    pub weights: Matrix,
    pub biases: Matrix,
    pub activation: fn(f32) -> f32,
    pub derivative: fn(f32) -> f32,
}

impl Layer {
    pub fn new(inputs: usize, neurons: usize, activation: fn(f32) -> f32, derivative: fn(f32) -> f32) -> Layer {
        Layer {
            weights: Matrix::random(inputs, neurons),
            biases: Matrix::random(1, neurons),
            activation,
            derivative,
        }
    }

    /*  
    Hace σ(a * W + b), se hace de esta forma porque nuestros vectores estan "acostados"
    por como diseñe las matrices
    */
    pub fn feed_forward(&self, inputs: &Matrix) -> Matrix {
        (inputs.mul(&self.weights)).sum(&self.biases).map(self.activation)
    }
}