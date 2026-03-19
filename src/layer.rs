use crate::activations::{relu, relu_prime, sigmoid, sigmoid_prime};
use crate::matrix::Matrix;

#[derive(Clone, Copy)]
pub enum Activation {
    Sigmoid,
    Relu,
}

pub struct Layer {
    pub weights: Matrix,
    pub biases: Matrix,
    pub activation: fn(f32) -> f32,
    pub derivative: fn(f32) -> f32,
}

impl Layer {
    pub fn weights(&self) -> &Matrix {
        &self.weights
    }
    pub fn biases(&self) -> &Matrix {
        &self.biases
    }

    // Constructor para entrenamiento
    pub fn new(inputs: usize, neurons: usize, activation_type: Activation) -> Layer {
        let (act, der): (fn(f32) -> f32, fn(f32) -> f32) = match activation_type {
            Activation::Sigmoid => (sigmoid, sigmoid_prime),
            Activation::Relu => (relu, relu_prime),
        };
        Layer {
            weights: Matrix::random(inputs, neurons),
            biases: Matrix::random(1, neurons),
            activation: act,
            derivative: der,
        }
    }

    // Constructor para carga desde archivo
    pub fn from(weights: Matrix, biases: Matrix, activation_type: Activation) -> Self {
        let (act, der): (fn(f32) -> f32, fn(f32) -> f32) = match activation_type {
            Activation::Sigmoid => (sigmoid, sigmoid_prime),
            Activation::Relu => (relu, relu_prime),
        };
        Self {
            weights,
            biases,
            activation: act,
            derivative: der,
        }
    }

    pub fn feed_forward(&self, inputs: &Matrix) -> Matrix {
        (inputs.mul(&self.weights))
            .sum(&self.biases)
            .map(self.activation)
    }
}
