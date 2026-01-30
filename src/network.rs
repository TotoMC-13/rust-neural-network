use crate::matrix::Matrix;
use crate::layer::{self, Layer};

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Network {
        Network { layers }
    }

    // Devuelve los input procesados, desde el primero al ultimo
    // Input es el que entra a la primer layer (por eso es, la entrada)
    pub fn feed_forward(&self, mut inputs: Matrix) -> Vec<Matrix> {
        let mut res: Vec<Matrix> = Vec::with_capacity(self.layers.len() + 1);
        let layers = &self.layers;

        for i in 0..layers.len() {
            res.push(inputs);
            inputs = layers[i].feed_forward(&res.last().unwrap()); 
        }

        // Agrego ultimo elemento (resultado)
        res.push(inputs);
        
        res
    }
    
    pub fn back_propagate(&mut self, outputs: Vec<Matrix>, targets: Matrix) {
    // Aquí empieza la magia
    }

    /*
    inputs: todas las layers
    targets: obetivos(?)
    epoch: epoca
    learning_rate: step size de n (para el gradient descent)
    */
    pub fn train(&mut self, inputs: &Vec<Matrix>, targets: &Vec<Matrix>, epochs: usize, learning_rate: f32) {
        todo!()
    }
}   