use crate::matrix::Matrix;
use crate::layer::{Layer};
use std::time::Instant;

pub struct Network {
    layers: Vec<Layer>,
    learning_rate: f32,
}

impl Network {
    pub fn new(layers: Vec<Layer>, learning_rate: f32) -> Network {
        Network { layers, learning_rate}
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

    /*
        Calculamos el primer error, calculamos hacia donde debo moverme con el gradiente, 
        calculo cuanto debo mover mis pesos y biases, calculo el error del que estaba 
        antes que yo para pasarselo (con el gradiente sin LR), ahora si, me actualizo y
        despues cambio el error viejo para que lo use el que sigue.
    */

    pub fn back_propagate(&mut self, outputs: Vec<Matrix>, targets: Matrix) {
        let mut error = targets.sub(outputs.last().unwrap());

        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            // Calculamos el gradiente (sin learning rate todavia)
            let gradients = outputs[i + 1].map(layer.derivative).dot_mul(&error);

            // Calculamos los deltas de pesos (Aca si aplicamos LR)
            let delta_weights = outputs[i].transpose().mul(&gradients).scalar_mul(self.learning_rate);
            
            // También necesitamos delta para bias (Gradients * LR)
            let delta_biases = gradients.scalar_mul(self.learning_rate);

            // Calculamos el error para la siguiente vuelta (Usando gradiente PURO)
            let next_error = gradients.mul(&layer.weights.transpose());

            // Actualizamos la capa
            layer.weights = layer.weights.sum(&delta_weights);
            layer.biases = layer.biases.sum(&delta_biases);

            // Pasamos al siguiente
            error = next_error;
        }
}

    /*
        inputs: datos de entrenamiento
        targets: output deseado, es decir, objetivos
        epoch: epoca, cuantas "vueltas" le vamos a dar al set de datos
        learning_rate: step size de n (para el gradient descent)
    */
    pub fn train(&mut self, inputs: &Vec<Matrix>, targets: &Vec<Matrix>, epochs: usize) {
        let total_start = Instant::now();
        
        println!("Iniciando entrenamiento con {} datos por {} epochs...", inputs.len(), epochs);
        
        for epoch in 0..epochs {
            let epoch_start = Instant::now();
            println!("--- Epoch {}/{} ---", epoch, epochs);

            for (i, input) in inputs.iter().cloned().enumerate() {
                let outputs = self.feed_forward(input);
                self.back_propagate(outputs, targets[i].clone());

                if (i + 1) % 10_000 == 0 {
                    let tiempo_parcial = epoch_start.elapsed();
                    println!("   -> Procesadas {}/{} imagenes ({:.2?})", i + 1, inputs.len(), tiempo_parcial);
                }
            }
            println!("> Epoch {} finalizada en {:.2?}", epoch, epoch_start.elapsed());
        }
        println!("Entrenamiento total finalizado en {:.2?}", total_start.elapsed());
    }
}   
