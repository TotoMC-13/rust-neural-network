use crate::layer::Layer;
use crate::matrix::Matrix;
use std::time::Instant;

pub struct Network {
    layers: Vec<Layer>,
    learning_rate: f32,
}

fn create_batches(data: &[Matrix], batch_size: usize) -> Vec<Matrix> {
    let cols = data[0].cols();
    data.chunks(batch_size)
        .map(|chunk| {
            let mut items: Vec<f32> = Vec::with_capacity(chunk.len() * cols);
            for matrix in chunk {
                items.extend_from_slice(matrix.items());
            }
            Matrix::from(items, chunk.len(), cols)
        })
        .collect()
}

impl Network {
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn new(layers: Vec<Layer>, learning_rate: f32) -> Network {
        Network {
            layers,
            learning_rate,
        }
    }

    pub fn from(layers: Vec<Layer>) -> Self {
        Self {
            layers,
            learning_rate: 0.5, // Valor por defecto
        }
    }

    // Devuelve los input procesados, desde el primero al ultimo
    // Input es el que entra a la primer layer (por eso es, la entrada)
    pub fn feed_forward(&self, inputs: &Matrix) -> Vec<Matrix> {
        let mut res: Vec<Matrix> = Vec::with_capacity(self.layers.len());

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_input = if i == 0 { inputs } else { res.last().unwrap() };
            res.push(layer.feed_forward(layer_input));
        }

        res
    }

    /*
        Calculamos el primer error, calculamos hacia donde debo moverme con el gradiente,
        calculo cuanto debo mover mis pesos y biases, calculo el error del que estaba
        antes que yo para pasarselo (con el gradiente sin LR), ahora si, me actualizo y
        despues cambio el error viejo para que lo use el que sigue.
    */

    pub fn back_propagate(
        &mut self,
        outputs: &[Matrix],
        targets: &Matrix,
        inputs: &Matrix,
        batch_size: usize,
    ) {
        let lr = self.learning_rate / batch_size as f32;
        let mut error = targets.sub(outputs.last().unwrap());
        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let gradients = outputs[i].map_and_dot_mul(layer.derivative, &error);

            let layer_input = if i == 0 { inputs } else { &outputs[i - 1] };

            let delta_weights = layer_input.mul_transpose_left(&gradients, lr);

            let delta_biases = gradients.sum_rows().scalar_mul(lr);

            let next_error = gradients.mul_transpose_right(&layer.weights);

            layer.weights.add_mut(&delta_weights);
            layer.biases.add_mut(&delta_biases);
            error = next_error;
        }
    }

    /*
        inputs: datos de entrenamiento
        targets: output deseado, es decir, objetivos
        epoch: epoca, cuantas "vueltas" le vamos a dar al set de datos
        learning_rate: step size de n (para el gradient descent, es decir, el "tamaño del paso")
    */
    pub fn train(
        &mut self,
        inputs: &[Matrix],
        targets: &[Matrix],
        epochs: usize,
        silence_training: bool,
        batch_size: usize,
    ) {
        let total_start = Instant::now();
        let batched_inputs = create_batches(inputs, batch_size);
        let batched_targets = create_batches(targets, batch_size);

        println!(
            "Starting training with {} samples for {} epochs...",
            inputs.len(),
            epochs
        );

        for epoch in 0..epochs {
            let epoch_start = Instant::now();

            if !silence_training {
                println!("--- Epoch {}/{} ---", epoch, epochs);
            }

            for (i, (batch_input, batch_target)) in batched_inputs
                .iter()
                .zip(batched_targets.iter())
                .enumerate()
            {
                let outputs = self.feed_forward(batch_input);
                self.back_propagate(&outputs, batch_target, batch_input, batch_size);
                if (i + 1) % 500 == 0 {
                    let tiempo_parcial = epoch_start.elapsed();
                    println!(
                        "   -> Batch {}/{} ({:.2?})",
                        i + 1,
                        batched_inputs.len(),
                        tiempo_parcial
                    );
                }
            }

            if !silence_training {
                println!(
                    "> Epoch {} finished in {:.2?}",
                    epoch,
                    epoch_start.elapsed()
                );
            }
        }
        println!("Total training finished in {:.2?}", total_start.elapsed());
    }
}
