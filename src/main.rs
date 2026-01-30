use neural_net::matrix::Matrix;
use neural_net::network::Network;
use neural_net::layer::Layer;
use neural_net::activations::{sigmoid, sigmoid_prime};

fn main() {
    // Datos de entrenamiento para resolver XOR
    // Dimensiones 1x2 (1 fila, 2 columnas)
    let inputs = vec![
        Matrix::from(vec![0.0, 0.0], 1, 2),
        Matrix::from(vec![0.0, 1.0], 1, 2),
        Matrix::from(vec![1.0, 0.0], 1, 2),
        Matrix::from(vec![1.0, 1.0], 1, 2),
    ];

    // Targets: Dimensiones 1x1
    let targets = vec![
        Matrix::from(vec![0.0], 1, 1),
        Matrix::from(vec![1.0], 1, 1),
        Matrix::from(vec![1.0], 1, 1),
        Matrix::from(vec![0.0], 1, 1),
    ];

    // Creamos la red
    // 2 entradas -> 3 ocultas -> 1 salida
    let layers = vec![
        Layer::new(2, 3, sigmoid, sigmoid_prime), 
        Layer::new(3, 1, sigmoid, sigmoid_prime), 
    ];

    let mut network = Network::new(layers, 0.5); 

    // Probar antes de entrenar
    println!("Antes de Entrenar: ");
    for i in 0..inputs.len() {
        let output = network.feed_forward(inputs[i].clone());
        println!("Input: {} -> Output: {:.4}", inputs[i], output.last().unwrap().get(0, 0));
    }

    // 4. Entrenar
    println!("\nEntrenando...");
    network.train(&inputs, &targets, 200000);

    // 5. Probar despues de entrenar
    println!("\nDespués de Entrenar: ");
    for i in 0..inputs.len() {
        let output = network.feed_forward(inputs[i].clone());
        println!("Input: {} -> Output: {:.4}", inputs[i], output.last().unwrap().get(0, 0));
    }
}