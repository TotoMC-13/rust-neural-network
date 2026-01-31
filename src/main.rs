use neural_net::matrix::Matrix;
use neural_net::network::Network;
use neural_net::layer::Layer;
use neural_net::activations::{sigmoid, sigmoid_prime};
use neural_net::mnist::{load_data, load_labels};

const DEMO: u8 = 1; // 0: xor demo, 1: images demo

fn main() {

    if DEMO == 1 {
        images_demo();
    } else if DEMO == 0 {
        xor_demo();
    }
}

fn images_demo() {
    let training_data = load_data("data/train-images.idx3-ubyte").unwrap();
    let training_labels = load_labels("data/train-labels.idx1-ubyte").unwrap();
    let testing_data = load_data("data/testing-images.idx3-ubyte").unwrap();
    let testing_labels = load_labels("data/testing-labels.idx1-ubyte").unwrap();

    let layers =  vec![
        Layer::new(784, 64, sigmoid, sigmoid_prime),
        Layer::new(64, 10, sigmoid, sigmoid_prime),
    ];

    let learning_rate = 0.5;

    let mut net = Network::new(layers, learning_rate);

    net.train(&training_data, &training_labels, 5);

    println!("Evaluando");
    let mut aciertos = 0;
    let total = testing_data.len();

    for (input, target) in testing_data.iter().zip(testing_labels.iter()) {
        
        let outputs = net.feed_forward(input.clone());
        let prediction = outputs.last().unwrap();

        if prediction.argmax() == target.argmax() {
            aciertos += 1;
        }
    }

    println!("Precision: {}/{} ({:.2}%)", 
        aciertos, total, (aciertos as f32 / total as f32) * 100.0);
}

fn xor_demo() {
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
        println!("Input: {} -> Output: {:.3}", inputs[i], output.last().unwrap().get(0, 0));
    }

    // 4. Entrenar
    println!("\nEntrenando...");
    network.train(&inputs, &targets, 20000);

    // 5. Probar despues de entrenar
    println!("\nDespués de Entrenar: ");
    for i in 0..inputs.len() {
        let output = network.feed_forward(inputs[i].clone());
        println!("Input: {} -> Output: {:.3}", inputs[i], output.last().unwrap().get(0, 0));
    }
}