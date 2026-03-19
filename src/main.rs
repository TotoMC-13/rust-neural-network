use neural_net::layer::{Activation, Layer};
use neural_net::matrix::Matrix;
use neural_net::mnist::{load_data, load_labels};
use neural_net::network::Network;

const DEMO: u8 = 2;

fn main() {
    if DEMO == 2 {
        images_demo(true);
    } else if DEMO == 1 {
        images_demo(false);
    } else if DEMO == 0 {
        xor_demo();
    }
}

fn images_demo(load: bool) {
    let training_data = load_data("data/train-images.idx3-ubyte").unwrap();
    let training_labels = load_labels("data/train-labels.idx1-ubyte").unwrap();
    let testing_data = load_data("data/testing-images.idx3-ubyte").unwrap();
    let testing_labels = load_labels("data/testing-labels.idx1-ubyte").unwrap();

    let net = if load {
        neural_net::io::load_network("images_model.nn").expect("Error al cargar")
    } else {
        let layers = vec![
            Layer::new(784, 64, Activation::Sigmoid),
            Layer::new(64, 10, Activation::Sigmoid),
        ];
        let mut n = Network::new(layers, 0.5);
        n.train(&training_data, &training_labels, 5);
        n
    };

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

    println!(
        "Precision: {}/{} ({:.2}%)",
        aciertos,
        total,
        (aciertos as f32 / total as f32) * 100.0
    );

    if !load {
        neural_net::io::save_network(&net, "images_model.nn").expect("Error al guardar");
    }
}

fn xor_demo() {
    let inputs = vec![
        Matrix::from(vec![0.0, 0.0], 1, 2),
        Matrix::from(vec![0.0, 1.0], 1, 2),
        Matrix::from(vec![1.0, 0.0], 1, 2),
        Matrix::from(vec![1.0, 1.0], 1, 2),
    ];

    let targets = vec![
        Matrix::from(vec![0.0], 1, 1),
        Matrix::from(vec![1.0], 1, 1),
        Matrix::from(vec![1.0], 1, 1),
        Matrix::from(vec![0.0], 1, 1),
    ];

    let layers = vec![
        Layer::new(2, 3, Activation::Sigmoid),
        Layer::new(3, 1, Activation::Sigmoid),
    ];

    let mut network = Network::new(layers, 0.5);

    println!("Antes de Entrenar: ");
    for i in 0..inputs.len() {
        let output = network.feed_forward(inputs[i].clone());
        println!(
            "Input: {} -> Output: {:.3}",
            inputs[i],
            output.last().unwrap().get(0, 0)
        );
    }

    println!("\nEntrenando...");
    network.train(&inputs, &targets, 2000);

    println!("\nDespués de Entrenar: ");
    for i in 0..inputs.len() {
        let output = network.feed_forward(inputs[i].clone());
        println!(
            "Input: {} -> Output: {:.3}",
            inputs[i],
            output.last().unwrap().get(0, 0)
        );
    }
}
