use neural_net::layer::{Activation, Layer};
use neural_net::matrix::Matrix;
use neural_net::mnist::{load_data, load_labels};
use neural_net::network::Network;
use std::env;
use std::io::{self, Write};

fn main() {
    let args: Vec<String> = env::args().collect();

    let demo = if args.len() > 1 {
        args[1].parse().unwrap_or(0)
    } else {
        pedir_entrada("Select Demo (0: XOR, 1: MNIST Train, 2: MNIST Load): ")
    };
    let (activation, lr) = if demo < 2 {
        let act_choice = if args.len() > 2 {
            args[2].parse().unwrap_or(0)
        } else {
            pedir_entrada("Select Activation (0: Sigmoid, 1: Relu): ")
        };

        match act_choice {
            1 => (Activation::Relu, 0.01),
            _ => (Activation::Sigmoid, 0.5),
        }
    } else {
        (Activation::Sigmoid, 0.0)
    };

    match demo {
        0 => xor_demo(activation),
        1 => images_demo(false, activation, lr),
        2 => images_demo(true, activation, lr),
        _ => println!("Invalid option"),
    }
}

fn pedir_entrada(mensaje: &str) -> u8 {
    print!("{}", mensaje);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().parse().unwrap_or(0)
}

fn images_demo(load: bool, activation: Activation, lr: f32) {
    let training_data = load_data("data/train-images.idx3-ubyte").unwrap();
    let training_labels = load_labels("data/train-labels.idx1-ubyte").unwrap();
    let testing_data = load_data("data/testing-images.idx3-ubyte").unwrap();
    let testing_labels = load_labels("data/testing-labels.idx1-ubyte").unwrap();

    let mut train_duration = 0.0;
    let epochs = 5;
    let net = if load {
        neural_net::io::load_network("images_model.nn").expect("Error loading network")
    } else {
        let layers = vec![
            Layer::new(784, 256, activation),
            Layer::new(256, 64, activation),
            Layer::new(64, 10, Activation::Sigmoid),
        ];
        let mut n = Network::new(layers, lr);
        let train_start = std::time::Instant::now();
        n.train(&training_data, &training_labels, epochs, false);
        train_duration = train_start.elapsed().as_secs_f32();
        n
    };

    println!("Evaluating");
    let mut aciertos = 0;
    let total = testing_data.len();

    for (input, target) in testing_data.iter().zip(testing_labels.iter()) {
        let outputs = net.feed_forward(input.clone());
        let prediction = outputs.last().unwrap();

        if prediction.argmax() == target.argmax() {
            aciertos += 1;
        }
    }

    let precision = (aciertos as f32 / total as f32) * 100.0;
    println!("Accuracy: {}/{} ({:.2}%)", aciertos, total, precision);

    if !load {
        neural_net::io::log_performance(
            &net,
            epochs,
            train_duration,
            precision,
            activation,
            "MNIST",
        )
        .expect("Error registering benchmark");
        neural_net::io::save_network(&net, "images_model.nn", activation)
            .expect("Error saving network");
    }
}

fn xor_demo(activation: Activation) {
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
        Layer::new(2, 3, activation),
        Layer::new(3, 1, Activation::Sigmoid),
    ];

    let mut network = Network::new(layers, 0.2);

    println!("Before Training: ");
    for input in &inputs {
        let output = network.feed_forward(input.clone());
        println!(
            "Input: {} -> Output: {:.3}",
            input,
            output.last().unwrap().get(0, 0)
        );
    }

    println!("\nTraining...");
    let epochs = 2000;
    let train_start = std::time::Instant::now();
    network.train(&inputs, &targets, epochs, true);
    let train_duration = train_start.elapsed().as_secs_f32();

    println!("\nAfter Training: ");
    let mut aciertos = 0;
    for i in 0..inputs.len() {
        let output = network.feed_forward(inputs[i].clone());
        let pred = output.last().unwrap().get(0, 0);
        let target = targets[i].get(0, 0);
        if (pred >= 0.5 && target >= 0.5) || (pred < 0.5 && target < 0.5) {
            aciertos += 1;
        }
        println!("Input: {} -> Output: {:.3}", inputs[i], pred);
    }
    let precision = (aciertos as f32 / inputs.len() as f32) * 100.0;

    neural_net::io::log_performance(
        &network,
        epochs,
        train_duration,
        precision,
        activation,
        "XOR",
    )
    .expect("Error registering benchmark");
}
