use neural_net::matrix::Matrix;
use neural_net::network::Network;
use neural_net::layer::Layer;
use neural_net::activations::*;


fn main() {
    let m: Matrix = Matrix::random(2, 2);
    println!("Matriz random m:\n{}", m);
}