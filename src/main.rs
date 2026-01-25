use crate::math::matrix::Matrix;

mod math;

fn main() {
    let m: Matrix = Matrix::random(2, 2);
    println!("Matriz random m:\n{}", m);
}