use crate::math::algebra::Matrix;

mod math;

fn main() {
    let m: Matrix = Matrix::from(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let n: Matrix = Matrix::from(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let mxn: Matrix = m.mul(&n);
    let msn: Matrix = m.sum(&n);
    println!("{}", m);
    println!("{}", n);
    println!("{}", mxn);
    println!("{}", msn);
}
