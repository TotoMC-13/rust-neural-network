use super::*;

#[test]
fn test_matrix_creation_and_get() {
    let items = vec![1.0, 2.0, 3.0, 4.0];
    let m = Matrix::from(items, 2, 2);
    
    assert_eq!(m.rows, 2);
    assert_eq!(m.cols, 2);
    assert_eq!(m.get(0, 0), 1.0);
    assert_eq!(m.get(0, 1), 2.0);
    assert_eq!(m.get(1, 0), 3.0);
    assert_eq!(m.get(1, 1), 4.0);
}

#[test]
fn test_sum() {
    let m1 = Matrix::from(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let m2 = Matrix::from(vec![4.0, 3.0, 2.0, 1.0], 2, 2);
    let sum = m1.sum(&m2);
    
    assert_eq!(sum.get(0, 0), 5.0);
    assert_eq!(sum.get(0, 1), 5.0);
    assert_eq!(sum.get(1, 0), 5.0);
    assert_eq!(sum.get(1, 1), 5.0);
}

#[test]
#[should_panic(expected = "Invalid matrix sum, dimensions must be equal")]
fn test_sum_invalid_dimensions() {
    let m1 = Matrix::from(vec![1.0, 2.0], 1, 2);
    let m2 = Matrix::from(vec![1.0, 2.0], 2, 1);
    m1.sum(&m2);
}

#[test]
fn test_mul() {
    // 1 2  x  5 6  =  (1*5 + 2*7) (1*6 + 2*8)  =  19 22
    // 3 4     7 8     (3*5 + 4*7) (3*6 + 4*8)     43 50
    let m1 = Matrix::from(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let m2 = Matrix::from(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let prod = m1.mul(&m2);

    assert_eq!(prod.rows, 2);
    assert_eq!(prod.cols, 2);
    assert_eq!(prod.get(0, 0), 19.0);
    assert_eq!(prod.get(0, 1), 22.0);
    assert_eq!(prod.get(1, 0), 43.0);
    assert_eq!(prod.get(1, 1), 50.0);
}

#[test]
#[should_panic(expected = "Invalid matrix multiplication")]
fn test_mul_invalid_dimensions() {
    let m1 = Matrix::from(vec![1.0, 2.0], 1, 2);
    let m2 = Matrix::from(vec![1.0, 2.0], 1, 2);
    m1.mul(&m2);
}

#[test]
fn test_transpose() {
    // 1 2 3  ->  1 4
    // 4 5 6      2 5
    //            3 6
    let m = Matrix::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let t = m.transpose();

    assert_eq!(t.rows, 3);
    assert_eq!(t.cols, 2);
    assert_eq!(t.get(0, 0), 1.0);
    assert_eq!(t.get(0, 1), 4.0);
    assert_eq!(t.get(1, 0), 2.0);
    assert_eq!(t.get(2, 0), 3.0);
}

#[test]
fn test_map() {
    let m = Matrix::from(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let mapped = m.map(|x| x * 2.0);
    
    assert_eq!(mapped.get(0, 0), 2.0);
    assert_eq!(mapped.get(0, 1), 4.0);
    assert_eq!(mapped.get(1, 0), 6.0);
    assert_eq!(mapped.get(1, 1), 8.0);
}
