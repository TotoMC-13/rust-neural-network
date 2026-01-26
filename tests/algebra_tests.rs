use neural_net::matrix::Matrix;


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
fn test_matrix_single_element() {
    let m = Matrix::from(vec![5.0], 1, 1);
    assert_eq!(m.rows, 1);
    assert_eq!(m.cols, 1);
    assert_eq!(m.get(0, 0), 5.0);
}

#[test]
fn test_matrix_row_vector() {
    let m = Matrix::from(vec![1.0, 2.0, 3.0], 1, 3);
    assert_eq!(m.rows, 1);
    assert_eq!(m.cols, 3);
    assert_eq!(m.get(0, 0), 1.0);
    assert_eq!(m.get(0, 1), 2.0);
    assert_eq!(m.get(0, 2), 3.0);
}

#[test]
fn test_matrix_column_vector() {
    let m = Matrix::from(vec![1.0, 2.0, 3.0], 3, 1);
    assert_eq!(m.rows, 3);
    assert_eq!(m.cols, 1);
    assert_eq!(m.get(0, 0), 1.0);
    assert_eq!(m.get(1, 0), 2.0);
    assert_eq!(m.get(2, 0), 3.0);
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
fn test_sum_with_negatives() {
    let m1 = Matrix::from(vec![1.0, -2.0, 3.0, -4.0], 2, 2);
    let m2 = Matrix::from(vec![-1.0, 2.0, -3.0, 4.0], 2, 2);
    let sum = m1.sum(&m2);
    
    assert_eq!(sum.get(0, 0), 0.0);
    assert_eq!(sum.get(0, 1), 0.0);
    assert_eq!(sum.get(1, 0), 0.0);
    assert_eq!(sum.get(1, 1), 0.0);
}

#[test]
#[should_panic(expected = "Invalid matrix sum, dimensions must be equal")]
fn test_sum_invalid_dimensions() {
    let m1 = Matrix::from(vec![1.0, 2.0], 1, 2);
    let m2 = Matrix::from(vec![1.0, 2.0], 2, 1);
    m1.sum(&m2);
}

#[test]
fn test_mul_non_square() {
    let m1 = Matrix::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let m2 = Matrix::from(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
    let prod = m1.mul(&m2);

    assert_eq!(prod.rows, 2);
    assert_eq!(prod.cols, 2);
    assert_eq!(prod.get(0, 0), 58.0);
    assert_eq!(prod.get(0, 1), 64.0);
    assert_eq!(prod.get(1, 0), 139.0);
    assert_eq!(prod.get(1, 1), 154.0);
}

#[test]
fn test_mul_identity() {
    let m = Matrix::from(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let identity = Matrix::from(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
    let prod = m.mul(&identity);

    assert_eq!(prod.get(0, 0), 1.0);
    assert_eq!(prod.get(0, 1), 2.0);
    assert_eq!(prod.get(1, 0), 3.0);
    assert_eq!(prod.get(1, 1), 4.0);
}

#[test]
fn test_transpose_square() {
    let m = Matrix::from(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let t = m.transpose();

    assert_eq!(t.rows, 2);
    assert_eq!(t.cols, 2);
    assert_eq!(t.get(0, 0), 1.0);
    assert_eq!(t.get(0, 1), 3.0);
    assert_eq!(t.get(1, 0), 2.0);
    assert_eq!(t.get(1, 1), 4.0);
}

#[test]
fn test_transpose_row_to_column() {
    let m = Matrix::from(vec![1.0, 2.0, 3.0], 1, 3);
    let t = m.transpose();

    assert_eq!(t.rows, 3);
    assert_eq!(t.cols, 1);
    assert_eq!(t.get(0, 0), 1.0);
    assert_eq!(t.get(1, 0), 2.0);
    assert_eq!(t.get(2, 0), 3.0);
}

#[test]
fn test_map_with_addition() {
    let m = Matrix::from(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let mapped = m.map(|x| x + 10.0);
    
    assert_eq!(mapped.get(0, 0), 11.0);
    assert_eq!(mapped.get(0, 1), 12.0);
    assert_eq!(mapped.get(1, 0), 13.0);
    assert_eq!(mapped.get(1, 1), 14.0);
}

#[test]
fn test_map_square() {
    let m = Matrix::from(vec![2.0, 3.0, 4.0, 5.0], 2, 2);
    let mapped = m.map(|x| x * x);
    
    assert_eq!(mapped.get(0, 0), 4.0);
    assert_eq!(mapped.get(0, 1), 9.0);
    assert_eq!(mapped.get(1, 0), 16.0);
    assert_eq!(mapped.get(1, 1), 25.0);
}

#[test]
fn test_dot_mul() {
    let m1 = Matrix::from(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let m2 = Matrix::from(vec![2.0, 3.0, 4.0, 5.0], 2, 2);
    let res = m1.dot_mul(&m2);

    assert_eq!(res.get(0, 0), 2.0);
    assert_eq!(res.get(0, 1), 6.0);
    assert_eq!(res.get(1, 0), 12.0);
    assert_eq!(res.get(1, 1), 20.0);
}

#[test]
#[should_panic(expected = "Invalid matrix dot multiplication, dimensions must be equal")]
fn test_dot_mul_invalid_dimensions() {
    let m1 = Matrix::from(vec![1.0, 2.0], 1, 2);
    let m2 = Matrix::from(vec![1.0, 2.0], 2, 1);
    m1.dot_mul(&m2);
}

#[test]
fn test_sub() {
    let m1 = Matrix::from(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let m2 = Matrix::from(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let sub = m1.sub(&m2);
    
    assert_eq!(sub.get(0, 0), 4.0);
    assert_eq!(sub.get(0, 1), 4.0);
    assert_eq!(sub.get(1, 0), 4.0);
    assert_eq!(sub.get(1, 1), 4.0);
}

#[test]
#[should_panic(expected = "Invalid matrix sub, dimensions must be equal")]
fn test_sub_invalid_dimensions() {
    let m1 = Matrix::from(vec![1.0, 2.0], 1, 2);
    let m2 = Matrix::from(vec![1.0, 2.0], 2, 1);
    m1.sub(&m2);
}

#[test]
fn test_scalar_mul() {
    let m = Matrix::from(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let res = m.scalar_mul(2.0);
    
    assert_eq!(res.get(0, 0), 2.0);
    assert_eq!(res.get(0, 1), 4.0);
    assert_eq!(res.get(1, 0), 6.0);
    assert_eq!(res.get(1, 1), 8.0);
}

#[test]
fn test_random_dimensions() {
    let m = Matrix::random(3, 4);
    assert_eq!(m.rows, 3);
    assert_eq!(m.cols, 4);
}

#[test]
fn test_random_range() {
    let m = Matrix::random(10, 10);
    for i in 0..m.rows {
        for j in 0..m.cols {
            let val = m.get(i, j);
            assert!(val >= -1.0 && val <= 1.0);
        }
    }
}

#[test]
fn test_random_with_seed() {
    let m1 = Matrix::random_with_seed(5, 5, 42);
    let m2 = Matrix::random_with_seed(5, 5, 42);
    
    for i in 0..5 {
        for j in 0..5 {
            assert_eq!(m1.get(i, j), m2.get(i, j));
        }
    }
}

#[test]
fn test_display_fmt() {
    let m = Matrix::from(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let output = format!("{}", m);
    let expected = "|  1.00   2.00  |\n|  3.00   4.00  |\n";
    assert_eq!(output, expected);
}
