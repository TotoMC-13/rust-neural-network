use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neural_net::matrix::Matrix;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Duration;

fn old_mul(m1: &Matrix, m2: &Matrix) -> Vec<f32> {
    let mut res: Vec<f32> = vec![0.0; m1.rows * m2.cols];
    for i in 0..m1.rows {
        for k in 0..m1.cols {
            let a_ik = m1.get(i, k);
            for j in 0..m2.cols {
                let idx = (i * m2.cols) + j;
                res[idx] += a_ik * m2.get(k, j);
            }
        }
    }
    res
}

fn old_rayon_mul(m1: &Matrix, m2: &Matrix) -> Vec<f32> {
    let mut res: Vec<f32> = vec![0.0; m1.rows * m2.cols];
    res.par_chunks_mut(m2.cols)
        .enumerate()
        .for_each(|(i, row)| {
            for k in 0..m1.cols {
                let a_ik = m1.get(i, k);
                for j in 0..m2.cols {
                    row[j] += a_ik * m2.get(k, j);
                }
            }
        });
    res
}

fn new_rayon_mul(m1: &Matrix, m2: &Matrix) -> Vec<f32> {
    let mut res: Vec<f32> = vec![0.0; m1.rows * m2.cols];
    res.par_chunks_mut(m2.cols)
        .enumerate()
        .for_each(|(i, res_row)| {
            for k in 0..m1.cols {
                let a_ik = m1.get(i, k);

                let b_row_start = k * m2.cols;
                let b_row_end = b_row_start + m2.cols;
                let b_row = &m2.items()[b_row_start..b_row_end];

                for j in 0..m2.cols {
                    res_row[j] += a_ik * b_row[j];
                }
            }
        });
    res
}

fn bench_matrix_multiplication(c: &mut Criterion) {
    let sizes = [128, 512, 1024];

    for size in sizes.iter() {
        let mut group = c.benchmark_group(format!("Matrix_Mul_{}", size));

        let m1 = Matrix::random(*size, *size);
        let m2 = Matrix::random(*size, *size);

        // Multiplicacion Secuencial (i-k-j)
        group.bench_with_input(
            BenchmarkId::new("Secuencial (i-k-j)", size),
            size,
            |b, _| b.iter(|| black_box(old_mul(&m1, black_box(&m2)))),
        );

        // Multiplicacion Rayon Vieja
        group.bench_with_input(BenchmarkId::new("Rayon Viejo", size), size, |b, _| {
            b.iter(|| black_box(old_rayon_mul(&m1, black_box(&m2))))
        });

        // Multiplicacion Rayon Nueva
        group.bench_with_input(
            BenchmarkId::new("Rayon Nuevo (Optimizado)", size),
            size,
            |b, _| b.iter(|| black_box(new_rayon_mul(&m1, black_box(&m2)))),
        );

        group.finish();
    }
}

fn custom_criterion() -> Criterion {
    Criterion::default().measurement_time(Duration::from_secs(20))
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = bench_matrix_multiplication
}
criterion_main!(benches);
