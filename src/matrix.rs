use rayon::prelude::*;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

static SEED_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    items: Vec<f32>,
}

impl Matrix {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn items(&self) -> &Vec<f32> {
        &self.items
    }

    // Constructor
    pub fn from(items: Vec<f32>, rows: usize, cols: usize) -> Matrix {
        assert_eq!(items.len(), rows * cols, "Invalid dimensions");

        Matrix { items, rows, cols }
    }

    // Devuelve el elemento [i][j]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.items[(i * self.cols) + j]
    }

    pub fn sum(&self, m: &Matrix) -> Matrix {
        let msg = String::from("Invalid matrix sum, dimensions must be equal");
        assert_eq!(self.rows, m.rows, "{}", &msg);
        assert_eq!(self.cols, m.cols, "{}", &msg);

        let mut res: Vec<f32> = vec![0.0; self.rows * self.cols];

        for i in 0..self.items.len() {
            res[i] = self.items[i] + m.items[i];
        }

        Matrix {
            items: res,
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub fn sub(&self, m: &Matrix) -> Matrix {
        let msg = String::from("Invalid matrix sub, dimensions must be equal");
        assert_eq!(self.rows, m.rows, "{}", &msg);
        assert_eq!(self.cols, m.cols, "{}", &msg);

        let mut res: Vec<f32> = vec![0.0; self.rows * self.cols];

        for i in 0..self.items.len() {
            res[i] = self.items[i] - m.items[i];
        }

        Matrix {
            items: res,
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub fn mul(&self, m2: &Matrix) -> Matrix {
        let mut res: Vec<f32> = vec![0.0; self.rows * m2.cols];
        res.par_chunks_mut(m2.cols)
            .enumerate()
            .for_each(|(i, res_row)| {
                for k in 0..self.cols {
                    let a_ik = self.get(i, k);

                    // Agarramos el slice para no usar los getter y asi hacer menos calculos
                    let b_row_start = k * m2.cols;
                    let b_row_end = b_row_start + m2.cols;
                    let b_row = &m2.items()[b_row_start..b_row_end];

                    for j in 0..m2.cols {
                        res_row[j] += a_ik * b_row[j];
                    }
                }
            });

        Matrix {
            items: res,
            rows: self.rows,
            cols: m2.cols,
        }
    }

    pub fn dot_mul(&self, m: &Matrix) -> Matrix {
        let msg = String::from("Invalid matrix dot multiplication, dimensions must be equal");
        assert_eq!(self.rows, m.rows, "{}", &msg);
        assert_eq!(self.cols, m.cols, "{}", &msg);

        let mut res: Vec<f32> = vec![0.0; self.items.len()];

        for i in 0..self.items.len() {
            res[i] = self.items[i] * m.items[i];
        }

        Matrix {
            items: res,
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub fn scalar_mul(&self, scalar: f32) -> Matrix {
        let mut res: Vec<f32> = vec![0.0; self.items.len()];

        for i in 0..self.items.len() {
            res[i] = self.items[i] * scalar
        }

        Matrix {
            items: res,
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub fn transpose(&self) -> Matrix {
        let mut res: Vec<f32> = vec![0.0; self.rows * self.cols];

        for i in 0..self.rows {
            for j in 0..self.cols {
                let idx: usize = (j * self.rows) + i;
                res[idx] = self.get(i, j);
            }
        }

        Matrix {
            items: res,
            rows: self.cols,
            cols: self.rows,
        }
    }

    pub fn map(&self, func: fn(f32) -> f32) -> Matrix {
        let items = &self.items;

        let mut res: Vec<f32> = vec![0.0; items.len()];

        for i in 0..items.len() {
            res[i] = func(items[i]);
        }

        Matrix {
            items: res,
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub fn argmax(&self) -> usize {
        let mut max_index = 0;
        let mut max_val = self.items[0];

        for (i, &val) in self.items.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_index = i;
            }
        }
        max_index
    }

    pub fn random(rows: usize, cols: usize) -> Matrix {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Le sumamos 1 al contador
        let count = SEED_COUNTER.fetch_add(1, Ordering::SeqCst);

        let unique_seed = seed.wrapping_add(count * 1_000_000);

        Self::random_with_seed(rows, cols, unique_seed)
    }

    pub fn random_with_seed(rows: usize, cols: usize, mut seed: u64) -> Matrix {
        let mut res = vec![0.0; rows * cols];
        let scale = (2.0 / rows as f32).sqrt(); // He initialization Formula

        for val_ref in &mut res {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;

            let val = (seed as f32) / (u64::MAX as f32);
            *val_ref = ((val * 2.0) - 1.0) * scale;
        }

        Matrix {
            items: res,
            rows,
            cols,
        }
    }
}

// Funcion para poder imprimir, ej: println!("{}", matriz)
impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Filas
        for i in 0..self.rows {
            write!(f, "|")?;
            // Columnas
            for j in 0..self.cols {
                write!(f, " {:>5.2} ", self.get(i, j))?;
            }
            writeln!(f, " |")?;
        }
        Ok(())
    }
}
