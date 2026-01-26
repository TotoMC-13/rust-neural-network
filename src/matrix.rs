use std::fmt;

pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    items: Vec<f32>,
}

impl Matrix {
    // Constructor
    pub fn from(items: Vec<f32>, rows: usize, cols: usize) -> Matrix {
        assert_eq!(items.len(), rows * cols, "Invalid dimensions");

        Matrix {
            items,
            rows,
            cols,
        }
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

    // Hacemos FILAS por COLUMNAS
    pub fn mul(&self, m: &Matrix) -> Matrix {
        assert_eq!(self.cols, m.rows, "Invalid matrix multiplication, m1 rows must be equal to m2 cols");

        let mut res: Vec<f32> = vec![0.0; self.rows * m.cols];

        for i in 0..self.rows {
            for j in 0..m.cols {
                let mut sum: f32 = 0.0;

                for k in 0..self.cols {
                    sum += self.get(i, k) * m.get(k, j);
                }

                let idx: usize = (i * m.cols) + j;
                res[idx] = sum;
            }
        }

        Matrix {
            items: res,
            rows: self.rows,
            cols: m.cols,
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

    pub fn random(rows: usize, cols: usize) -> Matrix {
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;

            Self::random_with_seed(rows, cols, seed)
        }

    pub fn random_with_seed(rows: usize, cols: usize, mut seed: u64) -> Matrix {
        let mut res = vec![0.0; rows * cols];

        for i in 0..res.len() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;

            let val = (seed as f32) / (u64::MAX as f32);
            res[i] = (val * 2.0) - 1.0;
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
            write!(f, " |\n")?;
        }
        Ok(())
    }
}
