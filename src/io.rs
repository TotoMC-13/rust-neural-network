use crate::layer::{Activation, Layer};
use crate::matrix::Matrix;
use crate::network::Network;
use std::fs::{self, File};
use std::io::Write;

// Definimos un alias para no escribir tanto.
// Esto acepta cualquier error (IO, Slice, etc.)
type DynResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/* ESTRUCTURA DEL ARCHIVO BINARIO (.nn )
    Utiliza Little Endian (LE).

    Byte 0             : Cantidad de capas (u8) - 1 byte
    Byte 1             : Tipo de funcion de activacion (0: Sigmoid, 1: Relu) (u8) - 1 byte

    REPETIR POR CADA CAPA:
    ---------------------------------------------------------------
    PESOS (Weights):
    Byte +0..4         : Número de Filas (u32)
    Byte +4..8         : Número de Columnas (u32)
    Byte +8..N         : Datos de la Matriz (f32 * filas * columnas)
                         Donde N = 8 + (filas * columnas * 4)

    SESGOS (Biases):
    Byte N..N+4        : Número de Filas (u32)
    Byte N+4..N+8      : Número de Columnas (u32)
    Byte N+8..M        : Datos de la Matriz (f32 * filas * columnas)
                         Donde M = N + 8 + (filas * columnas * 4)
    ---------------------------------------------------------------
*/

pub fn save_network(net: &Network, filename: &str, activation: Activation) -> DynResult<()> {
    let mut file = File::create(filename)?;

    let layer_count: u8 = net.layers().len() as u8;
    let activation_type: u8 = match activation {
        Activation::Sigmoid => 0,
        Activation::Relu => 1,
    };
    
    file.write_all(&layer_count.to_le_bytes())?;
    file.write_all(&activation_type.to_le_bytes())?;

    for layer in net.layers() {
        // Guardamos los pesos
        let w = layer.weights();
        file.write_all(&(w.rows() as u32).to_le_bytes())?;
        file.write_all(&(w.cols() as u32).to_le_bytes())?;

        for &val in w.items() {
            file.write_all(&val.to_le_bytes())?;
        }

        // Guardamos los biases
        let b = layer.biases();
        file.write_all(&(b.rows() as u32).to_le_bytes())?;
        file.write_all(&(b.cols() as u32).to_le_bytes())?;

        for &val in b.items() {
            file.write_all(&val.to_le_bytes())?;
        }
    }

    Ok(())
}

pub fn load_network(filename: &str) -> DynResult<Network> {
    let data = fs::read(filename)?;
    let mut rest = &data[..];

    let layer_count = rest[0];
    let activation: u8 = rest[1];
    let act_name: &str = match activation {
        0 => "Sigmoid",
        1 => "Relu",
        _ => return Err("Tipo de activación inválido".into()),
    };

    println!("Cargando red con {} capas y activación {}", layer_count, act_name);

    rest = &rest[2..];

    let mut layers = Vec::with_capacity(layer_count as usize);

    for i in 0..layer_count {
        let (w, next) = load_matrix_from_slice(rest)?;
        rest = next;
        let (b, next) = load_matrix_from_slice(rest)?;
        rest = next;
        
        
        let act = if i == layer_count - 1 {
        Activation::Sigmoid
        } else {
            match activation {
                0 => Activation::Sigmoid,
                1 => Activation::Relu,
                _ => return Err("ERROR: Funcion de activacion invalida".into()),
            }
        };

        layers.push(Layer::from(w, b, act));
    }


    Ok(Network::from(layers))
}

fn load_matrix_from_slice(slice: &[u8]) -> DynResult<(Matrix, &[u8])> {
    let rows = u32::from_le_bytes(slice[0..4].try_into()?) as usize;
    let cols = u32::from_le_bytes(slice[4..8].try_into()?) as usize;
    let mut items = Vec::with_capacity(rows * cols);
    let mut curr = 8;
    for _ in 0..(rows * cols) {
        items.push(f32::from_le_bytes(slice[curr..curr + 4].try_into()?));
        curr += 4;
    }
    Ok((Matrix::from(items, rows, cols), &slice[curr..]))
}
