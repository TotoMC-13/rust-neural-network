use std::fs;
use crate::matrix::Matrix;

/*  
    Bytes
    0..4 : Magic Number
    4..8: Cantidad de Imagenes
    8..12: Filas
    12..16: Columnas
    16.. : Imagenes
*/

pub fn load_data(filename: &str) -> Result<Vec<Matrix>, Box<dyn std::error::Error + 'static>> {
    let data: Vec<u8> = fs::read(filename)?;

    // Cantidad de Imagenes (4..8)
    let num_images = u32::from_be_bytes(data[4..8].try_into()?);

    // Cargar imagenes
    
    let images_data = &data[16..];

    let chunk_size = 784; // Size de cada imagen en bytes

    let mut chunks = images_data.chunks_exact(chunk_size);

    let mut images: Vec<Matrix> = Vec::with_capacity(num_images as usize);

    while let Some(chunk) = chunks.next() {
        let mut pixel_data: Vec<f32> = Vec::with_capacity(chunk_size);

        for &pixel in chunk {
            pixel_data.push(pixel as f32 / 255.0);
        }

        let matrix_imagen = Matrix::from(pixel_data, 1, 784);

        images.push(matrix_imagen);
    }

    Ok(images)
}

/*
    Bytes
    0..4: Magic number
    4..8: Cantidad de Items
    8.. : Labels 
*/

pub fn load_labels(filename: &str) -> Result<Vec<Matrix>, Box<dyn std::error::Error + 'static>> {
    let data: Vec<u8> = fs::read(filename)?;

    // let magic_num = u32::from_be_bytes(data[0..4].try_into()?);

    let labels_data = &data[8..];
    let capacity: u32 = u32::from_be_bytes(data[4..8].try_into()?);

    let mut labels: Vec<Matrix> = Vec::with_capacity(capacity as usize);

    for &byte in labels_data {
        let mut label: Vec<f32> = vec![0.0; 10];

        label[byte as usize] = 1.0;

        let res = Matrix::from(label, 1, 10);

        labels.push(res);
    }

    Ok(labels)
}