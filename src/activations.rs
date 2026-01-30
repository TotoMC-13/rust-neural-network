pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Pasamos el resultado de sigmoid como arg para evitar calcularla de nuevo
pub fn sigmoid_prime(y: f32) -> f32 {
    y * (1.0 - y)
}