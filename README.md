# Red neuronal en Rust "from scratch"
Una implementación completa y desde cero de una Red Neuronal Feed-Forward escrita en Rust. 
Este proyecto fue desarrollado para comprender en profundidad el funcionamiento interno del Deep Learning, 
evitando el uso de librerías externas para el álgebra lineal, los algoritmos de machine learning como para la carga/lectura de archivos.

## Características
- Motor Matemático Propio: Una implementación a medida de la estructura Matrix que soporta productos punto, multiplicaciones de matrices, operaciones escalares, transposiciones e inicializaciones aleatorias con semillas.

- Core de Deep Learning: Implementación completa del feed-forward y del algoritmo de backpropagation para el entrenamiento de la red.

- Funciones de Activación: Soporte para las funciones Sigmoid y ReLU, junto con sus respectivas derivadas para el cálculo de gradientes.

- Cargado y guardado de datos: Un módulo de entrada/salida (io.rs) dedicado para guardar y cargar sin esfuerzo los modelos entrenados utilizando un formato binario propio en Little Endian (archivos .nn).

- Integración con MNIST: Incluye un parser personalizado para leer y procesar directamente los archivos binarios crudos de la base de datos de dígitos manuscritos MNIST.

## Uso
```
git clone https://github.com/TotoMC-13/rust-neural-network.git
cd rust-neural-network
cargo run --release
```

- Problema XOR: Una demostración ligera de la red aprendiendo la compuerta lógica no lineal XOR a partir de entradas básicas.

- Entrenamiento MNIST: Entrena la red con el dataset MNIST, mostrando el progreso de cada epoch y evaluando su precisión final.

- Carga MNIST: Carga un modelo .nn previamente entrenado para evaluar rápidamente imágenes de MNIST sin necesidad de volver a entrenar la red desde el principio.

## Estructura
- matrix.rs: Maneja todas las operaciones de álgebra lineal.

- network.rs y layer.rs: Definen la arquitectura de la red neuronal, manejando el feed-forward y las actualizaciones de los biases durante la backpropagation.

- io.rs: Gestiona la carga y guardado de las estructuras de la red en el formato binario personalizado (.nn).

- mnist.rs: Decodifica los números mágicos y las estructuras de bytes del dataset MNIST para transformarlos en matrices utilizables.
