use matrix_test::{matrix::Matrix, metal::MetalRuntime};

fn main() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("Metal runtime unavailable: {err}");
            return;
        }
    };

    let lhs = Matrix::<f32, 3, 3>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let rhs = Matrix::<f32, 3, 3>::from([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);

    let cpu_product = lhs.multiply(&rhs);
    println!("CPU product:\n{cpu_product:?}");

    match runtime.multiply(&lhs, &rhs) {
        Ok(gpu_product) => {
            println!("GPU product:\n{gpu_product:?}");
            println!("GPU matches CPU: {}", gpu_product == cpu_product);
        }
        Err(err) => eprintln!("GPU multiply failed: {err}"),
    }
}
