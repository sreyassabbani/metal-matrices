use matrix_test::matrix::Matrix;

fn main() {
    let lhs = Matrix::<f32, 2, 3>::from([[1.0, 2.0, 0.0], [2.0, 3.0, 3.0]]);
    let rhs = Matrix::<f32, 3, 2>::from([[4.0, 2.0], [4.0, -1.0], [3.0, -2.0]]);

    let product = lhs.multiply(&rhs);
    println!("CPU product:\n{product:?}");

    let transpose = product.transpose();
    println!("Transpose:\n{transpose:?}");

    let first_column = product
        .get_vector(0)
        .expect("column 0 should exist for a 2x2 matrix");
    println!("First column: {first_column:?}");
}
