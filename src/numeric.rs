use std::ops;

/// Internal type for dealing with general matrices.
pub trait Numeric:
    ops::Add<Output = Self>
    + ops::Sub<Output = Self>
    + ops::SubAssign
    + ops::AddAssign
    + ops::Mul<Output = Self>
    + Default
    + std::fmt::Debug
    + Copy
{
    fn mul_idnt() -> Self;
    fn add_idnt() -> Self;
}

impl Numeric for f64 {
    fn mul_idnt() -> Self {
        1_f64
    }
    fn add_idnt() -> Self {
        0_f64
    }
}

impl Numeric for f32 {
    fn mul_idnt() -> Self {
        1_f32
    }
    fn add_idnt() -> Self {
        0_f32
    }
}

impl Numeric for i64 {
    fn mul_idnt() -> Self {
        1_i64
    }
    fn add_idnt() -> Self {
        0_i64
    }
}

impl Numeric for i32 {
    fn mul_idnt() -> Self {
        1_i32
    }
    fn add_idnt() -> Self {
        0_i32
    }
}

impl Numeric for u64 {
    fn mul_idnt() -> Self {
        1_u64
    }
    fn add_idnt() -> Self {
        0_u64
    }
}

impl Numeric for u32 {
    fn mul_idnt() -> Self {
        1_u32
    }
    fn add_idnt() -> Self {
        0_u32
    }
}
