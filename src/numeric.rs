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

impl Numeric for i16 {
    fn mul_idnt() -> Self {
        1_i16
    }
    fn add_idnt() -> Self {
        0_i16
    }
}

impl Numeric for i8 {
    fn mul_idnt() -> Self {
        1_i8
    }
    fn add_idnt() -> Self {
        0_i8
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

impl Numeric for u16 {
    fn mul_idnt() -> Self {
        1_u16
    }
    fn add_idnt() -> Self {
        0_u16
    }
}

impl Numeric for u8 {
    fn mul_idnt() -> Self {
        1_u8
    }
    fn add_idnt() -> Self {
        0_u8
    }
}
