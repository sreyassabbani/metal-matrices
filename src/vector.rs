use crate::numeric::Numeric;

/// A column vector struct, where
/// - `const M: usize` = number of rows
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Vector<T: Numeric, const M: usize> {
    entries: Box<[T; M]>,
}

// NOTE: make macro to let doing more impls like From<Box<[T; M]>> less terse
/// Implementation of [`From`] trait to build a [`Vector<T, M>`] from `[T; M]`
impl<T, const M: usize> From<[T; M]> for Vector<T, M>
where
    T: Numeric,
{
    /// Convert from a `[T; M]` to a [`Vector<T, M>`].
    /// Build a vector from given entries
    fn from(value: [T; M]) -> Self {
        Self {
            entries: Box::new(value),
        }
    }
}

impl<T, const M: usize> AsRef<[T; M]> for Vector<T, M>
where
    T: Numeric,
{
    /// Gives a reference to a contiguous array of entries (stored on the heap).
    #[inline]
    fn as_ref(&self) -> &[T; M] {
        // Box impls AsRef<>
        &self.entries
    }
}

/// Implement addition for [`Vector`]
impl<T: Numeric, const N: usize> std::ops::Add for Vector<T, N> {
    type Output = Vector<T, N>;

    /// Pairwise addition through each entry of [`Vector<T, N>`]
    fn add(self, other: Self) -> Self::Output {
        let mut entries = [T::add_idnt(); N];
        for (i, entry) in entries.iter_mut().enumerate() {
            *entry = self.entries[i] + other.entries[i];
        }
        Vector::from(entries)
    }
}

// Semantically no. It would allow for weird things like passing arrays to the get fn of a HashMap
// use std::borrow::Borrow;
// impl<T, const N: usize> Borrow<[T; N]> for Vector<T, N> {
//     fn borrow(&self) -> &[T; N] {
//         &self.entries
//     }
// }
