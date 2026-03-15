use crate::numeric::Numeric;
use std::mem::MaybeUninit;

/// A column vector struct, where
/// - `const M: usize` = number of rows
#[derive(Debug, PartialEq)]
pub struct Vector<T: Numeric, const M: usize> {
    entries: Box<[T; M]>,
}

impl<T, const M: usize> From<[T; M]> for Vector<T, M>
where
    T: Numeric,
{
    fn from(value: [T; M]) -> Self {
        Self {
            entries: Box::new(value),
        }
    }
}

impl<T: Numeric, const M: usize> Vector<T, M> {
    pub fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        let mut uninit_entries: Box<MaybeUninit<[T; M]>> = Box::new_uninit();
        let base_ptr = uninit_entries.as_mut_ptr() as *mut T;

        for idx in 0..M {
            unsafe {
                // SAFETY: `idx` is in-bounds for the contiguous `M` allocation.
                base_ptr.add(idx).write(f(idx));
            }
        }

        Self {
            // SAFETY: every entry was initialized exactly once above.
            entries: unsafe { uninit_entries.assume_init() },
        }
    }
}

impl<T, const M: usize> AsRef<[T; M]> for Vector<T, M>
where
    T: Numeric,
{
    #[inline]
    fn as_ref(&self) -> &[T; M] {
        &self.entries
    }
}

impl<T: Numeric, const N: usize> std::ops::Add<&Vector<T, N>> for &Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, other: &Vector<T, N>) -> Self::Output {
        Vector::from_fn(|idx| self.entries[idx] + other.entries[idx])
    }
}
