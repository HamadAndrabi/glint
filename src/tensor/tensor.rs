//! Core tensor data structure.

use std::fmt;

/// A contiguous f32 tensor with shape and strides.
///
/// Uses row-major layout: the last dimension varies fastest in memory.
/// For a shape `[3, 4]`, element `[i, j]` is at index `i * 4 + j`.
#[derive(Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            data: vec![0.0; numel],
            shape: shape.to_vec(),
            strides: compute_strides(shape),
        }
    }

    /// Create a tensor from existing data. Panics if `data.len() != product(shape)`.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            numel
        );
        Self {
            data,
            shape: shape.to_vec(),
            strides: compute_strides(shape),
        }
    }

    /// Create a tensor from a slice (copies the data).
    pub fn from_slice(data: &[f32], shape: &[usize]) -> Self {
        Self::from_vec(data.to_vec(), shape)
    }

    /// Create a 1D tensor from a slice.
    pub fn from_data(data: Vec<f32>) -> Self {
        let len = data.len();
        Self::from_vec(data, &[len])
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Raw data as a slice.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable access to raw data.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get element by flat index.
    pub fn get_flat(&self, idx: usize) -> f32 {
        self.data[idx]
    }

    /// Get element by multi-dimensional indices.
    pub fn get(&self, indices: &[usize]) -> f32 {
        let idx = self.flat_index(indices);
        self.data[idx]
    }

    /// Set element by multi-dimensional indices.
    pub fn set(&mut self, indices: &[usize], value: f32) {
        let idx = self.flat_index(indices);
        self.data[idx] = value;
    }

    /// Reshape to a new shape (must have same total elements).
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            new_numel,
            "Cannot reshape {:?} ({}) to {:?} ({})",
            self.shape,
            self.numel(),
            new_shape,
            new_numel
        );
        Self::from_vec(self.data.clone(), new_shape)
    }

    /// Extract a row from a 2D tensor. Returns a 1D tensor.
    pub fn row(&self, i: usize) -> Self {
        assert_eq!(self.ndim(), 2, "row() requires 2D tensor");
        let cols = self.shape[1];
        let start = i * cols;
        Self::from_slice(&self.data[start..start + cols], &[cols])
    }

    fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len());
        indices
            .iter()
            .zip(self.strides.iter())
            .map(|(&i, &s)| i * s)
            .sum()
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, numel={})", self.shape, self.numel())
    }
}

/// Compute row-major strides for a given shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert!(t.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_strides() {
        assert_eq!(compute_strides(&[2, 3]), vec![3, 1]);
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
    }

    #[test]
    fn test_indexing() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[0, 2]), 3.0);
        assert_eq!(t.get(&[1, 0]), 4.0);
        assert_eq!(t.get(&[1, 2]), 6.0);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let r = t.reshape(&[3, 2]);
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.get(&[0, 0]), 1.0);
        assert_eq!(r.get(&[2, 1]), 6.0);
    }

    #[test]
    fn test_row() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let row0 = t.row(0);
        assert_eq!(row0.data(), &[1.0, 2.0, 3.0]);
        let row1 = t.row(1);
        assert_eq!(row1.data(), &[4.0, 5.0, 6.0]);
    }
}
