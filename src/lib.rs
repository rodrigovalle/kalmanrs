#![allow(dead_code)]
extern crate ndarray;
use ndarray::{ArrayViewMut1, Array1, Array2, Axis};

struct KalmanFilter {
    state_vec: Array1<f64>,  // N vector representing the system's current state
    cov_mat: Array2<f64>,    // NxN covariance matrix
    pred_mat: Array2<f64>,   // prediction matrix (F_k in academic literature)
}

impl KalmanFilter {
    // state_vec: state that the system begins in
    // predict_fn: mutates a state vec into a prediction of the next state vec
    fn new(
        state_vec: Array1<f64>,
        predict_fn: fn(ArrayViewMut1<f64>),
    ) -> KalmanFilter {
        let n = state_vec.dim();

        // iterate through all unit vectors (columns of the identity matrix) and
        // call predict_fn to construct the prediction matrix, a matrix which
        // transforms any state vec into the prediction of the next state vec
        let mut pred_mat = Array2::<f64>::eye(n);
        pred_mat.axis_iter_mut(Axis(1)).for_each(predict_fn);

        KalmanFilter {
            state_vec,
            pred_mat,
            cov_mat: Array2::ones((n, n)),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::KalmanFilter;
    use ndarray::{arr1, arr2, ArrayViewMut1};

    const TIMESTEP: f64 = 1.0 / 10.0;

    fn predict(mut state_vec: ArrayViewMut1<f64>) {
        let (mut x, v) = (state_vec[0], state_vec[1]);
        x += v * TIMESTEP;
        // v remains unchanged
        state_vec[0] = x;
        state_vec[1] = v;
    }

    // Test KalmanFilter::new to make sure it creates the appropriate prediction
    // matrix. For this test, the state vector is (x, v)^T, where x is an
    // x-coordinate and v is velocity, and the prediction function predicts the
    // next x-coordinate after a timestep assuming velocity doesn't change.
    #[test] fn test_predict_matrix() {
        let state = arr1(&[0.0, 10.0]);
        let kf = KalmanFilter::new(state, predict);

        assert!(kf.pred_mat == arr2(&[[1.0, TIMESTEP],
                                      [0.0, 1.0     ]]))
    }
}
