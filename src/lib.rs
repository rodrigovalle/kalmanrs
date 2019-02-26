#![allow(dead_code)]
extern crate ndarray;
use ndarray::{ArrayViewMut1, Array1, Array2, Axis};

struct KalmanFilter {
    state: Array1<f64>,  // N vector representing the system's current state
    covariance: Array2<f64>,  // NxN matrix
    F_k: Array2<f64>,
}

impl KalmanFilter {
    // predict is a function that mutates a state vector into the next state
    // vector
    fn new(
        state: Array1<f64>,
        predict: fn(ArrayViewMut1<f64>),
    ) -> KalmanFilter {
        let n = state.dim();

        // iterate through all unit vectors and call predict to construct the
        // matrix F_k, the prediction matrix
        let mut F_k = Array2::<f64>::eye(n);
        F_k.axis_iter_mut(Axis(1)).for_each(predict);

        KalmanFilter {
            state,
            F_k,
            covariance: Array2::ones((n, n)),
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

    // test predict with a 
    #[test] fn test_predict() {
        let state = arr1(&[0.0, 10.0]);
        let kf = KalmanFilter::new(state, predict);

        assert!(kf.F_k == arr2(&[[1.0, TIMESTEP],
                                 [0.0, 1.0     ]]))
    }
}
