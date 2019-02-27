#![allow(dead_code)]
extern crate ndarray;
use ndarray::{ArrayViewMut1, Array1, Array2, Axis};

type PredictFn = fn(ArrayViewMut1<f64>);
type ControlFn = fn(&Array1<f64>) -> Array1<f64>;

struct KalmanFilter {
    state_vec: Array1<f64>,  // N vector representing the system's current state
    cov_mat: Array2<f64>,    // NxN covariance matrix
    pred_mat: Array2<f64>,   // prediction matrix (F_k in academic literature)
    control_fn: ControlFn,   // represents the control matrix
}

impl KalmanFilter {
    // state_vec: state that the system begins in
    // predict_fn: mutates a state vec into a prediction of the next state vec
    // control_fn: takes control vec and gives delta vec which is added to state
    //   i.e. the return value of this function should be a vector with the same
    //   dimension as state_vec
    fn new(
        state_vec: Array1<f64>,
        predict_fn: PredictFn,
        control_fn: ControlFn,
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
            control_fn,
        }
    }

    // update covariances according to the prediction matrix
    // follows from the identity:
    //     Cov(x) = sigma
    //     Cov(Ax) = A * sigma * A^T
    fn update_cov_mat(&mut self) {
        let tmp = self.cov_mat.dot(&self.pred_mat.t());
        self.cov_mat = self.pred_mat.dot(&tmp);
    }

    fn predict_state(&self, control_vec: &Array1<f64>) -> Array1<f64> {
        self.pred_mat.dot(&self.state_vec) + (self.control_fn)(control_vec)
    }
}

#[cfg(test)]
mod tests {
    use crate::KalmanFilter;
    use ndarray::{arr1, arr2, Array1, ArrayViewMut1};

    const TIMESTEP: f64 = 1.0 / 10.0;

    fn predict(mut state_vec: ArrayViewMut1<f64>) {
        let (mut x, v) = (state_vec[0], state_vec[1]);
        x += v * TIMESTEP;
        // v remains unchanged
        state_vec[0] = x;
        state_vec[1] = v;
    }

    fn control(control_vec: &Array1<f64>) -> Array1<f64> {
        let a = control_vec[0];
        let x_delta = 0.5 * TIMESTEP * TIMESTEP * a;
        let v_delta = TIMESTEP * a;
        arr1(&[x_delta, v_delta])
    }

    // Test KalmanFilter::new to make sure it creates the appropriate prediction
    // matrix. For this test, the state vector is (x, v)^T, where x is an
    // x-coordinate and v is velocity, and the prediction function predicts the
    // next x-coordinate after a timestep assuming velocity doesn't change.
    #[test] fn test_predict_matrix() {
        let state = arr1(&[0.0, 0.0]);
        let kf = KalmanFilter::new(state, predict, control);

        assert!(kf.pred_mat == arr2(&[[1.0, TIMESTEP],
                                      [0.0, 1.0     ]]))
    }

    #[test] fn test_predict_state() {
        let x = 0.0;
        let v = 10.0;
        let state = arr1(&[x, v]);

        let a = 5.0;
        let control_vec = arr1(&[a]);

        let kf = KalmanFilter::new(state, predict, control);

        let expected_x = x + v * TIMESTEP + 0.5 * a * TIMESTEP * TIMESTEP;
        let expected_v = v + a * TIMESTEP;
        assert!(
            kf.predict_state(&control_vec) == arr1(&[expected_x, expected_v])
        );
    }
}
