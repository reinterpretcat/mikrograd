use mikrograd::{Module, Value, MLP};
use ndarray::prelude::*;

/// Generate test data.
fn make_moons(n_samples: usize) -> (Array<f64, Ix2>, Array<f64, Ix1>) {
    let n_samples_out = n_samples / 2;
    let n_samples_in = n_samples - n_samples_out;

    let outer_circ_x = Array::linspace(0., std::f64::consts::PI, n_samples_out).mapv(f64::cos);
    let outer_circ_y = Array::linspace(0., std::f64::consts::PI, n_samples_out).mapv(f64::sin);
    let inner_circ_x = 1. - Array::linspace(0., std::f64::consts::PI, n_samples_in).mapv(f64::cos);
    let inner_circ_y = 1. - Array::linspace(0., std::f64::consts::PI, n_samples_in).mapv(f64::sin) - 0.5;

    let data = ndarray::stack(
        Axis(1),
        &[
            ndarray::concatenate(Axis(0), &[outer_circ_x.view(), inner_circ_x.view()]).unwrap().view(),
            ndarray::concatenate(Axis(0), &[outer_circ_y.view(), inner_circ_y.view()]).unwrap().view(),
        ],
    )
    .unwrap();

    let labels = 2.
        * ndarray::concatenate(
            Axis(0),
            &[
                Array::<f64, Ix1>::zeros(outer_circ_x.shape()[0]).view(),
                Array::<f64, Ix1>::ones(inner_circ_x.shape()[0]).view(),
            ],
        )
        .unwrap()
        - 1.;

    (data, labels)
}

fn loss(x_data: &Array<f64, Ix2>, y_labels: &Array<f64, Ix1>, model: &MLP) -> (Value, f64) {
    let inputs = x_data.map_axis(Axis(1), |data| data.mapv(mikrograd::new_value));

    // forward the model to get scores
    let scores = inputs.mapv(|input| model.call(input.as_slice().unwrap())[0].clone());

    //svm "max-margin" loss
    let losses = ndarray::Zip::from(y_labels).and(&scores).map_collect(|yi, scorei| (1. + -1. * yi * scorei).relu());
    let losses_len = losses.len() as f64;
    let data_loss = losses.into_iter().sum::<Value>() / losses_len;

    // L2 regularization
    let alpha = 1E-4;
    let reg_loss = alpha * model.parameters().map(|p| p * p).sum::<Value>();
    let total_loss = data_loss + reg_loss;

    // also get accuracy
    let accuracy =
        ndarray::Zip::from(y_labels).and(&scores).map_collect(|&yi, scorei| (yi > 0.) == (scorei.get_data() > 0.));
    let accuracy = accuracy.fold(0., |acc, &hit| acc + if hit { 1. } else { 0. }) / accuracy.len() as f64;

    return (total_loss, accuracy);
}

fn main() {
    let model = mikrograd::new_mlp(2, &[16, 16, 1]);

    println!("{}", model);
    println!("number of parameters: {}", model.parameters().count());

    let (x_data, y_labels) = make_moons(11);
    let (total_loss, accuracy) = loss(&x_data, &y_labels, &model);

    println!("{} {}", total_loss, accuracy);
}
