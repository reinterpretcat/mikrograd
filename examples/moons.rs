use mikrograd::{Module, Value, MLP};
use ndarray::prelude::*;
use plotters::prelude::*;

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

    // make label be -1 or 1
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
    let losses = ndarray::Zip::from(y_labels).and(&scores).map_collect(|&yi, scorei| (1. + -yi * scorei).relu());
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

fn run_optimization(x_data: &Array<f64, Ix2>, y_labels: &Array<f64, Ix1>, model: &mut MLP, n_opt_steps: usize) {
    // optimization
    for k in 0..n_opt_steps {
        // forward
        let (total_loss, accuracy) = loss(&x_data, &y_labels, &model);

        // backward
        model.zero_grad();
        total_loss.backward();

        // update (sgd)
        let learning_rate = 1. - 0.9 * k as f64 / 100.;
        for p in model.parameters_mut() {
            p.set_data(p.get_data() - learning_rate * p.get_grad());
        }

        println!("step {} loss {}, accuracy {:.2}%", k, total_loss.get_data(), accuracy * 100.);
    }
}

fn visualize_results(
    x_data: &Array<f64, Ix2>,
    _y_labels: &Array<f64, Ix1>,
    model: &MLP,
    image_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    const POINTS: usize = 100;

    let root = BitMapBackend::new(image_path, (640, 640)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_x = -2.;
    let max_x = 3.;
    let min_y = -2.;
    let max_y = 2.;

    let step_x = (max_x - min_x) / POINTS as f64;
    let step_y = (max_y - min_y) / POINTS as f64;

    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .top_x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart
        .configure_mesh()
        .x_labels(15)
        .y_labels(10)
        .disable_x_mesh()
        .disable_y_mesh()
        .label_style(("sans-serif", 20))
        .draw()?;

    #[derive(Copy, Clone, Default)]
    struct MatrixPoint {
        coords: [(f64, f64); 2],
        prediction: f64,
    }
    let mut matrix = Vec::with_capacity(POINTS);

    for x in 0..POINTS {
        matrix.push(Vec::with_capacity(POINTS));
        for y in 0..POINTS {
            let coord_1 = (min_x + x as f64 * step_x, min_y + y as f64 * step_y);
            let coord_2 = (coord_1.0 + step_x, coord_1.1 + step_y);
            let point_x = coord_1.0 + step_x / 2.;
            let point_y = coord_1.1 + step_y / 2.;

            let prediction =
                model.call(&[mikrograd::new_value(point_x), mikrograd::new_value(point_y)]).first().unwrap().get_data();
            matrix[x].push(MatrixPoint { coords: [coord_1, coord_2], prediction });
        }
    }

    chart.draw_series(matrix.iter().flat_map(|points| points.iter()).map(|point| {
        let color = if point.prediction > 0. { RGBColor(255, 0, 0).filled() } else { RGBColor(0, 0, 255).filled() };
        Rectangle::new(point.coords, color)
    }))?;

    chart
        .draw_series(x_data.map_axis(Axis(1), |data| {
            // TODO use different colors depending on labels
            TriangleMarker::new((data[0], data[1]), 5, &YELLOW)
        }))
        .unwrap();

    root.present().expect("Unable to write result to file");
    println!("Result has been saved to {}", image_path);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_samples = 100;
    let n_opt_steps = 10;
    let mut model = mikrograd::new_mlp(2, &[16, 16, 1]);

    println!("{}", model);
    println!("number of parameters: {}", model.parameters().count());

    // generate test data
    let (x_data, y_labels) = make_moons(n_samples);

    let (total_loss, accuracy) = loss(&x_data, &y_labels, &model);
    println!("{}{}", total_loss, accuracy);

    // run gradient descent optimization
    run_optimization(&x_data, &y_labels, &mut model, n_opt_steps);

    // generate and store bitmap
    visualize_results(&x_data, &y_labels, &model, format!("moons_{}.png", n_samples).as_str())
}
