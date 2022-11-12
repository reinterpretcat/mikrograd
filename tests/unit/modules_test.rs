use super::*;
use crate::GradientData;
use std::cell::RefCell;
use std::rc::Rc;

fn create_dummy_grad() -> GradientDataFactory {
    Rc::new(Box::new(|data| Rc::new(RefCell::new(GradientData::new(data)))))
}

#[test]
fn can_create_neuron() {
    let neuron = Neuron::new(2, NeuronType::ReLU, create_dummy_grad());

    assert_eq!(neuron.w.len(), 2);
    assert_eq!(neuron.parameters().count(), 3);
    assert!(matches!(neuron.ntype, NeuronType::ReLU));
}

#[test]
fn can_create_layer() {
    let layer = Layer::new(3, 4, NeuronType::Linear, create_dummy_grad());

    assert_eq!(layer.neurons.len(), 4);
    assert_eq!(layer.parameters().count(), 16);
}

#[test]
fn can_create_mlp() {
    let layer = MLP::new(2, &[16, 16, 1], create_dummy_grad());

    assert_eq!(layer.layers.len(), 2 + 1);
    assert_eq!(layer.parameters().count(), 337);
}

#[test]
fn can_process_data_in_neuron() {
    let gradient_fn = create_dummy_grad();
    let neuron = Neuron {
        w: vec![Value::new(10., gradient_fn.clone()), Value::new(100., gradient_fn.clone())],
        b: Value::new(3., gradient_fn.clone()),
        ntype: NeuronType::Linear,
    };

    let result = neuron.call(&[Value::new(1.2, gradient_fn.clone()), Value::new(1.3, gradient_fn.clone())]);

    assert_eq!(result.get_data(), 145.);
    assert_eq!(result.get_grad(), 0.);
}
