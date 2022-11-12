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
