use super::*;

#[test]
fn can_create_neuron() {
    let neuron = Neuron::new(2, NeuronType::ReLU);

    assert_eq!(neuron.w.len(), 2);
    assert_eq!(neuron.parameters().count(), 3);
    assert!(matches!(neuron.ntype, NeuronType::ReLU));
}

#[test]
fn can_create_layer() {
    let layer = Layer::new(3, 4, NeuronType::Linear);

    assert_eq!(layer.neurons.len(), 4);
    assert_eq!(layer.parameters().count(), 16);
}

#[test]
fn can_create_mlp() {
    let layer = MLP::new(2, &[16, 16, 1]);

    assert_eq!(layer.layers.len(), 2 + 1);
    assert_eq!(layer.parameters().count(), 337);
}

#[test]
fn can_process_data_in_neuron() {
    let neuron = Neuron { w: vec![Value::new(10.), Value::new(100.)], b: Value::new(3.), ntype: NeuronType::Linear };

    let result = neuron.call(&[Value::new(1.2), Value::new(1.3)]);

    assert_eq!(result.get_data(), 145.);
    assert_eq!(result.get_grad(), 0.);
}
