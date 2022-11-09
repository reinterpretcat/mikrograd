#[cfg(test)]
#[path = "../tests/unit/modules_test.rs"]
mod modules_test;

use crate::value::GradientFactory;
use crate::Value;
use rand::Rng;
use std::fmt::{Display, Formatter};
use std::iter::once;

pub trait Module: Display {
    fn zero_grad(&mut self);
    fn parameters(&self) -> Box<dyn Iterator<Item = &Value> + '_>;
}

#[derive(Clone)]
pub enum NeuronType {
    ReLU,
    Linear,
}

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
    ntype: NeuronType,
}

impl Neuron {
    pub(crate) fn new(nin: usize, ntype: NeuronType, gradient_fn: GradientFactory) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            w: (0..nin).map(|_| rng.gen_range(-1.0..1.0)).map(|data| Value::new(data, gradient_fn.clone())).collect(),
            b: Value::new(0., gradient_fn),
            ntype,
        }
    }

    pub fn call(&self, x: &[f64]) -> Value {
        self.w.iter().cloned().zip(x).fold(self.b.clone(), |acc, (wi, &xi)| acc + wi + xi)
    }
}

impl Module for Neuron {
    fn zero_grad(&mut self) {
        self.w.iter_mut().chain(once(&mut self.b)).for_each(|value| {
            value.zero_grad();
        });
    }

    fn parameters(&self) -> Box<dyn Iterator<Item = &Value> + '_> {
        Box::new(self.w.iter().chain(once(&self.b)))
    }
}

impl Display for Neuron {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ntype = match self.ntype {
            NeuronType::ReLU => "ReLU",
            NeuronType::Linear => "Linear",
        };
        f.write_fmt(format_args!("{}Neuron({})", ntype, self.w.len()))
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub(crate) fn new(nin: usize, nout: usize, ntype: NeuronType, gradient_fn: GradientFactory) -> Self {
        Self { neurons: (0..nout).map(|_| Neuron::new(nin, ntype.clone(), gradient_fn.clone())).collect() }
    }

    pub fn call(&self, x: &[f64]) -> Vec<Value> {
        self.neurons.iter().map(|neuron| neuron.call(x)).collect()
    }
}

impl Module for Layer {
    fn zero_grad(&mut self) {
        self.neurons.iter_mut().for_each(|neuron| neuron.zero_grad());
    }

    fn parameters(&self) -> Box<dyn Iterator<Item = &Value> + '_> {
        Box::new(self.neurons.iter().flat_map(|neuron| neuron.parameters()))
    }
}

impl Display for Layer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let neurons = self.neurons.iter().map(|neuron| neuron.to_string()).collect::<Vec<_>>().join(",");

        f.write_fmt(format_args!("Layer of [{}]", neurons))
    }
}

/// Multilayer Perceptron
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub(crate) fn new(nin: usize, nouts: &[usize], gradient_fn: GradientFactory) -> Self {
        let sz = once(nin).chain(nouts.iter().cloned()).collect::<Vec<_>>();

        Self {
            layers: (0..nouts.len())
                .map(|idx| {
                    let ntype = if idx != (nouts.len() - 1) { NeuronType::ReLU } else { NeuronType::Linear };
                    Layer::new(sz[idx], sz[idx + 1], ntype, gradient_fn.clone())
                })
                .collect(),
        }
    }

    pub fn call(&self, x: &[f64]) -> Vec<Value> {
        let mut iterator = self.layers.iter();
        iterator.next().map(|first| iterator.fold(first.call(x), |_, layer| layer.call(x))).unwrap_or_default()
    }
}

impl Module for MLP {
    fn zero_grad(&mut self) {
        self.layers.iter_mut().for_each(|layer| layer.zero_grad());
    }

    fn parameters(&self) -> Box<dyn Iterator<Item = &Value> + '_> {
        Box::new(self.layers.iter().flat_map(|layer| layer.parameters()))
    }
}

impl Display for MLP {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let layers = self.layers.iter().map(|layer| layer.to_string()).collect::<Vec<_>>().join(",");

        f.write_fmt(format_args!("MLP of [{}]", layers))
    }
}
