mod modules;

pub use self::modules::*;
use crate::value::{GradientData, GradientDataFactory};
use std::cell::RefCell;
use std::rc::Rc;

mod value;
pub use self::value::Value;

pub fn new_mlp(nin: usize, nouts: &[usize]) -> MLP {
    MLP::new(nin, nouts, create_gradient_fn())
}

pub fn new_value(data: f64) -> Value {
    Value::new(data, create_gradient_fn())
}

fn create_gradient_fn() -> GradientDataFactory {
    Rc::new(Box::new(|data| Rc::new(RefCell::new(GradientData::new(data)))))
}
