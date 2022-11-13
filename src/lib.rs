mod modules;
pub use self::modules::*;

mod value;
pub use self::value::Value;

// TODO add prelude

pub fn new_mlp(nin: usize, nouts: &[usize]) -> MLP {
    MLP::new(nin, nouts)
}

pub fn new_value(data: f64) -> Value {
    Value::new(data)
}
