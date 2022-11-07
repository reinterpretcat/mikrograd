#[cfg(test)]
#[path = "../tests/unit/value_test.rs"]
mod value_test;

use std::cell::{RefCell, RefMut};
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

type Gradient = Rc<RefCell<f64>>;
type GradientDataMut<'a> = (RefMut<'a, f64>, f64);
type BackwardFn = Box<dyn Fn()>;

pub struct Value {
    grad: Gradient,
    children: Vec<Value>,
    data: f64,
    backward_fn: Option<BackwardFn>,
    op: String,
}

impl Value {
    pub fn new(data: f64) -> Self {
        Self { grad: Rc::new(RefCell::new(0.)), children: vec![], data, backward_fn: None, op: "".to_string() }
    }

    /// Returns a gradient.
    pub fn get_gradient(&self) -> f64 {
        *self.grad.borrow()
    }

    /// Applies gradients.
    pub fn backward(&self) {
        /*
           # topological order all of the children in the graph
           topo = []
           visited = set()
           def build_topo(v):
               if v not in visited:
                   visited.add(v)
                   for child in v._prev:
                       build_topo(child)
                   topo.append(v)
           build_topo(self)

           # go one variable at a time and apply the chain rule to get its gradient
           self.grad = 1
           for v in reversed(topo):
               v._backward()
        */

        unimplemented!()
    }
}

mod gradients {
    use super::*;

    pub fn add(mut lhs: GradientDataMut, mut rhs: GradientDataMut, out: (f64, f64)) {
        let (out_grad, _) = out;
        *lhs.0 += out_grad;
        *rhs.0 += out_grad;
    }

    pub fn mul(mut lhs: GradientDataMut, mut rhs: GradientDataMut, out: (f64, f64)) {
        let (out_grad, _) = out;
        *lhs.0 += rhs.1 * out_grad;
        *rhs.0 -= lhs.1 * out_grad;
    }

    pub fn powf(mut lhs: GradientDataMut, rhs: f64, out: (f64, f64)) {
        let (out_grad, _) = out;
        *lhs.0 += (rhs * lhs.1.powf(rhs - 1.)) * out_grad;
    }

    pub fn relu(mut lhs: GradientDataMut, out: (f64, f64)) {
        let (out_grad, out_data) = out;
        *lhs.0 = if out_data > 0. { out_grad } else { 0. };
    }
}

mod scalars {
    pub fn powf(lhs: f64, rhs: f64) -> f64 {
        lhs.powf(rhs)
    }

    pub fn relu(value: f64) -> f64 {
        value.max(0.)
    }
}

macro_rules! binary_operator_impl {
    (impl $trait_: ident for $type_: ident { fn $method: ident }) => {
        impl $trait_<$type_> for $type_ {
            type Output = $type_;

            fn $method(self, rhs: $type_) -> $type_ {
                let op = String::from(stringify!($method));
                let data = self.data.$method(&rhs.data);
                let grad = Rc::new(RefCell::new(0.));

                let lhs_grad = Rc::downgrade(&self.grad);
                let rhs_grad = Rc::downgrade(&rhs.grad);
                let out_grad = Rc::downgrade(&grad);

                let lhs_data = self.data;
                let rhs_data = rhs.data;

                let backward_fn: Option<BackwardFn> = Some(Box::new(move || {
                    lhs_grad.upgrade().zip(rhs_grad.upgrade()).zip(out_grad.upgrade()).iter().for_each(
                        |((lhs_grad, rhs_grad), out_grad)| {
                            gradients::$method(
                                (lhs_grad.borrow_mut(), lhs_data),
                                (rhs_grad.borrow_mut(), rhs_data),
                                (*out_grad.borrow(), data),
                            );
                        },
                    );
                }));

                Value { grad, children: vec![self, rhs], data, backward_fn, op }
            }
        }

        impl $trait_<f64> for $type_ {
            type Output = $type_;

            fn $method(self, rhs: f64) -> $type_ {
                self.$method($type_::new(rhs))
            }
        }
    };
}

macro_rules! vararg_operator_impl {
    (use $fn_name: ident for $type_: ident { fn $method: ident$( with $v:tt: $t:ty)? }) => {
        impl $type_ {
            pub fn $method(self$(, $v: $t)?) -> $type_ {
                let op = String::from(stringify!($method));
                let data = scalars::$fn_name(self.data $(,$v)?);
                let grad = Rc::new(RefCell::new(0.));

                let lhs_grad = Rc::downgrade(&self.grad);
                let out_grad = Rc::downgrade(&grad);

                let lhs_data = self.data;
                let backward_fn: Option<BackwardFn> = Some(Box::new(move || {
                    lhs_grad.upgrade().zip(out_grad.upgrade()).iter()
                        .for_each(|(lhs_grad, out_grad)| {
                            gradients::$fn_name(
                                (lhs_grad.borrow_mut(), lhs_data),
                                $($v,)?
                                (*out_grad.borrow(), data)
                            );
                        });
                }));

                Value { grad, children: vec![self], data, backward_fn, op }
            }
        }
    };
}

macro_rules! reverse_operator_impl {
    (impl $trait_: ident for $type_: ident { fn $method: ident by ($reverse_val: ident, $reverse_arg: ident) }) => {
        impl $trait_<$type_> for $type_ {
            type Output = $type_;

            fn $method(self, rhs: $type_) -> $type_ {
                let mut value = self.$reverse_val(rhs.$reverse_arg(-1.));
                value.op = String::from(stringify!($method));

                value
            }
        }

        impl $trait_<f64> for $type_ {
            type Output = $type_;

            fn $method(self, rhs: f64) -> $type_ {
                self.$method(Value::new(rhs))
            }
        }
    };
}

// Implement binary operations with values
binary_operator_impl! { impl Add for Value { fn add } }
binary_operator_impl! { impl Mul for Value { fn mul } }

reverse_operator_impl! { impl Sub for Value { fn sub by (add, mul) } }
reverse_operator_impl! { impl Div for Value { fn div by (mul, pow) } }

vararg_operator_impl! { use powf for Value { fn pow with rhs: f64 } }
vararg_operator_impl! { use relu for Value { fn relu } }
