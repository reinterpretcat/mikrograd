#[cfg(test)]
#[path = "../tests/unit/value_test.rs"]
mod value_test;

use crate::create_gradient_fn;
use auto_ops::{impl_op, impl_op_commutative};
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::Sum;
use std::ops::{Add, Deref, Mul};
use std::rc::Rc;

pub(crate) struct GradientData {
    grad: f64,
    data: f64,
}

impl GradientData {
    pub fn new(data: f64) -> Self {
        Self { grad: 0., data }
    }
}

pub(crate) type SharedGradientData = Rc<RefCell<GradientData>>;
pub(crate) type GradientDataFactory = Rc<Box<dyn Fn(f64) -> SharedGradientData>>;
type BackwardFn = Rc<Box<dyn Fn()>>;

#[derive(Clone)]
pub struct Value {
    grad_data: SharedGradientData,
    children: Vec<Value>,
    backward_fn: Option<BackwardFn>,
    gradient_fn: GradientDataFactory,
    op: String,
}

impl Value {
    pub(crate) fn new(data: f64, gradient_fn: GradientDataFactory) -> Self {
        Self { grad_data: gradient_fn(data), children: vec![], backward_fn: None, gradient_fn, op: "".to_string() }
    }

    /// Returns underlying data.
    pub fn get_data(&self) -> f64 {
        self.grad_data.borrow().data
    }

    pub fn set_data(&mut self, value: f64) {
        self.grad_data.borrow_mut().data = value;
    }

    /// Returns a gradient.
    pub fn get_grad(&self) -> f64 {
        self.grad_data.borrow().grad
    }

    /// Sets gradient to zero.
    pub fn zero_grad(&mut self) {
        self.grad_data.borrow_mut().grad = 0.;
    }

    /// Applies gradients.
    pub fn backward(&self) {
        // topological order all of the children in the graph
        let topo = RefCell::new(Vec::new());
        let visited = RefCell::new(HashSet::new());

        fn build_topo<'a>(v: &'a Value, topo: &RefCell<Vec<&'a Value>>, visited: &RefCell<HashSet<&'a Value>>) {
            if !visited.borrow().contains(&v) {
                visited.borrow_mut().insert(v);
                v.children.iter().for_each(|child| build_topo(child, topo, visited));
                topo.borrow_mut().push(v)
            }
        }
        build_topo(self, &topo, &visited);

        // go one variable at a time and apply the chain rule to get its gradient
        self.grad_data.borrow_mut().grad = 1.;
        topo.borrow().iter().rev().filter_map(|v| v.backward_fn.as_ref()).for_each(|backward| backward());
    }
}

mod gradients {
    use super::*;

    pub(crate) fn add(lhs: &SharedGradientData, rhs: &SharedGradientData, out: &SharedGradientData) {
        let out_grad = out.borrow().grad;

        if Rc::ptr_eq(lhs, rhs) {
            lhs.borrow_mut().grad += 2. * out_grad;
        } else {
            lhs.borrow_mut().grad += out_grad;
            rhs.borrow_mut().grad += out_grad;
        }
    }

    pub(crate) fn mul(lhs: &SharedGradientData, rhs: &SharedGradientData, out: &SharedGradientData) {
        let out_grad = out.borrow().grad;

        if Rc::ptr_eq(lhs, rhs) {
            let lhs_data = lhs.borrow().data;
            lhs.borrow_mut().grad += 2. * (lhs_data * out_grad);
        } else {
            lhs.borrow_mut().grad += rhs.borrow().data * out_grad;
            rhs.borrow_mut().grad += lhs.borrow().data * out_grad;
        }
    }

    pub(crate) fn powf(lhs: &SharedGradientData, rhs: f64, out: &SharedGradientData) {
        let lhs_data = lhs.borrow().data;
        lhs.borrow_mut().grad += (rhs * lhs_data.powf(rhs - 1.)) * out.borrow().grad;
    }

    pub(crate) fn relu(lhs: &SharedGradientData, out: &SharedGradientData) {
        let out = out.borrow();

        lhs.borrow_mut().grad += if out.data > 0. { out.grad } else { 0. };
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

macro_rules! custom_operator_impl {
    (use $fn_name: ident for $type_: ident { fn $method: ident$( with $v:tt: $t:ty)? }) => {
        impl $type_ {
            pub fn $method(&self$(, $v: $t)?) -> $type_ {
                let grad_data = self.gradient_fn.deref()(scalars::$fn_name(self.get_data() $(,$v)?));

                let (lhs_gd, out_gd) = (Rc::downgrade(&self.grad_data), Rc::downgrade(&grad_data));

                let backward_fn: Option<BackwardFn> = Some(Rc::new(Box::new(move || {
                    lhs_gd.upgrade().zip(out_gd.upgrade()).iter()
                        .for_each(|(lhs_gd, out_gd)| {
                            gradients::$fn_name(lhs_gd, $($v,)? out_gd);
                        });
                })));

                let op = String::from(stringify!($method));
                let gradient_fn = self.gradient_fn.clone();

                Value { grad_data, children: vec![self.clone()], backward_fn, gradient_fn, op }
            }
        }
    };
}

macro_rules! binary_operator_impl {
    (impl $op:tt for $type_: ident with fn $method: ident and reverse $op_rev:tt fn $method_rev: ident by ($reverse_val: ident, $reverse_arg: ident) ) => {
        fn $method(lhs: &$type_, rhs: &$type_) -> $type_ {
            let grad_data = lhs.gradient_fn.deref()(lhs.get_data().$method(&rhs.get_data()));
            let (lhs_gd, rhs_gd, out_gd) =
                (Rc::downgrade(&lhs.grad_data), Rc::downgrade(&rhs.grad_data), Rc::downgrade(&grad_data));

            let backward_fn: Option<BackwardFn> = Some(Rc::new(Box::new(move || {
                lhs_gd.upgrade().zip(rhs_gd.upgrade()).zip(out_gd.upgrade()).iter().for_each(
                    |((lhs_gd, rhs_gd), out_gd)| {
                        gradients::$method(lhs_gd, rhs_gd, out_gd);
                    },
                );
            })));

            let op = String::from(stringify!($method));
            let gradient_fn = lhs.gradient_fn.clone();
            let children = if Rc::ptr_eq(&lhs.grad_data, &rhs.grad_data) {
                vec![lhs.clone()]
            } else {
                vec![lhs.clone(), rhs.clone()]
            };

            Value { grad_data, children, backward_fn, gradient_fn, op }
        }

        fn $method_rev(lhs: &$type_, rhs: &$type_) -> $type_ {
            let mut value = lhs.$reverse_val(rhs.clone().$reverse_arg(-1.));
            value.op = String::from(stringify!($method_rev));
            value
        }

        impl_op! { $op |a: &Value, b: &Value| -> Value { $method(a, b) } }
        impl_op_commutative! { $op |a: Value, b: &Value| -> Value { $method(&a, b) } }
        impl_op! { $op |a: Value, b: Value| -> Value { $method(&a, &b) } }
        impl_op_commutative! { $op |a: &Value, b: f64| -> Value { $method(a, &Value::new(b, a.gradient_fn.clone()))  } }
        impl_op_commutative! { $op |a: Value, b: f64| -> Value { &a $op b } }

        impl_op! { $op_rev |a: &Value, b: &Value| -> Value { $method_rev(a, b) } }
        impl_op! { $op_rev |a: Value, b: &Value| -> Value { $method_rev(&a, b) } }
        impl_op! { $op_rev |a: Value, b: Value| -> Value { $method_rev(&a, &b) } }
        impl_op! { $op_rev |a: &Value, b: f64| -> Value { $method_rev(a, &Value::new(b, a.gradient_fn.clone()))  } }
        impl_op! { $op_rev |a: Value, b: f64| -> Value { &a $op_rev b } }
        impl_op! { $op_rev |a: f64, b: &Value| -> Value { $method_rev(&Value::new(a, b.gradient_fn.clone()), b)  } }
        impl_op! { $op_rev |a: f64, b: Value| -> Value { a $op_rev &b } }
    };
}

// NOTE assumption: main operator is commutative, reverse - is not
binary_operator_impl! { impl + for Value with fn add and reverse - fn sub by (add, mul) }
binary_operator_impl! { impl * for Value with fn mul and reverse / fn div by (mul, pow) }
custom_operator_impl! { use powf for Value { fn pow with rhs: f64 } }
custom_operator_impl! { use relu for Value { fn relu } }

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.grad_data.as_ref() as *const RefCell<GradientData>).hash(state)
    }
}

impl PartialEq<Self> for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.grad_data, &other.grad_data)
    }
}

impl Eq for Value {}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Value[data={}, grad={}]", self.get_data(), self.get_grad()))
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Value::new(0., create_gradient_fn()), |acc, v| acc + v)
    }
}
