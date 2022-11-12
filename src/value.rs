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

pub(crate) type Gradient = Rc<RefCell<f64>>;
pub(crate) type GradientFactory = Rc<Box<dyn Fn() -> Gradient>>;
type BackwardFn = Rc<Box<dyn Fn()>>;

#[derive(Clone)]
pub struct Value {
    grad: Gradient,
    children: Vec<Value>,
    data: f64,
    backward_fn: Option<BackwardFn>,
    gradient_fn: GradientFactory,
    op: String,
}

impl Value {
    pub(crate) fn new(data: f64, gradient_fn: GradientFactory) -> Self {
        Self { grad: gradient_fn(), children: vec![], data, backward_fn: None, gradient_fn, op: "".to_string() }
    }

    /// Returns underlying data.
    pub fn get_data(&self) -> f64 {
        self.data
    }

    pub fn set_data(&mut self, value: f64) {
        self.data = value;
    }

    /// Returns a gradient.
    pub fn get_grad(&self) -> f64 {
        *self.grad.borrow()
    }

    /// Sets gradient to zero.
    pub fn zero_grad(&mut self) {
        *self.grad.borrow_mut() = 0.;
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
        *self.grad.borrow_mut() = 1.;
        topo.borrow().iter().rev().filter_map(|v| v.backward_fn.as_ref()).for_each(|backward| backward());
    }
}

mod gradients {
    use super::*;

    pub fn add(lhs: (&Gradient, f64), rhs: (&Gradient, f64), out: (f64, f64)) {
        let (out_grad, _) = out;

        if Rc::ptr_eq(lhs.0, rhs.0) {
            let mut lhs_grad = lhs.0.borrow_mut();
            *lhs_grad += 2. * out_grad;
        } else {
            let (mut lhs_grad, mut rhs_grad) = (lhs.0.borrow_mut(), rhs.0.borrow_mut());

            *lhs_grad += out_grad;
            *rhs_grad += out_grad;
        }
    }

    pub fn mul(lhs: (&Gradient, f64), rhs: (&Gradient, f64), out: (f64, f64)) {
        let (out_grad, _) = out;

        if Rc::ptr_eq(lhs.0, rhs.0) {
            let (mut lhs_grad, lhs_data) = (lhs.0.borrow_mut(), lhs.1);
            *lhs_grad += 2. * (lhs_data * out_grad);
        } else {
            let (mut lhs_grad, lhs_data) = (lhs.0.borrow_mut(), lhs.1);
            let (mut rhs_grad, rhs_data) = (rhs.0.borrow_mut(), rhs.1);

            *lhs_grad += rhs_data * out_grad;
            *rhs_grad += lhs_data * out_grad;
        }
    }

    pub fn powf(lhs: (&Gradient, f64), rhs: f64, out: (&Gradient, f64)) {
        let (mut lhs_grad, lhs_data) = (lhs.0.borrow_mut(), lhs.1);
        let (out_grad, _) = (*out.0.borrow(), out.1);

        *lhs_grad += (rhs * lhs_data.powf(rhs - 1.)) * out_grad;
    }

    pub fn relu(lhs: (&Gradient, f64), out: (&Gradient, f64)) {
        let (mut lhs_grad, _) = (lhs.0.borrow_mut(), lhs.1);
        let (out_grad, out_data) = (*out.0.borrow(), out.1);

        *lhs_grad += if out_data > 0. { out_grad } else { 0. };
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
                let data = scalars::$fn_name(self.data $(,$v)?);
                let grad = self.gradient_fn.deref()();
                let (lhs_grad, out_grad) = (Rc::downgrade(&self.grad), Rc::downgrade(&grad));
                let lhs_data = self.data;

                let backward_fn: Option<BackwardFn> = Some(Rc::new(Box::new(move || {
                    lhs_grad.upgrade().zip(out_grad.upgrade()).iter()
                        .for_each(|(lhs_grad, out_grad)| {
                            gradients::$fn_name(
                                (lhs_grad, lhs_data),
                                $($v,)?
                                (out_grad, data)
                            );
                        });
                })));

                let op = String::from(stringify!($method));
                let gradient_fn = self.gradient_fn.clone();

                Value { grad, children: vec![self.clone()], data, backward_fn, gradient_fn, op }
            }
        }
    };
}

macro_rules! binary_operator_impl {
    (impl $op:tt for $type_: ident with fn $method: ident and reverse $op_rev:tt fn $method_rev: ident by ($reverse_val: ident, $reverse_arg: ident) ) => {
        fn $method(lhs: &$type_, rhs: &$type_) -> $type_ {
            let data = lhs.data.$method(&rhs.data);
            let grad = lhs.gradient_fn.deref()();

            let (lhs_data, rhs_data) = (lhs.data, rhs.data);
            let (lhs_grad, rhs_grad, out_grad) =
                (Rc::downgrade(&lhs.grad), Rc::downgrade(&rhs.grad), Rc::downgrade(&grad));

            let backward_fn: Option<BackwardFn> = Some(Rc::new(Box::new(move || {
                lhs_grad.upgrade().zip(rhs_grad.upgrade()).zip(out_grad.upgrade()).iter().for_each(
                    |((lhs_grad, rhs_grad), out_grad)| {
                        gradients::$method((lhs_grad, lhs_data), (rhs_grad, rhs_data), (*out_grad.borrow(), data));
                    },
                );
            })));

            let op = String::from(stringify!($method));
            let gradient_fn = lhs.gradient_fn.clone();
            let children =
                if Rc::ptr_eq(&lhs.grad, &rhs.grad) { vec![lhs.clone()] } else { vec![lhs.clone(), rhs.clone()] };

            Value { grad, children, data, backward_fn, gradient_fn, op }
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
        (self.grad.as_ref() as *const RefCell<f64>).hash(state)
    }
}

impl PartialEq<Self> for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.grad, &other.grad)
    }
}

impl Eq for Value {}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Value[data={}, grad={}]", self.data, self.get_grad()))
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
