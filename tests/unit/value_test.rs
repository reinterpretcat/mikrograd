use super::*;

fn create_dummy_grad() -> GradientFactory {
    Rc::new(Box::new(|| Rc::new(RefCell::new(0.))))
}

fn create_value(data: f64) -> Value {
    Value::new(data, create_dummy_grad())
}

#[test]
fn can_sum_values() {
    let lhs = create_value(3.);
    let rhs = create_value(2.);

    let result = lhs + rhs;

    assert_eq!(result.data, 5.);
    assert_eq!(result.op, "add");
    assert_eq!(result.children.len(), 2);

    let result = create_value(3.) + 2.;
    assert_eq!(result.data, 5.);
    assert_eq!(result.op, "add");
}

#[test]
fn can_subtract_values() {
    let lhs = create_value(3.);
    let rhs = create_value(2.);

    let result = lhs - rhs;
    assert_eq!(result.data, 1.);
    assert_eq!(result.op, "sub");
    assert_eq!(result.children.len(), 2);

    let result = create_value(3.) - 2.;
    assert_eq!(result.data, 1.);
    assert_eq!(result.op, "sub");
}

#[test]
fn can_multiply_values() {
    let lhs = create_value(3.);
    let rhs = create_value(2.);

    let result = lhs * rhs;

    assert_eq!(result.data, 6.);
    assert_eq!(result.op, "mul");
    assert_eq!(result.children.len(), 2);

    let result = create_value(3.4) * 2.;
    assert_eq!(result.data, 6.8);
    assert_eq!(result.op, "mul");
}

#[test]
fn can_divide_values() {
    let lhs = create_value(3.);
    let rhs = create_value(2.);

    let result = lhs / rhs;

    assert_eq!(result.data, 1.5);
    assert_eq!(result.op, "div");
    assert_eq!(result.children.len(), 2);

    let result = create_value(5.) / 2.;
    assert_eq!(result.data, 2.5);
    assert_eq!(result.op, "div");
}

#[test]
fn can_pow_value() {
    let result = create_value(5.).pow(2.);
    assert_eq!(result.data, 25.);
    assert_eq!(result.op, "pow");
    assert_eq!(result.children.len(), 1);
}

#[test]
fn can_relu_value() {
    let result = create_value(5.).relu();
    assert_eq!(result.data, 5.);
    assert_eq!(result.op, "relu");
    assert_eq!(result.children.len(), 1);

    let result = create_value(-1.).relu();
    assert_eq!(result.data, 0.);
    assert_eq!(result.op, "relu");
    assert_eq!(result.children.len(), 1);
}
