use super::*;

#[test]
fn can_sum_values() {
    let lhs = Value::new(3.);
    let rhs = Value::new(2.);

    let result = lhs + rhs;

    assert_eq!(result.data, 5.);
    assert_eq!(result.op, "add");
    assert_eq!(result.children.len(), 2);

    let result = Value::new(3.) + 2.;
    assert_eq!(result.data, 5.);
    assert_eq!(result.op, "add");
}

#[test]
fn can_subtract_values() {
    let lhs = Value::new(3.);
    let rhs = Value::new(2.);

    let result = lhs - rhs;
    assert_eq!(result.data, 1.);
    assert_eq!(result.op, "sub");
    assert_eq!(result.children.len(), 2);

    let result = Value::new(3.) - 2.;
    assert_eq!(result.data, 1.);
    assert_eq!(result.op, "sub");
}

#[test]
fn can_multiply_values() {
    let lhs = Value::new(3.);
    let rhs = Value::new(2.);

    let result = lhs * rhs;

    assert_eq!(result.data, 6.);
    assert_eq!(result.op, "mul");
    assert_eq!(result.children.len(), 2);

    let result = Value::new(3.4) * 2.;
    assert_eq!(result.data, 6.8);
    assert_eq!(result.op, "mul");
}

#[test]
fn can_divide_values() {
    let lhs = Value::new(3.);
    let rhs = Value::new(2.);

    let result = lhs / rhs;

    assert_eq!(result.data, 1.5);
    assert_eq!(result.op, "div");
    assert_eq!(result.children.len(), 2);

    let result = Value::new(5.) / 2.;
    assert_eq!(result.data, 2.5);
    assert_eq!(result.op, "div");
}

#[test]
fn can_pow_value() {
    let result = Value::new(5.).pow(2.);
    assert_eq!(result.data, 25.);
    assert_eq!(result.op, "pow");
    assert_eq!(result.children.len(), 1);
}

#[test]
fn can_relu_value() {
    let result = Value::new(5.).relu();
    assert_eq!(result.data, 5.);
    assert_eq!(result.op, "relu");
    assert_eq!(result.children.len(), 1);

    let result = Value::new(-1.).relu();
    assert_eq!(result.data, 0.);
    assert_eq!(result.op, "relu");
    assert_eq!(result.children.len(), 1);
}
