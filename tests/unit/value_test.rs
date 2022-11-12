use super::*;

fn create_value(data: f64) -> Value {
    Value::new(data)
}

#[test]
fn can_sum_values() {
    let lhs = create_value(3.);
    let rhs = create_value(2.);

    let result1 = &lhs + &rhs;
    let result2 = lhs + rhs;

    assert_eq!(result1.get_data(), 5.);
    assert_eq!(result1.op, "add");
    assert_eq!(result1.children.len(), 2);
    assert_eq!(result2.get_data(), 5.);

    let result = create_value(3.) + 2.;
    assert_eq!(result.get_data(), 5.);
    assert_eq!(result.op, "add");

    let result = 3. + create_value(2.) + 2.;
    assert_eq!(result.get_data(), 7.);
    assert_eq!(result.op, "add");
}

#[test]
fn can_multiply_values() {
    let lhs = create_value(3.);
    let rhs = create_value(2.);

    let result = lhs * rhs;

    assert_eq!(result.get_data(), 6.);
    assert_eq!(result.op, "mul");
    assert_eq!(result.children.len(), 2);

    let result = create_value(3.4) * 2.;
    assert_eq!(result.get_data(), 6.8);
    assert_eq!(result.op, "mul");

    let result = 2. * create_value(3.4);
    assert_eq!(result.get_data(), 6.8);
    assert_eq!(result.op, "mul");
}

#[test]
fn can_subtract_values() {
    let lhs = create_value(3.);
    let rhs = create_value(2.);

    let result = lhs - rhs;
    assert_eq!(result.get_data(), 1.);
    assert_eq!(result.op, "sub");
    assert_eq!(result.children.len(), 2);

    let result = create_value(3.) - 2.;
    assert_eq!(result.get_data(), 1.);
    assert_eq!(result.op, "sub");

    let result = 3. - create_value(2.);
    assert_eq!(result.get_data(), 1.);
    assert_eq!(result.op, "sub");
}

#[test]
fn can_divide_values() {
    let lhs = create_value(3.);
    let rhs = create_value(2.);

    let result = lhs / rhs;

    assert_eq!(result.get_data(), 1.5);
    assert_eq!(result.op, "div");
    assert_eq!(result.children.len(), 2);

    let result = create_value(5.) / 2.;
    assert_eq!(result.get_data(), 2.5);
    assert_eq!(result.op, "div");

    let result = 5. / create_value(2.);
    assert_eq!(result.get_data(), 2.5);
    assert_eq!(result.op, "div");
}

#[test]
fn can_pow_value() {
    let result = create_value(5.).pow(2.);
    assert_eq!(result.get_data(), 25.);
    assert_eq!(result.op, "pow");
    assert_eq!(result.children.len(), 1);
}

#[test]
fn can_relu_value() {
    let result = create_value(5.).relu();
    assert_eq!(result.get_data(), 5.);
    assert_eq!(result.op, "relu");
    assert_eq!(result.children.len(), 1);

    let result = create_value(-1.).relu();
    assert_eq!(result.get_data(), 0.);
    assert_eq!(result.op, "relu");
    assert_eq!(result.children.len(), 1);
}

#[test]
fn can_calculate_simple_gradient() {
    let x = Value::new(-4.);
    let z = 2.5 * x.clone();
    z.backward();
    assert_eq!(x.get_grad(), 2.5);
}

#[test]
fn can_calculate_gradient_with_double_borrowing() {
    let x = Value::new(-4.);
    let z = x.clone() * x.clone();
    z.backward();
    assert_eq!(x.get_grad(), -8.);

    let x = Value::new(-4.);
    let z = x.clone() + x.clone();
    z.backward();
    assert_eq!(x.get_grad(), 2.)
}

#[test]
fn can_calculate_reference_gradients() {
    let x = Value::new(-4.);
    let z = 2. * &x + 2. + &x;
    let q = z.relu() + &z * &x;
    let h = (&z * &z).relu();
    let y = h + &q + q * &x;
    y.backward();

    assert_eq!(x.get_grad(), 46.);
    assert_eq!(y.get_grad(), 1.);
}
