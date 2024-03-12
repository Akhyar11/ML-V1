import mj from "../math";
import Matrix from "../matrix";

export function sigmoid(a: Matrix): [Matrix, Matrix] {
  const result = mj.map(a, (val) => 1 / (1 + Math.exp(-val)));
  const dResult = mj.map(result, (val) => val * (1 - val));
  return [result, dResult];
}

export function tanh(a: Matrix): [Matrix, Matrix] {
  const result = mj.map(a, (val) => Math.tanh(val));
  const dResult = mj.map(result, (val) => 1 - val ** 2);
  return [result, dResult];
}

export function relu(a: Matrix): [Matrix, Matrix] {
  const result = mj.map(a, (val) => (val < 0 ? 0 : val));
  const dResult = mj.map(result, (val) => (val === 0 ? 0 : 1));
  return [result, dResult];
}

export function lRelu(a: Matrix): [Matrix, Matrix] {
  const result = mj.map(a, (val) => (val < 0 ? val * 1e-5 : val));
  const dResult = mj.map(result, (val) => (val < 0 ? val * 1e-5 : 1));
  return [result, dResult];
}

export default function linear(a: Matrix): [Matrix, Matrix] {
  const result = a;
  const dResult = mj.ones(a._shape);
  return [result, dResult];
}
