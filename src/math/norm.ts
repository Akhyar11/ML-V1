import Matrix from "../matrix";

export default function norm(a: Matrix) {
  const array = Math.sqrt(a._value[0].reduce((acc, val) => acc + val ** 2, 0));
  return array;
}
