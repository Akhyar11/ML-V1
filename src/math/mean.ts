import Matrix from "../matrix";

/**
 * Merata rata nilai matrix
 * @param a Matrix
 * @returns Number
 */
export default function mean(a: Matrix): number {
  let value: number = 0;
  for (let i = 0; i < a._shape[0]; i++) {
    for (let j = 0; j < a._shape[1]; j++) {
      value += a._value[i][j];
    }
  }

  return value / (a._shape[0] * a._shape[1]);
}
