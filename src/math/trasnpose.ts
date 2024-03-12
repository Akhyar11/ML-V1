import Matrix from "../matrix";
import zeros from "./zeros";

/**
 * Transposisi matrix [i, j] => [j, i]
 * @param a Matrix
 * @returns Matrix
 */
export default function transpose(a: Matrix): Matrix {
  const matrix = zeros([a._shape[1], a._shape[0]]);
  for (let i = 0; i < a._shape[0]; i++) {
    for (let j = 0; j < a._shape[1]; j++) {
      matrix._value[j][i] = a._value[i][j];
    }
  }
  return matrix;
}
