import Matrix from "../matrix";
import zeros from "./zeros";

/**
 * Perkalian product matrix a dan b
 * @param a Matrix
 * @param b Matrix
 * @returns Matrix
 */
export default function dotProduct(a: Matrix, b: Matrix): Matrix {
  try {
    if (a._shape[1] !== b._shape[0]) {
      throw new Error(
        `row dan col dari matrix harus sama ${a._shape[1]}!=${b._shape[0]}`
      );
    }
    const matrix = zeros([a._shape[0], b._shape[1]]);
    for (let i = 0; i < a._shape[0]; i++) {
      for (let j = 0; j < a._shape[1]; j++) {
        for (let k = 0; k < b._shape[1]; k++) {
          matrix._value[i][k] += a._value[i][j] * b._value[j][k];
        }
      }
    }

    return matrix;
  } catch (error) {
    throw error;
  }
}
