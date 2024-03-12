import mj from ".";
import Matrix from "../matrix";

/**
 * Menghitung convolution dari matrix a dengan kernel
 * @param a Matrix
 * @param kernel Matrix
 * @returns Matrix
 */
export default function convolution(a: Matrix, kernel: Matrix): Matrix {
  const matrix = mj.zeros([
    a._shape[0] - kernel._shape[0] + 1,
    a._shape[1] - kernel._shape[1] + 1,
  ]);

  for (let i = 0; i < a._shape[0] - kernel._shape[0] + 1; i++) {
    for (let j = 0; j < a._shape[1] - kernel._shape[1] + 1; j++) {
      for (let k = 0; k < kernel._shape[0]; k++) {
        for (let l = 0; l < kernel._shape[1]; l++) {
          matrix._value[i][j] += a._value[i + k][j + l] * kernel._value[k][l];
        }
      }
    }
  }

  return matrix;
}
