import { MatrixShape, matrix2d } from "../@types/type";
import Matrix from "../matrix";

/**
 * Memberikan nilai matrix 1 dengan ukuran [n, n]
 * @param shape [number, number]
 * @returns Matrix
 */
export default function ones(shape: MatrixShape): Matrix {
  const array: matrix2d = new Array(shape[0]);
  for (let i = 0; i < shape[0]; i++) {
    array[i] = new Array(shape[1]).fill(1);
  }
  return new Matrix({ array });
}
