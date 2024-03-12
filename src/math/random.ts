import { MatrixShape, matrix2d } from "../@types/type";
import Matrix from "../matrix";

/**
 * Memberikan nilai matrix random -1 sampai 1 dengan ukuran [n, n]
 * @param shape [number, number]
 * @returns Matrix
 */
export default function random(shape: MatrixShape): Matrix {
  const array: matrix2d = [];
  for (let i = 0; i < shape[0]; i++) {
    const row = [];
    for (let j = 0; j < shape[1]; j++) {
      row.push(Math.random() * 2 - 1);
    }
    array.push(row);
  }
  return new Matrix({ array });
}
