import Matrix from "../matrix";

/**
 * Meratakan matrix menjadi ukuran [1, n]
 * @param a Matrix
 * @returns Matrix
 */
export default function flatten(a: Matrix): Matrix {
  const array: number[][] = [];
  let index = 0;
  for (let i = 0; i < a._shape[0]; i++) {
    for (let j = 0; j < a._shape[1]; j++) {
      array[0][index] = a._value[i][j];
      index++;
    }
  }

  return new Matrix({ array });
}
