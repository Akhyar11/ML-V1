import Matrix from "../matrix";

/**
 * Matrix a => Matrix exp(a)
 * @param a Matrix
 * @returns Number
 */
export default function expm(a: Matrix): Matrix {
  const array: number[][] = new Array(a._shape[0]);
  for (let i = 0; i < a._shape[0]; i++) {
    const row = new Array(a._shape[1]);
    for (let j = 0; j < a._shape[1]; j++) {
      row[j] = Math.exp(a._value[i][j]);
    }
    array[i] = row;
  }

  return new Matrix({ array });
}
