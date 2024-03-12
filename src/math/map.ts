import Matrix from "../matrix";

/**
 * Memetakan matrix kedalam function
 * @param a Matrix
 * @param func (number) => number
 * @returns Matrix
 */
export default function map(
  a: Matrix,
  func: (value: number) => number
): Matrix {
  const array: number[][] = new Array(a._shape[0]);
  for (let i = 0; i < a._shape[0]; i++) {
    const row = new Array(a._shape[1]);
    for (let j = 0; j < a._shape[1]; j++) {
      row[j] = func(a._value[i][j]);
    }
    array[i] = row;
  }

  return new Matrix({ array });
}
