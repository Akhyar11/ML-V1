import Matrix from "../matrix";

/**
 * Memeberikan nilai absolute pada matrix
 * @param a Matrix
 * @returns Matrix
 */
export default function absm(a: Matrix) {
  const array: number[][] = new Array(a._shape[0]);
  for (let i = 0; i < a._shape[0]; i++) {
    const row = new Array(a._shape[1]);
    for (let j = 0; j < a._shape[1]; j++) {
      row[j] = Math.abs(a._value[i][j]);
    }
    array[i] = row;
  }

  return new Matrix({ array });
}
