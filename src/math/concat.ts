import Matrix from "../matrix";

/**
 * Menggabungkan dua buah matrix, pastikan matrix sudah di flatten atau berbentuk [1, n]
 * @param a Matrix
 * @param b Matrix
 * @returns Matrix
 */
export default function concat(a: Matrix, b: Matrix) {
  try {
    if (a._shape[0] !== 1 || b._shape[0] !== 1) {
      throw new Error(`pastikan matrix sudah di flatten atau berbentuk [1, n]`);
    }
    const array = [a._value[0].concat(b._value[0])];
    return new Matrix({ array });
  } catch (error) {
    throw error;
  }
}
