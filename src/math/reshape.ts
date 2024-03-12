import { MatrixShape } from "../@types/type";
import Matrix from "../matrix";
import zeros from "./zeros";

export default function reshape(a: Matrix, shape: MatrixShape) {
  try {
    if (a._shape[0] * a._shape[1] !== shape[0] * shape[1]) {
      throw new Error(
        `panjang dari a tidak sama dengan bentuk yang diinginkan ${
          a._shape[0] * a._shape[1]
        }!=${shape[0] * shape[1]}`
      );
    }
    const flat = [];

    if (a._shape[0] === 1) {
      for (let j = 0; j < a._shape[1]; j++) {
        flat.push(a._value[0][j]);
      }
    } else if (a._shape[1] === 1) {
      for (let i = 0; i < a._shape[0]; i++) {
        flat.push(a._value[i][0]);
      }
    } else {
      for (let i = 0; i < a._shape[0]; i++) {
        for (let j = 0; j < a._shape[1]; j++) {
          flat.push(a._value[i][j]);
        }
      }
    }

    const matrix = zeros(shape);
    let index = 0;
    for (let i = 0; i < shape[0]; i++) {
      for (let j = 0; j < shape[1]; j++) {
        matrix._value[i][j] = flat[index];
        index++;
      }
    }

    return matrix;
  } catch (error) {
    throw error;
  }
}
