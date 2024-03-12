import { MatrixCollection, MatrixShape, matrix2d } from "../@types/type";

export default class Matrix {
  _value: matrix2d;
  _shape: MatrixShape;
  constructor({ array }: { array: matrix2d }) {
    this._value = array;
    this._shape = [
      array !== undefined ? array.length : 0,
      array[0] !== undefined ? array[0].length : 0,
    ];
  }

  /**
   * Menampilkan nilai dari matrix
   */
  print(): void {
    console.table(
      this._value.map((col) => col.map((val) => parseFloat(val.toFixed(2))))
    );
  }

  /**
   * Memetakan nilai dari matrix ke dalam sebuah function
   * @param func (value: number) => number
   */
  map(func: (value: number) => number) {
    for (let i = 0; i < this._shape[0]; i++) {
      for (let j = 0; j < this._shape[1]; j++) {
        this._value[i][j] = func(this._value[i][j]);
      }
    }
  }

  /**
   * Menjumlahkan matrix dengan a
   * @param a Matix | Number
   */
  add(a: MatrixCollection) {
    try {
      if (typeof a === "number") {
        this.map((val) => val + a);
      } else if (a instanceof Matrix) {
        if (this._shape[0] !== a._shape[0] && this._shape[1] !== a._shape[1]) {
          throw new Error(
            `bentuk dari a harus sama dengan matrix ${this._shape} != ${a._shape}`
          );
        }

        for (let i = 0; i < this._shape[0]; i++) {
          for (let j = 0; j < this._shape[1]; j++) {
            this._value[i][j] += a._value[i][j];
          }
        }
      }
    } catch (err) {
      throw err;
    }
  }

  /**
   * Mengkurangkan matrix dengan a
   * @param a Matix | Number
   */
  sub(a: MatrixCollection) {
    try {
      if (typeof a === "number") {
        this.map((val) => val - a);
      } else if (a instanceof Matrix) {
        if (this._shape[0] !== a._shape[0] && this._shape[1] !== a._shape[1]) {
          throw new Error(
            `bentuk dari a harus sama dengan matrix ${this._shape} != ${a._shape}`
          );
        }

        for (let i = 0; i < this._shape[0]; i++) {
          for (let j = 0; j < this._shape[1]; j++) {
            this._value[i][j] -= a._value[i][j];
          }
        }
      }
    } catch (err) {
      throw err;
    }
  }

  /**
   * Mengkalikan matrix dengan a
   * @param a Matix | Number
   */
  mul(a: MatrixCollection) {
    try {
      if (typeof a === "number") {
        this.map((val) => val * a);
      } else if (a instanceof Matrix) {
        if (this._shape[0] !== a._shape[0] && this._shape[1] !== a._shape[1]) {
          throw new Error(
            `bentuk dari a harus sama dengan matrix ${this._shape} != ${a._shape}`
          );
        }

        for (let i = 0; i < this._shape[0]; i++) {
          for (let j = 0; j < this._shape[1]; j++) {
            this._value[i][j] *= a._value[i][j];
          }
        }
      }
    } catch (err) {
      throw err;
    }
  }

  /**
   * Membagi matrix dengan a
   * @param a Matix | Number
   */
  div(a: MatrixCollection) {
    try {
      if (typeof a === "number") {
        this.map((val) => val / a);
      } else if (a instanceof Matrix) {
        if (this._shape[0] !== a._shape[0] && this._shape[1] !== a._shape[1]) {
          throw new Error(
            `bentuk dari a harus sama dengan matrix ${this._shape} != ${a._shape}`
          );
        }

        for (let i = 0; i < this._shape[0]; i++) {
          for (let j = 0; j < this._shape[1]; j++) {
            this._value[i][j] /= a._value[i][j];
          }
        }
      }
    } catch (err) {
      throw err;
    }
  }
  /**
   * Meratakan matrix menjadi [1, n]
   */
  flatten() {
    const flat: matrix2d = [];
    let index = 0;
    for (let i = 0; i < this._shape[0]; i++) {
      for (let j = 0; j < this._shape[1]; j++) {
        flat[0][index] = this._value[i][j];
        index++;
      }
    }
    this._value = flat;
    this._shape = [1, index];
  }

  /**
   * Merubah bentuk dari matrix
   * @param shape [number, number]
   */
  reshape(shape: MatrixShape) {
    try {
      if (shape[0] * shape[1] !== this._shape[0] * this._shape[1]) {
        throw new Error(
          `Panjang dari shape baru tidak sama dengan yang lama ${
            this._shape[0] * this._shape[1]
          }!=${shape[0] * shape[1]}`
        );
      }

      const newArray: matrix2d = [];
      let index = 0;
      if (this._shape[0] === 1) {
        for (let i = 0; i < shape[0]; i++) {
          for (let j = 0; j < shape[1]; j++) {
            newArray[i][j] = this._value[0][index];
            index++;
          }
        }
      } else {
        this.flatten();
        const newArray: matrix2d = [];
        for (let i = 0; i < shape[0]; i++) {
          for (let j = 0; j < shape[1]; j++) {
            newArray[i][j] = this._value[0][index];
            index++;
          }
        }
      }
      this._value = newArray;
      this._shape = shape;
    } catch (error) {
      throw error;
    }
  }
}
