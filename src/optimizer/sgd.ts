import mj from "../math";
import Matrix from "../matrix";

export default class SGD {
  calculate(a: Matrix, alpha: number): Matrix {
    return mj.mul(a, alpha);
  }
}
