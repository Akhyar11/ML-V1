import { MatrixShape } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";

export default class AdaGrad {
  shape: MatrixShape;
  sumGradien: Matrix;
  epsilon: number = 0.1;
  constructor(shape: MatrixShape, epsilon: number) {
    this.shape = shape;
    this.sumGradien = mj.zeros(this.shape);
    this.epsilon = epsilon;
  }

  calculate(a: Matrix, alpha: number) {
    const sumGradien = mj.add(this.sumGradien, a);
    const powerGradien = mj.map(sumGradien, (val) => val ** 2);
    const addEpsilon = mj.add(powerGradien, this.epsilon);
    const sqrtGradien = mj.map(addEpsilon, (val) => Math.sqrt(val));
    const newAlpha = mj.div(alpha, sqrtGradien);
    this.sumGradien = sumGradien;
    return mj.mul(newAlpha, a);
  }
}
