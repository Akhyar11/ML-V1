import { MatrixShape } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";

export default class Momentum {
  prevGradien: Matrix;
  beta = 0.9;
  constructor(shape: MatrixShape) {
    this.prevGradien = mj.zeros(shape);
  }
  calculate(a: Matrix, alpha: number) {
    const meanPrevGradien = mj.mean(this.prevGradien);
    const betaMean = this.beta * meanPrevGradien;
    const newGradien = mj.add(betaMean, mj.mul(a, alpha));
    this.prevGradien = newGradien;
    return newGradien;
  }
}
