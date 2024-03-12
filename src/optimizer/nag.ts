import { MatrixShape } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";

export default class NAG {
  prevGradien: Matrix;
  beta = 0.9;
  constructor(shape: MatrixShape) {
    this.prevGradien = mj.zeros(shape);
  }

  calculate(a: Matrix, alpha: number) {
    const meanGradien = mj.mean(this.prevGradien);
    const betaGradien = this.beta * meanGradien;
    const wUpdate = mj.sub(a, betaGradien);
    const newGradien = mj.add(betaGradien, mj.mul(alpha, wUpdate));
    this.prevGradien = newGradien;
    return newGradien;
  }
}
