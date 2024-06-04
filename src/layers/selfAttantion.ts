import { Cost, StatusLayer } from "../@types/type";
import { softmax } from "../activation";
import mj from "../math";
import Matrix from "../matrix";
import { setLoss } from "../utils";

interface SelfAttantionLayer {
  units: number;
  alpha?: number;
  loss?: Cost;
  status?: StatusLayer;
}

export default class SelfAttantion {
  name = "self attantion layer";
  units: number;
  outputUnits: number;
  q: Matrix;
  k: Matrix;
  v: Matrix;
  alpha: number;
  loss: number = 0;
  status: StatusLayer = "input";
  private lossFunc: Function;
  private input: Matrix = mj.matrix([]);
  private output: Matrix = mj.matrix([]);
  private attantion: Matrix = mj.matrix([]);
  private dAttantion: Matrix = mj.matrix([]);
  private Q: Matrix = mj.matrix([]);
  private K: Matrix = mj.matrix([]);
  private V: Matrix = mj.matrix([]);
  constructor({
    units,
    alpha = 0.1,
    loss = "mse",
    status = "input",
  }: SelfAttantionLayer) {
    this.units = units;
    this.outputUnits = Math.floor(units / 2);
    this.q = mj.random([this.outputUnits, this.units]);
    this.k = mj.random([this.outputUnits, this.units]);
    this.v = mj.random([this.outputUnits, this.units]);
    this.lossFunc = setLoss(loss);
    this.status = status;
    this.alpha = alpha;
  }

  forward(x: Matrix): Matrix {
    const wq = mj.dotProduct(this.q, x);
    const wk = mj.dotProduct(this.k, x);
    const wv = mj.dotProduct(this.v, x);

    const qk = mj.dotProduct(mj.transpose(wk), wq);
    [this.attantion, this.dAttantion] = softmax(
      mj.div(qk, Math.sqrt(this.outputUnits))
    );
    const output = mj.dotProduct(wv, this.attantion);

    this.input = x;
    this.Q = wq;
    this.K = wk;
    this.V = wv;
    this.output = output;
    return output;
  }

  backward(y: Matrix, err: Matrix) {
    let backwardInput = err;
    let loss = 0;
    if (this.status === "output") {
      [loss, backwardInput] = this.lossFunc(y, this.output);
      console.log(loss);
    } else {
      if (err._shape[1] === 1) {
        backwardInput = mj.reshape(err, this.output._shape);
      }
    }

    const errV = mj.dotProduct(backwardInput, mj.transpose(this.attantion));
    const errAttantion = mj.dotProduct(mj.transpose(this.V), backwardInput);

    const errQK = mj.mul(errAttantion, this.dAttantion);
    const errQ = mj.dotProduct(this.K, errQK);
    const errK = mj.dotProduct(this.Q, errQK);

    const gradQ = mj.dotProduct(errQ, mj.transpose(this.input));
    const gradK = mj.dotProduct(errK, mj.transpose(this.input));
    const gradV = mj.dotProduct(errV, mj.transpose(this.input));

    // Update bobot dengan gradien yang dihitung
    this.q = mj.sub(this.q, mj.mul(this.alpha, gradQ));
    this.k = mj.sub(this.k, mj.mul(this.alpha, gradK));
    this.v = mj.sub(this.v, mj.mul(this.alpha, gradV));

    const gradQOutput = mj.dotProduct(mj.transpose(this.q), errQ);
    const gradKOutput = mj.dotProduct(mj.transpose(this.q), errK);
    const gradVOutput = mj.dotProduct(mj.transpose(this.q), errV);

    let errOutput = mj.mul(gradKOutput, gradQOutput);
    errOutput = mj.mul(errOutput, gradVOutput);
    return errOutput;
  }
}
