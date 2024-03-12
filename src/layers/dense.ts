import mj from "../math";
import {
  ActivationType,
  Cost,
  Optimzier,
  OptimzierType,
  StatusLayer,
  matrix2d,
} from "../@types/type";
import setActivation from "../utils/setActivation";
import Matrix from "../matrix";
import setOptimizer from "../utils/setOptimizer";
import setLoss from "../utils/setLoss";

interface DenseLayers {
  units: number;
  outputUnits: number;
  alpha?: number;
  loss?: Cost;
  activation?: ActivationType;
  optimizer?: Optimzier;
  status?: StatusLayer;
}

export interface CompileDenseLayers {
  alpha?: number;
  optimizer?: Optimzier;
  error?: Cost;
}

export default class Dense {
  name = "dense layer";
  units: number;
  outputUnits: number;
  alpha: number;
  loss: number = 0;
  params: number;
  inputShape: [number, number];
  outputShape: [number, number];
  status: StatusLayer;
  bias: Matrix;
  weight: Matrix;
  private sumLoss: number = 0;
  private index: number = 0;
  private optimizerWeight: OptimzierType;
  private optimizerBias: OptimzierType;
  private activationName: ActivationType;
  private optimizerName: Optimzier;
  private lossName: Cost;
  private input: Matrix = mj.matrix([]);
  private dInput: Matrix = mj.matrix([]);
  private result: Matrix = mj.matrix([]);
  private lossFunc: Function;
  private activation: (a: Matrix) => [Matrix, Matrix];
  constructor({
    units,
    outputUnits,
    activation = "linear",
    optimizer = "sgd",
    status = "input",
    alpha = 0.1,
    loss = "mse",
  }: DenseLayers) {
    this.units = units;
    this.outputUnits = outputUnits;
    this.inputShape = [units, 1];
    this.outputShape = [outputUnits, 1];
    this.weight = mj.random([outputUnits, units]);
    this.bias = mj.zeros([outputUnits, 1]);
    this.activation = setActivation(activation);
    this.activationName = activation;
    this.optimizerName = optimizer;
    this.lossName = loss;
    this.status = status;
    this.optimizerWeight = setOptimizer(optimizer, this.weight._shape, 1e-5);
    this.optimizerBias = setOptimizer(optimizer, this.bias._shape, 1e-5);
    this.lossFunc = setLoss(loss);
    this.alpha = alpha;
    this.params = outputUnits * units + units;
  }

  save() {
    const data = {
      name: this.name,
      status: this.status,
      units: this.units,
      outputUnits: this.outputUnits,
      activation: this.activationName,
      optimizer: this.optimizerName,
      loss: this.lossName,
      weight: this.weight._value,
      bias: this.bias._value,
    };
    return data;
  }

  load(weight: matrix2d, bias: matrix2d): void {
    this.weight._value = weight;
    this.bias._value = bias;
  }

  compile({
    alpha = 0.1,
    optimizer = "sgd",
    error = "mse",
  }: CompileDenseLayers): void {
    this.alpha = alpha;
    this.optimizerWeight = setOptimizer(optimizer, this.weight._shape, 1e-5);
    this.optimizerBias = setOptimizer(optimizer, this.bias._shape, 1e-5);
    this.lossFunc = setLoss(error);
    this.optimizerName = optimizer;
    this.lossName = error;
  }

  forward(x: Matrix): Matrix {
    this.input = x;
    const calculateWeightBias = mj.add(
      mj.dotProduct(this.weight, this.input),
      this.bias
    );
    const result = this.activation(calculateWeightBias);
    this.dInput = result[1];
    this.result = result[0];
    return result[0];
  }

  backward(y: Matrix, err: Matrix): Matrix {
    let e: Matrix = mj.matrix([]);
    let loss = 0;
    if (this.status === "output") {
      [loss, e] = this.lossFunc(y, this.result);
      this.index++;
      this.sumLoss += loss;
      this.loss = this.sumLoss / this.index;
    } else {
      e = err;
    }
    const errActivation = mj.mul(e, this.dInput);
    const errWeight = mj.dotProduct(errActivation, mj.transpose(this.input));
    const optimizerWeight = this.optimizerWeight.calculate(
      errWeight,
      this.alpha
    );
    const optimizerBias = this.optimizerBias.calculate(
      errActivation,
      this.alpha
    );
    const errOutput = mj.dotProduct(mj.transpose(this.weight), errActivation);
    this.weight = mj.sub(this.weight, optimizerWeight);
    this.bias = mj.sub(this.bias, optimizerBias);
    return errOutput;
  }
}
