import Activation from "../layers/activation";
import Convolution from "../layers/convolution";
import Dense from "../layers/dense";
import Matrix from "../matrix";
import AdaGrad from "../optimizer/adaGrad";
import Momentum from "../optimizer/momentum";
import NAG from "../optimizer/nag";
import SGD from "../optimizer/sgd";

export type vector = number[];
export type matrix2d = number[][];
export type matrix3d = number[][][];
export type MatrixCollection = Matrix | number;
export type MatrixShape = [number, number];
export { Matrix };
export type ActivationType = "sigmoid" | "tanh" | "relu" | "lRelu" | "linear";
export type StatusLayer =
  | "input"
  | "output"
  | "norm"
  | "outputReduction"
  | "convOutput";
export type Optimzier = "sgd" | "adaGrad" | "momentum" | "nag";
export type OptimzierType = SGD | AdaGrad | NAG | Momentum;
export type Cost = "mse";
export type Layers = Dense | Activation | Convolution;
