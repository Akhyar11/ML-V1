import { MatrixShape, Optimzier } from "../@types/type";
import AdaGrad from "../optimizer/adaGrad";
import Momentum from "../optimizer/momentum";
import NAG from "../optimizer/nag";
import SGD from "../optimizer/sgd";

export default function setOptimizer(
  optimzier: Optimzier,
  shape: MatrixShape,
  alpha: number
) {
  switch (optimzier) {
    case "adaGrad":
      return new AdaGrad(shape, alpha);
    case "sgd":
      return new SGD();
    case "momentum":
      return new Momentum(shape);
    case "nag":
      return new NAG(shape);
  }
}
