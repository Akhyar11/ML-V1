import linear, { lRelu, relu, sigmoid, tanh } from "../activation";
import { ActivationType } from "../@types/type";

export default function setActivation(activation: ActivationType) {
  switch (activation) {
    case "sigmoid":
      return sigmoid;
    case "tanh":
      return tanh;
    case "relu":
      return relu;
    case "lRelu":
      return lRelu;
    case "linear":
      return linear;
  }
}
