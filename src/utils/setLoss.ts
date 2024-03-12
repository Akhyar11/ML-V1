import { Cost } from "../@types/type";
import MeanSquerError from "../cost/mse";

export default function setLoss(cost: Cost) {
  switch (cost) {
    case "mse":
      return MeanSquerError;
  }
}
