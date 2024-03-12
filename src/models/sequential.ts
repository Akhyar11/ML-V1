import { readFileSync, writeFileSync } from "fs";
import { Layers, Matrix } from "../@types/type";
import { setLayers } from "../utils";
import { CompileDenseLayers, Dense, Convolution } from "../layers";

export type SequentialLayers = Layers[];

export default class Sequential {
  layers: SequentialLayers;
  loss = 0;
  constructor({ layers = [] }: { layers?: SequentialLayers } = {}) {
    this.layers = layers;
  }

  summary() {
    console.log("========== Model Info ==========");
    let totalParams = 0;
    for (let layer of this.layers) {
      console.log(`Layer name   : ${layer.name}`);
      console.log(`Layer input  : [${layer.inputShape}]`);
      console.log(`Layer output : [${layer.outputShape}]`);
      console.log(`Layer param  : ${layer.params}`);
      console.log("");
      totalParams += layer.params;
    }

    console.log("Total params =", totalParams);
    console.log("========== End Info ==========");
  }

  save(path: string) {
    const data = [];
    for (let layer of this.layers) {
      data.push(layer.save());
    }
    const dataJson = JSON.stringify(data);
    writeFileSync(path, dataJson);
  }

  load(path: string) {
    const dataJson = readFileSync(path, "utf-8");
    const data = JSON.parse(dataJson);
    this.layers = setLayers(data);
  }

  compile({
    alpha = 0.1,
    optimizer = "sgd",
    error = "mse",
  }: CompileDenseLayers) {
    for (let layer of this.layers) {
      if (layer instanceof Dense) {
        layer.compile({ alpha, optimizer, error });
      } else if (layer instanceof Convolution) {
        layer.compile({ alpha, optimizer, error });
      }
    }
  }

  forward(x: Matrix) {
    let input = x;
    for (let layer of this.layers) {
      input = layer.forward(input);
    }
  }

  backward(y: Matrix) {
    let err = y;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      err = this.layers[i].backward(y, err);
      if (this.layers[i].status === "output") this.loss = this.layers[i].loss;
    }
  }

  predict(x: Matrix) {
    let input = x;
    for (let layer of this.layers) {
      input = layer.forward(input);
    }
    return input;
  }

  fit(
    X: Matrix[],
    y: Matrix[],
    epochs: number,
    cb: (err: number) => any = (_) => {}
  ) {
    for (let i = 0; i < epochs; i++) {
      for (let j in X) {
        this.forward(X[j]);
        this.backward(y[j]);
      }
      cb(this.loss);
      if (this.loss < 0.01) {
        return 0;
      }
    }
  }
}
