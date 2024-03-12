import { readFileSync } from "fs";
import Sequential, { SequentialLayers } from "./sequential";
import { Matrix } from "../@types/type";
import { setLayers } from "../utils";

export default class DimentionalityReduction extends Sequential {
  layersEncode: SequentialLayers = [];
  layersDecode: SequentialLayers = [];
  constructor({ layers = [] }: { layers?: SequentialLayers } = {}) {
    super({ layers });
    let isLayerEncode = true;
    for (let layer of this.layers) {
      if (isLayerEncode) {
        this.layersEncode.push(layer);
        if (layer.status === "outputReduction") isLayerEncode = false;
      } else {
        this.layersDecode.push(layer);
      }
    }
  }

  load(path: string): void {
    const dataJson = readFileSync(path, "utf-8");
    const data = JSON.parse(dataJson);
    this.layers = setLayers(data);
    let isLayerEncode = true;
    for (let layer of this.layers) {
      if (isLayerEncode) {
        this.layersEncode.push(layer);
        if (layer.status === "outputReduction") isLayerEncode = false;
      } else {
        this.layersDecode.push(layer);
      }
    }
  }

  encode(x: Matrix): Matrix {
    let input = x;
    for (let layer of this.layersEncode) {
      input = layer.forward(input);
    }
    return input;
  }

  decode(enc: Matrix) {
    let input = enc;
    for (let layer of this.layersDecode) {
      input = layer.forward(input);
    }
    return input;
  }
}
