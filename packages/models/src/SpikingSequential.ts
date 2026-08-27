import { Matrix, mj } from "@oxide-js/core";
import { Sequential } from "./Sequential.js";
import { ForwardOptions } from "@oxide-js/layers";
import type { ModelConfig } from "./types.js";
import { poolerDistillationNativeWrapper, isNativeAvailable } from "@oxide-js/spiking";

export class SpikingSequential extends Sequential {
  
  constructor(layers: any[] = [], config?: ModelConfig) {
    super(layers, config);
  }

  /**
   * Forward pass khusus SNN yang menangani Layer BPTT secara temporal.
   */
  public override forward(inputs: Matrix, optionsOrTraining: ForwardOptions | boolean = this.training): Matrix {
    this.assertNotEmpty();

    const options: ForwardOptions =
      typeof optionsOrTraining === "boolean"
        ? { training: optionsOrTraining }
        : optionsOrTraining;

    if (!this.isBuilt) {
      this.build(inputs._shape);
    }

    let output = inputs;
    const batchSeq = inputs._shape[0];
    const actualLengths = options.maskLengths as number[] | undefined;

    for (const layer of this.layers) {
      if (layer.constructor.name === "SpikingDenseBPTT") {
        // Handle BPTT layer temporally
        const bpttLayer = layer as any;
        const maxSeqLength = actualLengths && actualLengths.length > 0 ? Math.max(...actualLengths) : 0;
        
        if (maxSeqLength === 0 || !actualLengths) {
            throw new Error("[SpikingSequential] options.maskLengths (actualLengths) diperlukan untuk menentukan jumlah timestep BPTT.");
        }

        const batchSize = actualLengths.length;
        const seqLen = batchSeq / batchSize;

        bpttLayer.resetSequence(maxSeqLength); // max_actual timesteps
        
        const inFeatures = output._shape[output._shape.length - 1];
        // Buat penampung sementara
        const finalOutData = new Float32Array(batchSize * bpttLayer.units);
        
        for (let t = 0; t < maxSeqLength; t++) {
            const stepInputData = new Float32Array(batchSize * inFeatures);
            for (let b = 0; b < batchSize; b++) {
                if (t < actualLengths![b]) {
                    const baseIdx = (b * seqLen + t) * inFeatures;
                    for (let i = 0; i < inFeatures; i++) {
                        stepInputData[b * inFeatures + i] = output._data[baseIdx + i];
                    }
                }
            }
            const stepMatrix = Matrix.fromFlat(stepInputData, [batchSize, inFeatures]);
            bpttLayer.computeStep(stepMatrix, t);
            
            // Akumulasi output (pooler) dari historyPotentials (sebelum reset)
            const potAtT = bpttLayer.historyPotentials[t]._data;
            for (let b = 0; b < batchSize; b++) {
                if (t < actualLengths![b]) {
                    const offset = b * bpttLayer.units;
                    for (let i = 0; i < bpttLayer.units; i++) {
                        finalOutData[offset + i] += potAtT[offset + i];
                    }
                }
            }
        }
        
        // Normalize final output
        const normalizedData = new Float32Array(batchSize * bpttLayer.units);
        for (let b = 0; b < batchSize; b++) {
            const offset = b * bpttLayer.units;
            let sum = 0;
            for (let i = 0; i < bpttLayer.units; i++) sum += finalOutData[offset + i];
            const mean = sum / bpttLayer.units;
            
            let sumSq = 0;
            for (let i = 0; i < bpttLayer.units; i++) {
                const centered = finalOutData[offset + i] - mean;
                sumSq += centered * centered;
            }
            const norm = Math.max(Math.sqrt(sumSq), 1e-8);
            for (let i = 0; i < bpttLayer.units; i++) {
                normalizedData[offset + i] = (finalOutData[offset + i] - mean) / norm;
            }
        }
        
        output = Matrix.fromFlat(normalizedData, [batchSize, bpttLayer.units]);
      } else {
        // Standard non-temporal forward (Embedding, SelfAttention)
        output = layer.forward(output, options);
      }
    }

    this.outputShape = [...output._shape];
    return output;
  }

  /**
   * Metode khusus untuk Knowledge Distillation training step (menggantikan trainStepDistill di Rust).
   * Implementasi Pooler Distillation di JS.
   */
  public trainStepDistill(texts: string[], targets: number[], margin: number, tokenizer: any, maxSeqLength: number, learningRate: number = 0.01): number {
      const batchSize = texts.length;
      const numPairs = Math.floor(batchSize / 2);
      
      const tokenizedBatch: number[] = [];
      const actualLengths: number[] = new Array(batchSize);

      // Tokenisasi
      for (let b = 0; b < batchSize; b++) {
          const tokens = tokenizer.encode(texts[b].toLowerCase());
          actualLengths[b] = Math.min(tokens.length, maxSeqLength);
          let sliced = tokens.slice(0, maxSeqLength);
          while (sliced.length < maxSeqLength) sliced.push(0); // Zero padding
          tokenizedBatch.push(...sliced);
      }
      
      const inputMatrix = Matrix.fromFlat(new Float32Array(tokenizedBatch), [batchSize * maxSeqLength, 1]); // Shape embedding dim 1
      
      // 1. Forward Pass
      const outputMatrix = this.forward(inputMatrix, { training: true, maskLengths: actualLengths });
      
      const bpttLayer = this.layers.find(l => l.constructor.name === "SpikingDenseBPTT") as any;
      const attentionLayer = this.layers.find(l => l.constructor.name === "SpikingSelfAttention") as any;
      const embeddingLayer = this.layers.find(l => l.constructor.name === "SpikingEmbedding") as any;

      if (!bpttLayer || !embeddingLayer) {
          throw new Error("[SpikingSequential] Model harus memiliki setidaknya SpikingEmbedding dan SpikingDenseBPTT");
      }

      const units = bpttLayer.units;
      const d_model = embeddingLayer.outputDim;
      
      // 2. Pooler Distillation Loss & Gradients
      const outData = outputMatrix._data;
      const errorFinalData = new Float32Array(batchSize * units);
      let totalLoss = 0;
      
      // Hitung total dari outData (jika 0 berarti spike mati, loss tidak akan menyusut)
      let sumOut = 0;
      for (let i = 0; i < outData.length; i++) sumOut += outData[i];
      if (sumOut === 0) {
          console.warn(`[WARNING] output pooler berisikan SEMUA NOL! (Dead Network)`);
      }

      if (isNativeAvailable()) {
          totalLoss = poolerDistillationNativeWrapper(
              outData,
              errorFinalData,
              numPairs,
              units,
              margin,
              new Float32Array(targets)
          );
      } else {
          for (let p = 0; p < numPairs; p++) {
              const a = p * 2;
              const b = p * 2 + 1;
              const offsetA = a * units;
              const offsetB = b * units;
              
              let sim = 0;
              for (let i = 0; i < units; i++) {
                  sim += outData[offsetA + i] * outData[offsetB + i];
              }
              
              const targetSim = targets[p];
              const diff = sim - targetSim;
              
              if (Math.abs(diff) > margin) {
                  totalLoss += 0.5 * diff * diff;
                  for (let i = 0; i < units; i++) {
                      errorFinalData[offsetA + i] = diff * outData[offsetB + i];
                      errorFinalData[offsetB + i] = diff * outData[offsetA + i];
                  }
              }
          }
      }

      // 3. Backward Pass BPTT
      // Karena BPTT menerima error dalam bentuk sequence (t_steps), tapi di Distillation error ada di output akhir pooler, 
      // kita berikan errorFinalData di akhir timestep (t = max_actual - 1) dan 0 di timestep lain.
      const maxActual = Math.max(...actualLengths);
      const errorSequence: Matrix[] = [];
      for (let t = 0; t < maxActual; t++) {
          // Gradient is distributed equally across all timesteps because pooler is a sum
          errorSequence.push(Matrix.fromFlat(errorFinalData, [batchSize, units]));
      }
      
      // We don't have a specific `B` matrix for distillation (loss comes directly from here)
      bpttLayer.learnThroughTime(errorSequence, undefined, learningRate);

      // 4. Backward Pass Embeddings dan Attention (E2E Spiking Routing)
      // Kita membutuhkan akses ke kernel BPTT untuk mendistribusikan error mundur ke Attention/Embedding
      const errToEmbed = new Float32Array(batchSize * maxSeqLength * d_model);
      const bpttKernel = bpttLayer.kernel._data; // [inFeatures, units]
      // Hitung Gradient Error untuk masuk ke BPTT / Attention Layer
      // bpttLayer kernel shape = [inFeatures, units]
      const bpttInFeatures = bpttLayer.kernel!._shape[0];
      const bpttErrInput = new Float32Array(batchSize * bpttInFeatures);
      for (let b = 0; b < batchSize; b++) {
          const errOffset = b * units;
          const inOffset = b * bpttInFeatures;
          for (let i = 0; i < bpttInFeatures; i++) {
              let sum = 0;
              for (let j = 0; j < units; j++) {
                  sum += bpttKernel[i * units + j] * errorFinalData[errOffset + j];
              }
              bpttErrInput[inOffset + i] = sum;
          }
      }

      const embGradientSeq = new Float32Array(batchSize * maxSeqLength * d_model);
      const attGradientSeq = new Float32Array(batchSize * maxSeqLength * d_model);
      
      const embSpikes = embeddingLayer.lastSpikes?._data;
      const attSpikes = attentionLayer ? attentionLayer.lastSpikes?._data : null;

      for (let b = 0; b < batchSize; b++) {
          for (let s = 0; s < maxSeqLength; s++) {
              if (s < actualLengths[b]) {
                  const seqOffset = (b * maxSeqLength + s) * d_model;
                  const poolerOffset = b * bpttInFeatures;
                  for (let i = 0; i < d_model; i++) {
                      const err = bpttErrInput[poolerOffset + i]; 
                      
                      if (attentionLayer && attSpikes) {
                          const spk1 = embSpikes ? (embSpikes[seqOffset + i] > 0.5 ? 1.0 : 0.0) : 0.0;
                          const attVal = attSpikes[seqOffset + i] > 0.5 ? 1.0 : 0.0;
                          
                          embGradientSeq[seqOffset + i] = attVal > 0.5 ? -err : err;
                          attGradientSeq[seqOffset + i] = spk1 > 0.5 ? -err : err;
                      } else {
                          embGradientSeq[seqOffset + i] = err;
                      }
                  }
              }
          }
      }

      embeddingLayer.learnEmbedding(Matrix.fromFlat(embGradientSeq, [batchSize * maxSeqLength, d_model]), undefined, learningRate);
      if (attentionLayer) {
          attentionLayer.learnAttention(Matrix.fromFlat(attGradientSeq, [batchSize * maxSeqLength, d_model]), learningRate, { maskLengths: actualLengths });
      }

      return totalLoss;
  }
}
