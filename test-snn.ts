import { BPETokenizer } from "./packages/core/src/tokenizer/bpe.js";
import { SpikingEmbedding } from "./packages/spiking/src/layers/SpikingEmbedding.js";
import { SpikingSelfAttention } from "./packages/spiking/src/layers/SpikingSelfAttention.js";
import { SpikingDenseBPTT } from "./packages/spiking/src/layers/SpikingDenseBPTT.js";
import { SpikingSequential } from "./packages/models/src/SpikingSequential.js";

async function main() {
    console.log("Inisialisasi Tokenizer...");
    const tokenizer = new BPETokenizer();
    
    // Setup model parameter
    const vocabSize = 1000;
    const dModel = 64;
    const maxSeqLength = 10;
    
    console.log("Membangun SpikingSequential Model...");
    const model = new SpikingSequential();
    
    // Tambahkan layer
    model.add(new SpikingEmbedding({
        inputDim: vocabSize,
        outputDim: dModel
    }));
    
    model.add(new SpikingSelfAttention({
        d_model: dModel,
        sequenceLength: maxSeqLength
    }));
    
    model.add(new SpikingDenseBPTT({
        units: dModel, // units sama dengan inFeatures
        useBias: false
    }));

    // Data Dummy
    const texts = [
        "hello world",
        "spiking neural network"
    ];
    const targets = [0.8]; // Target kemiripan

    console.log("Memulai proses trainStepDistill...");
    const margin = 0.5;
    const learningRate = 0.01;
    
    try {
        const loss = model.trainStepDistill(texts, targets, margin, tokenizer, maxSeqLength, learningRate);
        console.log(`Berhasil! Loss: ${loss}`);
        model.summary();
    } catch (e) {
        console.error("Gagal saat training:", e);
    }
}

main();
