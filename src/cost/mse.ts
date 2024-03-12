import mj from "../math";
import Matrix from "../matrix";

export default function MeanSquerError(
  yTrue: Matrix,
  yPred: Matrix
): [number, Matrix] {
  const result = mj.mean(mj.map(mj.sub(yTrue, yPred), (v) => v ** 2));
  const dResult = mj.sub(yPred, yTrue);
  return [result, dResult];
}
