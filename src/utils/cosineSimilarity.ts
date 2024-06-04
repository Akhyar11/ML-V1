import mj from "../math";
import Matrix from "../matrix";

export default function cosineSimilarity(a: Matrix, b: Matrix) {
  const flatA = mj.flatten(a);
  const flatB = mj.flatten(b);
  const magnitudeA = mj.norm(flatA);
  const magnitudeB = mj.norm(flatB);
  const dotProduct = mj.dotProduct(flatA, mj.transpose(flatB));
  return dotProduct._value[0][0] / (magnitudeA * magnitudeB);
}
