import { useState } from "react";
import { predictImage } from "../services/api";

function ImageUpload() {
  const [result, setResult] = useState(null);

  const handleChange = async (e) => {
    const file = e.target.files[0];
    const res = await predictImage(file);
    setResult(res);
  };

  return (
    <div>
      <input type="file" onChange={handleChange} />
      {result && (
        <div>
          <p>Prediction: {result.prediction}</p>
          <p>Confidence: {result.confidence}</p>
        </div>
      )}
    </div>
  );
}

export default ImageUpload;
