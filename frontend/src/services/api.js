import axios from "axios";

const API = axios.create({
  baseURL: import.meta.env.VITE_BACKEND_URL || "http://localhost:8000/api/v1",
  timeout: 30000, // 30s — model inference can take a few seconds
});

/**
 * Send an image file for skin lesion analysis.
 * Returns: { prediction, confidence, prob_malignant, risk_level,
 *            recommendation, disclaimer, gradcam_image }
 */
export const predictImage = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await API.post("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

/**
 * Health check — useful to verify backend is reachable.
 */
export const healthCheck = async () => {
  const response = await API.get("/health");
  return response.data;
};
