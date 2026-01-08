import axios from "axios";

const API = axios.create({
  baseURL: import.meta.env.VITE_BACKEND_URL,
});

export const predictImage = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await API.post("/predict", formData);
  return response.data;
};
