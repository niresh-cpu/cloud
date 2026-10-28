import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

const apiClient = axios.create({
  baseURL: API_BASE_URL ? `${API_BASE_URL}/api` : '/api',
  timeout: 120000,
});

export const healthCheck = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};

export const predictDisease = async (file, crop = null) => {
  const formData = new FormData();
  formData.append('file', file);
  if (crop && crop !== 'auto') {
    formData.append('crop', crop);
  }
  // Do NOT manually set Content-Type so Axios/browser sets boundary correctly
  const response = await apiClient.post('/predict', formData);
  return response.data;
};

export const validateImageFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  // Do NOT manually set Content-Type so Axios/browser sets boundary correctly
  const response = await apiClient.post('/validate-image', formData);
  return response.data;
};

export const getSupportedClasses = async () => {
  const response = await apiClient.get('/classes');
  return response.data;
};

export const getSupportedCrops = async () => {
  const response = await apiClient.get('/crops');
  return response.data;
};

export const getAllDiseases = async () => {
  const response = await apiClient.get('/diseases');
  return response.data;
};

export const getDiseaseDetail = async (diseaseId) => {
  const response = await apiClient.get(`/disease-info/${diseaseId}`);
  return response.data;
};

export const getModelInfo = async () => {
  const response = await apiClient.get('/model-info');
  return response.data;
};

export const getModelMetrics = async () => {
  const response = await apiClient.get('/model-metrics');
  return response.data;
};

export const getPerformanceDashboard = async () => {
  const response = await apiClient.get('/performance');
  return response.data;
};

export const getChartUrl = (chartName) => `${API_BASE_URL}/api/charts/${chartName}`;

export default apiClient;
