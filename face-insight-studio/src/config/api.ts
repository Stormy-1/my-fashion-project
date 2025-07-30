// API Configuration for both development and production environments

const isDevelopment = import.meta.env.DEV;

// API Base URLs
const API_ENDPOINTS = {
  development: "http://localhost:5000",
  production: "https://fashion-recommendation-system-2-2pyh.onrender.com"
};

// Get the appropriate API base URL based on environment
export const getApiBaseUrl = (): string => {
  return isDevelopment ? API_ENDPOINTS.development : API_ENDPOINTS.production;
};

// API endpoint paths
export const API_PATHS = {
  recommend: "/api/recommend",
  cameraCapture: "/api/camera-capture",
  health: "/api/health"
};

// Helper function to build full API URLs
export const buildApiUrl = (path: string): string => {
  return `${getApiBaseUrl()}${path}`;
};

// Export specific API URLs for easy use
export const API_URLS = {
  recommend: buildApiUrl(API_PATHS.recommend),
  cameraCapture: buildApiUrl(API_PATHS.cameraCapture),
  health: buildApiUrl(API_PATHS.health)
};

// Log current environment and API base URL
console.log(`🌍 Environment: ${isDevelopment ? 'Development' : 'Production'}`);
console.log(`🔗 API Base URL: ${getApiBaseUrl()}`);
