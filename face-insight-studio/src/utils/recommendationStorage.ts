// Utility functions for storing and retrieving recommendation history

export interface StoredRecommendation {
  id: string;
  timestamp: number;
  date: string;
  recommendations: any[];
  textRecommendations: any[];
  uploadedImage: string | null;
  userInputs?: {
    height?: string;
    weight?: string;
    occasion?: string;
    budget?: string;
  };
}

const STORAGE_KEY = 'fashion_recommendations_history';

export const saveRecommendation = (
  recommendations: any[],
  textRecommendations: any[],
  uploadedImage: string | null,
  userInputs?: any
): string => {
  try {
    const existingHistory = getRecommendationHistory();
    const timestamp = Date.now();
    const date = new Date(timestamp).toLocaleString();
    const id = `rec_${timestamp}`;

    const newRecommendation: StoredRecommendation = {
      id,
      timestamp,
      date,
      recommendations,
      textRecommendations,
      uploadedImage,
      userInputs
    };

    const updatedHistory = [newRecommendation, ...existingHistory];
    
    // Keep only the last 20 recommendations to avoid storage bloat
    const limitedHistory = updatedHistory.slice(0, 20);
    
    localStorage.setItem(STORAGE_KEY, JSON.stringify(limitedHistory));
    return id;
  } catch (error) {
    console.error('Error saving recommendation:', error);
    return '';
  }
};

export const getRecommendationHistory = (): StoredRecommendation[] => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];
    
    const history = JSON.parse(stored);
    // Sort by timestamp descending (newest first)
    return history.sort((a: StoredRecommendation, b: StoredRecommendation) => b.timestamp - a.timestamp);
  } catch (error) {
    console.error('Error retrieving recommendation history:', error);
    return [];
  }
};

export const getRecommendationById = (id: string): StoredRecommendation | null => {
  try {
    const history = getRecommendationHistory();
    return history.find(rec => rec.id === id) || null;
  } catch (error) {
    console.error('Error retrieving recommendation by ID:', error);
    return null;
  }
};

export const clearRecommendationHistory = (): void => {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (error) {
    console.error('Error clearing recommendation history:', error);
  }
};

export const deleteRecommendation = (id: string): void => {
  try {
    const history = getRecommendationHistory();
    const updatedHistory = history.filter(rec => rec.id !== id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedHistory));
  } catch (error) {
    console.error('Error deleting recommendation:', error);
  }
};

export const getLatestRecommendation = (): StoredRecommendation | null => {
  try {
    const history = getRecommendationHistory();
    return history.length > 0 ? history[0] : null;
  } catch (error) {
    console.error('Error getting latest recommendation:', error);
    return null;
  }
};
