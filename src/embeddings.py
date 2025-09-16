import os
import pandas as pd
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Optional, Dict
import time
from dotenv import load_dotenv
import pickle

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for product descriptions using Groq API and Sentence Transformers"""
    
    def __init__(self, groq_api_key: Optional[str] = None, use_sentence_transformer: bool = True):
        """
        Initialize embedding generator
        
        Args:
            groq_api_key: Groq API key (if None, will try to load from environment)
            use_sentence_transformer: Whether to use SentenceTransformer as fallback/primary
        """
        # Initialize Groq client
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if self.groq_api_key:
            self.groq_client = Groq(api_key=self.groq_api_key)
        else:
            logger.warning("No Groq API key found. Using SentenceTransformer only.")
            self.groq_client = None
            
        # Initialize SentenceTransformer as fallback/primary
        self.use_sentence_transformer = use_sentence_transformer
        if use_sentence_transformer:
            logger.info("Loading SentenceTransformer model...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer model loaded successfully")
        else:
            self.sentence_model = None
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text for embedding generation"""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and clean
        text = str(text)
        text = text.strip()
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Truncate if too long (important for API limits)
        if len(text) > 2000:  # Reasonable limit for embedding generation
            text = text[:2000] + "..."
            
        return text
    
    def generate_groq_embeddings(self, texts: List[str], model: str = "llama3-8b-8192") -> List[List[float]]:
        """
        Generate embeddings using Groq API (Note: This is a simulation since Groq doesn't have embedding endpoints)
        In practice, we'll use SentenceTransformer but this shows the structure for when Groq adds embedding support
        """
        if not self.groq_client:
            raise ValueError("Groq client not initialized. Please provide API key.")
        
        logger.info(f"Generating embeddings for {len(texts)} texts using Groq...")
        embeddings = []
        
        # Note: Groq doesn't currently have embedding endpoints, so we'll use text completion
        # to generate semantic representations and then use SentenceTransformer
        # This is a workaround - in practice, you'd use dedicated embedding APIs
        
        for i, text in enumerate(texts):
            try:
                # For now, we'll use SentenceTransformer since Groq doesn't have embeddings
                # But we keep this structure for when they add embedding support
                if self.sentence_model:
                    embedding = self.sentence_model.encode(text).tolist()
                    embeddings.append(embedding)
                else:
                    # Placeholder for future Groq embedding API
                    raise NotImplementedError("Groq embedding API not yet available")
                
                if i % 100 == 0:
                    logger.info(f"Processed {i}/{len(texts)} texts")
                    
            except Exception as e:
                logger.error(f"Error generating embedding for text {i}: {str(e)}")
                # Use zero vector as fallback
                if embeddings:
                    embedding_dim = len(embeddings[0])
                else:
                    embedding_dim = 384  # Default for all-MiniLM-L6-v2
                embeddings.append([0.0] * embedding_dim)
        
        return embeddings
    
    def generate_sentence_transformer_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using SentenceTransformer"""
        if not self.sentence_model:
            raise ValueError("SentenceTransformer model not initialized")
        
        logger.info(f"Generating embeddings for {len(texts)} texts using SentenceTransformer...")
        
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Generate embeddings in batches to manage memory
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i:i+batch_size]
            try:
                batch_embeddings = self.sentence_model.encode(batch, show_progress_bar=False)
                embeddings.append(batch_embeddings)
                
                if i % 320 == 0:  # Log every 10 batches
                    logger.info(f"Processed {min(i+batch_size, len(cleaned_texts))}/{len(cleaned_texts)} texts")
                    
            except Exception as e:
                logger.error(f"Error in batch {i}-{i+batch_size}: {str(e)}")
                # Create zero embeddings for this batch
                batch_embeddings = np.zeros((len(batch), 384))
                embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings)
        logger.info(f"Generated embeddings shape: {all_embeddings.shape}")
        
        return all_embeddings
    
    def generate_embeddings_for_dataframe(self, df: pd.DataFrame, 
                                        text_column: str = 'combined_text',
                                        use_groq: bool = False) -> np.ndarray:
        """
        Generate embeddings for all products in dataframe
        
        Args:
            df: DataFrame containing product data
            text_column: Column name containing text to embed
            use_groq: Whether to use Groq API (currently uses SentenceTransformer as fallback)
        
        Returns:
            numpy array of embeddings
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataframe")
        
        texts = df[text_column].tolist()
        
        if use_groq and self.groq_client:
            try:
                embeddings = self.generate_groq_embeddings(texts)
                return np.array(embeddings)
            except Exception as e:
                logger.warning(f"Groq embedding failed: {str(e)}. Falling back to SentenceTransformer")
                if self.sentence_model:
                    return self.generate_sentence_transformer_embeddings(texts)
                else:
                    raise
        else:
            return self.generate_sentence_transformer_embeddings(texts)
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to file"""
        try:
            np.save(filepath, embeddings)
            logger.info(f"Embeddings saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
            raise
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from file"""
        try:
            embeddings = np.load(filepath)
            logger.info(f"Embeddings loaded from {filepath}. Shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise
    
    def get_text_embedding(self, text: str, use_groq: bool = False) -> np.ndarray:
        """Get embedding for a single text"""
        cleaned_text = self.clean_text(text)
        
        if use_groq and self.groq_client:
            try:
                embeddings = self.generate_groq_embeddings([cleaned_text])
                return np.array(embeddings[0])
            except Exception as e:
                logger.warning(f"Groq embedding failed: {str(e)}. Using SentenceTransformer")
                
        if self.sentence_model:
            return self.sentence_model.encode([cleaned_text])[0]
        else:
            raise ValueError("No embedding model available")

# Utility function for creating and caching embeddings
def create_embeddings_cache(df: pd.DataFrame, cache_dir: str = "cache") -> str:
    """Create and cache embeddings for the dataset"""
    os.makedirs(cache_dir, exist_ok=True)
    
    embeddings_file = os.path.join(cache_dir, "product_embeddings.npy")
    metadata_file = os.path.join(cache_dir, "embeddings_metadata.pkl")
    
    # Check if embeddings already exist
    if os.path.exists(embeddings_file) and os.path.exists(metadata_file):
        logger.info("Loading existing embeddings from cache...")
        embeddings = np.load(embeddings_file)
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        # Verify cache is valid (same number of products)
        if len(embeddings) == len(df):
            logger.info("Using cached embeddings")
            return embeddings_file
        else:
            logger.info("Cache invalid, regenerating embeddings...")
    
    # Generate new embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.generate_embeddings_for_dataframe(df)
    
    # Save embeddings and metadata
    np.save(embeddings_file, embeddings)
    
    metadata = {
        'num_products': len(df),
        'embedding_dim': embeddings.shape[1],
        'created_at': pd.Timestamp.now(),
        'model_used': 'all-MiniLM-L6-v2'
    }
    
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info(f"Embeddings cached to {embeddings_file}")
    return embeddings_file

# Main function for testing
if __name__ == "__main__":
    # Test embedding generation
    sample_texts = [
        "High-quality wireless headphones with noise cancellation",
        "Comfortable running shoes for daily exercise",
        "Smartphone with excellent camera quality"
    ]
    
    generator = EmbeddingGenerator()
    
    try:
        # Test sentence transformer embeddings
        embeddings = generator.generate_sentence_transformer_embeddings(sample_texts)
        print(f"Generated embeddings shape: {embeddings.shape}")
        print(f"Sample embedding (first 10 values): {embeddings[0][:10]}")
        
    except Exception as e:
        print(f"Error: {str(e)}")