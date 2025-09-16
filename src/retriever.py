import numpy as np
import pandas as pd
import faiss
import pickle
import os
from typing import List, Tuple, Optional, Dict
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductRetriever:
    """
    FAISS-based similarity search for product recommendations
    """
    
    def __init__(self, embeddings: np.ndarray, df: pd.DataFrame, 
                 index_type: str = "IndexFlatIP"):
        """
        Initialize the retriever with embeddings and product data
        
        Args:
            embeddings: Product embeddings matrix (n_products, embedding_dim)
            df: DataFrame containing product information
            index_type: Type of FAISS index to use
        """
        self.embeddings = embeddings.astype(np.float32)
        self.df = df.reset_index(drop=True)
        self.embedding_dim = embeddings.shape[1]
        self.index_type = index_type
        self.index = None
        
        # Normalize embeddings for cosine similarity
        self.normalized_embeddings = self._normalize_embeddings(self.embeddings)
        
        # Build the index
        self._build_index()
        
        logger.info(f"Retriever initialized with {len(df)} products and {self.embedding_dim}-dim embeddings")
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    def _build_index(self):
        """Build FAISS index for fast similarity search"""
        if self.index_type == "IndexFlatIP":
            # Inner product index (good for normalized vectors = cosine similarity)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "IndexFlatL2":
            # L2 distance index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "IndexIVFFlat":
            # IVF index for larger datasets
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            nlist = min(100, len(self.embeddings) // 10)  # number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            
            # Train the index
            self.index.train(self.normalized_embeddings)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Add embeddings to index
        self.index.add(self.normalized_embeddings)
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
    
    def search_similar_products(self, query_embedding: np.ndarray, 
                              k: int = 10, 
                              exclude_indices: Optional[List[int]] = None) -> Tuple[List[float], List[int]]:
        """
        Search for k most similar products
        
        Args:
            query_embedding: Query embedding vector
            k: Number of similar products to return
            exclude_indices: Product indices to exclude from results
        
        Returns:
            Tuple of (similarities, indices)
        """
        # Normalize query embedding
        query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_normalized = self._normalize_embeddings(query_embedding)
        
        # Search with extra results in case we need to exclude some
        search_k = k + (len(exclude_indices) if exclude_indices else 0) + 10
        search_k = min(search_k, self.index.ntotal)
        
        # Perform search
        similarities, indices = self.index.search(query_normalized, search_k)
        
        # Convert to lists and filter out excluded indices
        similarities = similarities[0].tolist()
        indices = indices[0].tolist()
        
        if exclude_indices:
            filtered_results = []
            for sim, idx in zip(similarities, indices):
                if idx not in exclude_indices and len(filtered_results) < k:
                    filtered_results.append((sim, idx))
            similarities, indices = zip(*filtered_results) if filtered_results else ([], [])
            similarities, indices = list(similarities), list(indices)
        else:
            similarities = similarities[:k]
            indices = indices[:k]
        
        return similarities, indices
    
    def get_similar_products_info(self, query_embedding: np.ndarray, 
                                 k: int = 10, 
                                 exclude_product_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get detailed information about similar products
        
        Args:
            query_embedding: Query embedding vector
            k: Number of similar products to return
            exclude_product_ids: Product IDs to exclude from results
        
        Returns:
            DataFrame with similar products information
        """
        # Convert product IDs to indices if provided
        exclude_indices = None
        if exclude_product_ids:
            exclude_indices = []
            for product_id in exclude_product_ids:
                matching_indices = self.df[self.df['product_id'] == product_id].index.tolist()
                exclude_indices.extend(matching_indices)
        
        # Search for similar products
        similarities, indices = self.search_similar_products(
            query_embedding, k, exclude_indices
        )
        
        if not indices:
            return pd.DataFrame()
        
        # Get product information
        similar_products = self.df.iloc[indices].copy()
        similar_products['similarity_score'] = similarities
        
        # Reorder columns for better presentation
        columns_order = ['product_id', 'product_name', 'category', 'discounted_price', 
                        'actual_price', 'rating', 'rating_count', 'similarity_score',
                        'about_product', 'product_link', 'img_link']
        
        available_columns = [col for col in columns_order if col in similar_products.columns]
        similar_products = similar_products[available_columns]
        
        return similar_products
    
    def search_by_category(self, category: str, k: int = 10) -> pd.DataFrame:
        """
        Search products within a specific category
        
        Args:
            category: Category to search in
            k: Number of products to return
        
        Returns:
            DataFrame with products from the category
        """
        category_products = self.df[
            self.df['category'].str.contains(category, case=False, na=False)
        ]
        
        if len(category_products) == 0:
            logger.warning(f"No products found in category: {category}")
            return pd.DataFrame()
        
        # Sort by rating and return top k
        return category_products.nlargest(k, 'rating')
    
    def search_by_price_range(self, min_price: float, max_price: float, 
                            k: int = 10) -> pd.DataFrame:
        """
        Search products within a price range
        
        Args:
            min_price: Minimum price
            max_price: Maximum price
            k: Number of products to return
        
        Returns:
            DataFrame with products in the price range
        """
        price_filtered = self.df[
            (self.df['discounted_price'] >= min_price) & 
            (self.df['discounted_price'] <= max_price)
        ]
        
        if len(price_filtered) == 0:
            logger.warning(f"No products found in price range: ${min_price}-${max_price}")
            return pd.DataFrame()
        
        # Sort by rating and return top k
        return price_filtered.nlargest(k, 'rating')
    
    def hybrid_search(self, query_embedding: np.ndarray, 
                     category_filter: Optional[str] = None,
                     price_range: Optional[Tuple[float, float]] = None,
                     min_rating: Optional[float] = None,
                     k: int = 10) -> pd.DataFrame:
        """
        Hybrid search with filters
        
        Args:
            query_embedding: Query embedding vector
            category_filter: Category to filter by
            price_range: (min_price, max_price) tuple
            min_rating: Minimum rating filter
            k: Number of products to return
        
        Returns:
            DataFrame with filtered similar products
        """
        # Start with all products
        filtered_df = self.df.copy()
        
        # Apply filters
        if category_filter:
            filtered_df = filtered_df[
                filtered_df['category'].str.contains(category_filter, case=False, na=False)
            ]
        
        if price_range:
            min_price, max_price = price_range
            filtered_df = filtered_df[
                (filtered_df['discounted_price'] >= min_price) & 
                (filtered_df['discounted_price'] <= max_price)
            ]
        
        if min_rating:
            filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
        
        if len(filtered_df) == 0:
            logger.warning("No products found matching the filters")
            return pd.DataFrame()
        
        # Get indices of filtered products
        filtered_indices = filtered_df.index.tolist()
        
        # Get embeddings for filtered products
        filtered_embeddings = self.normalized_embeddings[filtered_indices]
        
        # Calculate similarities
        query_normalized = self._normalize_embeddings(query_embedding.reshape(1, -1))
        similarities = cosine_similarity(query_normalized, filtered_embeddings)[0]
        
        # Sort by similarity and get top k
        sorted_indices = np.argsort(similarities)[::-1][:k]
        top_indices = [filtered_indices[i] for i in sorted_indices]
        top_similarities = [similarities[i] for i in sorted_indices]
        
        # Get product information
        result_df = self.df.iloc[top_indices].copy()
        result_df['similarity_score'] = top_similarities
        
        return result_df
    
    def save_index(self, filepath: str):
        """Save FAISS index to file"""
        try:
            faiss.write_index(self.index, filepath)
            logger.info(f"Index saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise
    
    def load_index(self, filepath: str):
        """Load FAISS index from file"""
        try:
            self.index = faiss.read_index(filepath)
            logger.info(f"Index loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise
    
    def get_product_by_id(self, product_id: str) -> pd.Series:
        """Get product information by product ID"""
        product = self.df[self.df['product_id'] == product_id]
        if len(product) == 0:
            raise ValueError(f"Product with ID {product_id} not found")
        return product.iloc[0]
    
    def get_random_products(self, k: int = 10) -> pd.DataFrame:
        """Get random products for exploration"""
        return self.df.sample(n=min(k, len(self.df)))

class CategoryRetriever:
    """
    Category-based retrieval for content-based filtering
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.category_groups = self._build_category_groups()
        
    def _build_category_groups(self) -> Dict[str, List[int]]:
        """Build mapping of categories to product indices"""
        category_groups = {}
        for idx, category in enumerate(self.df['category']):
            if pd.notna(category):
                category = str(category).lower().strip()
                if category not in category_groups:
                    category_groups[category] = []
                category_groups[category].append(idx)
        
        logger.info(f"Built category groups for {len(category_groups)} categories")
        return category_groups
    
    def get_similar_category_products(self, product_idx: int, k: int = 10) -> pd.DataFrame:
        """Get products from the same category"""
        product = self.df.iloc[product_idx]
        category = str(product['category']).lower().strip()
        
        if category in self.category_groups:
            similar_indices = [idx for idx in self.category_groups[category] 
                             if idx != product_idx]
            
            if similar_indices:
                # Get top k products by rating
                similar_products = self.df.iloc[similar_indices]
                return similar_products.nlargest(min(k, len(similar_products)), 'rating')
        
        return pd.DataFrame()

class PriceBasedRetriever:
    """
    Price-based retrieval for finding products in similar price ranges
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def get_similar_price_products(self, target_price: float, 
                                 price_tolerance: float = 0.2, 
                                 k: int = 10) -> pd.DataFrame:
        """
        Get products in similar price range
        
        Args:
            target_price: Target price
            price_tolerance: Price tolerance as percentage (0.2 = 20%)
            k: Number of products to return
        """
        min_price = target_price * (1 - price_tolerance)
        max_price = target_price * (1 + price_tolerance)
        
        similar_products = self.df[
            (self.df['discounted_price'] >= min_price) & 
            (self.df['discounted_price'] <= max_price)
        ]
        
        if len(similar_products) == 0:
            return pd.DataFrame()
        
        # Sort by rating and return top k
        return similar_products.nlargest(min(k, len(similar_products)), 'rating')

# Utility functions
def create_retriever_cache(embeddings: np.ndarray, df: pd.DataFrame, 
                          cache_dir: str = "cache") -> str:
    """Create and cache retriever components"""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Save retriever data
    retriever_data = {
        'embeddings': embeddings,
        'dataframe': df,
        'created_at': pd.Timestamp.now()
    }
    
    cache_file = os.path.join(cache_dir, "retriever_cache.pkl")
    
    with open(cache_file, 'wb') as f:
        pickle.dump(retriever_data, f)
    
    logger.info(f"Retriever cache saved to {cache_file}")
    return cache_file

def load_retriever_cache(cache_file: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load retriever components from cache"""
    with open(cache_file, 'rb') as f:
        retriever_data = pickle.load(f)
    
    return retriever_data['embeddings'], retriever_data['dataframe']

# Main function for testing
if __name__ == "__main__":
    # Test with sample data
    print("Testing ProductRetriever...")
    
    # Create sample embeddings and dataframe
    sample_embeddings = np.random.rand(100, 384).astype(np.float32)
    sample_data = {
        'product_id': [f'P{i:03d}' for i in range(100)],
        'product_name': [f'Product {i}' for i in range(100)],
        'category': ['Electronics', 'Clothing', 'Books'] * 33 + ['Electronics'],
        'discounted_price': np.random.uniform(10, 1000, 100),
        'actual_price': np.random.uniform(10, 1200, 100),
        'rating': np.random.uniform(1, 5, 100),
        'rating_count': np.random.randint(1, 1000, 100),
        'about_product': [f'Description for product {i}' for i in range(100)],
        'product_link': [f'https://example.com/product/{i}' for i in range(100)],
        'img_link': [f'https://example.com/img/{i}.jpg' for i in range(100)]
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    try:
        # Initialize retriever
        retriever = ProductRetriever(sample_embeddings, sample_df)
        
        # Test similarity search
        query_embedding = np.random.rand(384).astype(np.float32)
        similar_products = retriever.get_similar_products_info(query_embedding, k=5)
        
        print(f"Found {len(similar_products)} similar products")
        print(similar_products[['product_name', 'category', 'rating', 'similarity_score']])
        
        # Test category search
        category_products = retriever.search_by_category('Electronics', k=3)
        print(f"\nFound {len(category_products)} Electronics products")
        
        # Test price range search
        price_products = retriever.search_by_price_range(50, 200, k=3)
        print(f"\nFound {len(price_products)} products in $50-$200 range")
        
    except Exception as e:
        print(f"Error: {str(e)}")