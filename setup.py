#!/usr/bin/env python3
"""
Setup script for Amazon GenAI Recommender System
Run this script to set up the project environment and test the installation.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

def create_directory_structure():
    """Create the required directory structure"""
    directories = [
        'data',
        'cache', 
        'logs',
        'src'
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created/verified directory: {directory}")

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\nChecking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'streamlit', 'groq', 'sentence_transformers',
        'faiss', 'scikit-learn', 'python-dotenv', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'faiss':
                import faiss
            else:
                __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def create_sample_dataset():
    """Create a sample dataset if no dataset exists"""
    data_file = Path('data/amazon_products.csv')
    
    if data_file.exists():
        print(f"✅ Dataset already exists: {data_file}")
        return True
    
    print("\n📊 Creating sample dataset...")
    
    # Generate sample data
    np.random.seed(42)
    n_products = 1000
    n_users = 100
    
    categories = [
        'Electronics', 'Clothing', 'Books', 'Home & Kitchen', 'Sports & Outdoors',
        'Beauty & Personal Care', 'Automotive', 'Health & Household', 'Toys & Games',
        'Office Products', 'Pet Supplies', 'Garden & Outdoor'
    ]
    
    # Product data
    products_data = []
    
    for i in range(n_products):
        category = np.random.choice(categories)
        
        # Generate price data
        base_price = np.random.uniform(10, 500)
        discount_pct = np.random.uniform(0, 50)
        discounted_price = base_price * (1 - discount_pct / 100)
        
        # Generate ratings
        rating = np.random.beta(5, 1.5) * 4 + 1  # Bias towards higher ratings
        rating_count = np.random.poisson(50) + 1
        
        # Generate descriptions based on category
        descriptions = {
            'Electronics': [
                'High-quality electronic device with advanced features',
                'Latest technology with excellent performance',
                'Durable and reliable electronic product',
                'User-friendly design with modern functionality'
            ],
            'Clothing': [
                'Comfortable and stylish clothing item',
                'High-quality fabric with excellent fit',
                'Trendy design suitable for various occasions',
                'Durable material with fashionable appearance'
            ],
            'Books': [
                'Engaging and informative reading material',
                'Well-written content with valuable insights',
                'Comprehensive guide with practical information',
                'Entertaining story with compelling characters'
            ],
            'Home & Kitchen': [
                'Essential household item for daily use',
                'Practical kitchen tool with excellent functionality',
                'Durable home accessory with modern design',
                'Convenient solution for home organization'
            ]
        }
        
        default_desc = 'High-quality product with excellent features and great value'
        about_product = np.random.choice(
            descriptions.get(category, [default_desc])
        )
        
        # Generate user reviews
        user_id = f"U{np.random.randint(1, n_users+1):04d}"
        review_titles = [
            'Great product!', 'Highly recommended', 'Good value for money',
            'Excellent quality', 'Perfect for my needs', 'Amazing product',
            'Worth the purchase', 'Outstanding performance'
        ]
        
        review_contents = [
            'This product exceeded my expectations. Great quality and fast delivery.',
            'Very satisfied with this purchase. Would buy again.',
            'Excellent product with good value for money. Highly recommend.',
            'Perfect quality and exactly what I was looking for.',
            'Great experience overall. Product works as described.',
            'Outstanding quality and performance. Very happy with the purchase.',
            'Good product with reliable performance. Satisfied customer.',
            'Amazing product that meets all my requirements perfectly.'
        ]
        
        product = {
            'product_id': f'P{i+1:06d}',
            'product_name': f'{category} Product {i+1}',
            'category': category,
            'discounted_price': round(discounted_price, 2),
            'actual_price': round(base_price, 2),
            'discount_percentage': round(discount_pct, 1),
            'rating': round(rating, 1),
            'rating_count': rating_count,
            'about_product': about_product,
            'user_id': user_id,
            'user_name': f'User{np.random.randint(1, n_users+1)}',
            'review_id': f'R{i+1:06d}',
            'review_title': np.random.choice(review_titles),
            'review_content': np.random.choice(review_contents),
            'img_link': f'https://example.com/images/product_{i+1}.jpg',
            'product_link': f'https://example.com/products/{i+1}'
        }
        
        products_data.append(product)
    
    # Create DataFrame and save
    df = pd.DataFrame(products_data)
    df.to_csv(data_file, index=False)
    
    print(f"✅ Sample dataset created: {data_file}")
    print(f"   - Products: {len(df)}")
    print(f"   - Categories: {df['category'].nunique()}")
    print(f"   - Price range: ${df['discounted_price'].min():.2f} - ${df['discounted_price'].max():.2f}")
    
    return True

def test_system_components():
    """Test individual system components"""
    print("\n🧪 Testing system components...")
    
    try:
        # Test data loader
        sys.path.append('src')
        from data_loader import DataLoader
        
        loader = DataLoader('data/amazon_products.csv')
        df = loader.load_data()
        clean_df = loader.clean_data(df)
        
        print(f"✅ Data Loader: Loaded {len(clean_df)} products")
        
        # Test embeddings (without API key)
        from embeddings import EmbeddingGenerator
        
        generator = EmbeddingGenerator(use_sentence_transformer=True)
        sample_embeddings = generator.generate_sentence_transformer_embeddings([
            "sample product description",
            "another product text"
        ])
        
        print(f"✅ Embeddings: Generated embeddings with shape {sample_embeddings.shape}")
        
        # Test retriever
        from retriever import ProductRetriever
        
        # Use sample data for testing
        sample_df = clean_df.head(100)
        sample_embeddings_full = generator.generate_embeddings_for_dataframe(sample_df)
        
        retriever = ProductRetriever(sample_embeddings_full, sample_df)
        query_embedding = generator.get_text_embedding("electronics")
        similar_products = retriever.get_similar_products_info(query_embedding, k=3)
        
        print(f"✅ Retriever: Found {len(similar_products)} similar products")
        
        # Test recommender
        from recommender import HybridRecommendationSystem
        
        recommender = HybridRecommendationSystem(sample_df, sample_embeddings_full)
        recommendations = recommender.get_hybrid_recommendations(
            "electronics", k=3, use_llm_rerank=False
        )
        
        print(f"✅ Recommender: Generated {len(recommendations)} recommendations")
        
        print("✅ All components working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {str(e)}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path('.env')
    
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    print("📝 Creating .env file...")
    
    env_content = """# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# Optional: Other API keys for future extensions
# OPENAI_API_KEY=your_openai_api_key_here
# HUGGINGFACE_API_TOKEN=your_huggingface_token_here

# Application Configuration
DATA_PATH=data/amazon_products.csv
CACHE_DIR=cache
LOG_LEVEL=INFO
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print("✅ .env file created")
    print("   ⚠️  Please add your Groq API key to enable LLM re-ranking")

def main():
    """Main setup function"""
    print("🚀 Amazon GenAI Recommender System Setup")
    print("=" * 50)
    
    # Step 1: Create directories
    create_directory_structure()
    
    # Step 2: Check dependencies
    if not check_dependencies():
        print("\n❌ Setup failed due to missing dependencies")
        print("Please run: pip install -r requirements.txt")
        return False
    
    # Step 3: Create sample dataset
    if not create_sample_dataset():
        print("\n❌ Failed to create sample dataset")
        return False
    
    # Step 4: Create .env file
    create_env_file()
    
    # Step 5: Test components
    if not test_system_components():
        print("\n❌ Component tests failed")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your Groq API key to the .env file")
    print("2. Replace the sample dataset with your real data (optional)")
    print("3. Run the application: streamlit run src/app.py")
    print("\nFor production use:")
    print("- Review and customize the configuration")
    print("- Implement proper error handling")
    print("- Set up monitoring and logging")
    print("- Consider scalability requirements")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)