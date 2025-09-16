import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from typing import Optional, Dict, List
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader
from embeddings import EmbeddingGenerator, create_embeddings_cache
from retriever import ProductRetriever
from recommender import HybridRecommendationSystem, evaluate_recommendations

# Page configuration
st.set_page_config(
    page_title="Amazon GenAI Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF9900;
        text-align: center;
        margin-bottom: 2rem;
    }
    .product-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
    .price {
        font-size: 1.2rem;
        font-weight: bold;
        color: #B12704;
    }
    .rating {
        color: #FF9900;
        font-weight: bold;
    }
    .similarity-score {
        background-color: #e8f5e8;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        loader = DataLoader("../data/amazon_products.csv")
        raw_data = loader.load_data()
        clean_data = loader.clean_data(raw_data)
        return clean_data, loader.get_data_info(clean_data)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

@st.cache_data
def load_embeddings(df):
    """Load or create embeddings"""
    try:
        embeddings_file = create_embeddings_cache(df)
        embeddings = np.load(embeddings_file)
        return embeddings
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return None

@st.cache_resource
def initialize_recommender(df, embeddings, groq_api_key):
    """Initialize the recommendation system"""
    try:
        return HybridRecommendationSystem(df, embeddings, groq_api_key)
    except Exception as e:
        st.error(f"Error initializing recommender: {str(e)}")
        return None

def display_product_card(product, show_similarity=True):
    """Display a product in a card format"""
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Display product image
            if product.get('img_link') and product['img_link'] != '':
                try:
                    st.image(product['img_link'], width=150, caption=product['product_name'][:50])
                except:
                    st.write("🖼️ Image not available")
            else:
                st.write("🖼️ No image")
        
        with col2:
            # Product details
            st.markdown(f"### {product['product_name']}")
            st.markdown(f"**Category:** {product['category']}")
            
            # Price and rating
            col2_1, col2_2, col2_3 = st.columns(3)
            with col2_1:
                st.markdown(f"<div class='price'>${product['discounted_price']:.2f}</div>", 
                           unsafe_allow_html=True)
                if product['actual_price'] > product['discounted_price']:
                    discount = (1 - product['discounted_price'] / product['actual_price']) * 100
                    st.markdown(f"~~${product['actual_price']:.2f}~~ ({discount:.0f}% off)")
            
            with col2_2:
                st.markdown(f"<div class='rating'>⭐ {product['rating']:.1f}/5.0</div>", 
                           unsafe_allow_html=True)
                st.markdown(f"({product['rating_count']} reviews)")
            
            with col2_3:
                if show_similarity and 'similarity_score' in product:
                    score = product['similarity_score']
                    st.markdown(f"<div class='similarity-score'>Match: {score:.1%}</div>", 
                               unsafe_allow_html=True)
                elif 'base_score' in product:
                    score = product['base_score']
                    st.markdown(f"<div class='similarity-score'>Score: {score:.3f}</div>", 
                               unsafe_allow_html=True)
            
            # Product description
            if 'about_product' in product and product['about_product']:
                description = product['about_product'][:200] + "..." if len(product['about_product']) > 200 else product['about_product']
                st.markdown(f"**Description:** {description}")
            
            # Product link
            if product.get('product_link'):
                st.markdown(f"[🔗 View Product]({product['product_link']})")
        
        st.markdown("---")

def create_analytics_dashboard(df, recommendations):
    """Create analytics dashboard"""
    st.subheader("📊 Analytics Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", len(df))
    with col2:
        st.metric("Average Rating", f"{df['rating'].mean():.2f}")
    with col3:
        st.metric("Categories", df['category'].nunique())
    with col4:
        if len(recommendations) > 0:
            st.metric("Avg Recommended Rating", f"{recommendations['rating'].mean():.2f}")
        else:
            st.metric("Recommendations", 0)
    
    # Category distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Category Distribution")
        category_counts = df['category'].value_counts().head(10)
        fig1 = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            title="Top 10 Categories",
            labels={'x': 'Number of Products', 'y': 'Category'}
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        if len(recommendations) > 0:
            st.subheader("Price Distribution of Recommendations")
            fig2 = px.histogram(
                recommendations,
                x='discounted_price',
                nbins=20,
                title="Price Distribution",
                labels={'discounted_price': 'Price ($)', 'count': 'Number of Products'}
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No recommendations to analyze yet.")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("<h1 class='main-header'>🛒 Amazon GenAI Recommender</h1>", unsafe_allow_html=True)
    st.markdown("*Powered by Groq API and Advanced Machine Learning*")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Key input
    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key for LLM re-ranking"
    )
    
    if not groq_api_key:
        st.sidebar.warning("⚠️ Add Groq API key for LLM re-ranking")
    
    # Load data
    with st.spinner("Loading dataset..."):
        df, data_info = load_data()
    
    if df is None:
        st.error("Failed to load dataset. Please check if 'data/amazon_products.csv' exists.")
        return
    
    # Display dataset info
    st.sidebar.subheader("📊 Dataset Info")
    st.sidebar.metric("Total Products", data_info['total_products'])
    st.sidebar.metric("Categories", data_info['unique_categories'])
    st.sidebar.metric("Price Range", f"${data_info['price_range']['min_price']:.0f} - ${data_info['price_range']['max_price']:.0f}")
    
    # Load embeddings
    with st.spinner("Loading embeddings..."):
        embeddings = load_embeddings(df)
    
    if embeddings is None:
        st.error("Failed to load embeddings.")
        return
    
    # Initialize recommender
    with st.spinner("Initializing recommendation system..."):
        recommender = initialize_recommender(df, embeddings, groq_api_key)
    
    if recommender is None:
        st.error("Failed to initialize recommendation system.")
        return
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search & Recommend", "📱 Product-Based", "📊 Analytics", "⚙️ Settings"])
    
    with tab1:
        st.header("Search & Get Recommendations")
        
        # Search input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query_text = st.text_area(
                "Enter your search query or describe what you're looking for:",
                placeholder="e.g., 'wireless headphones with noise cancellation' or 'comfortable running shoes'",
                height=100
            )
        
        with col2:
            st.write("**Search Options:**")
            num_recommendations = st.slider("Number of recommendations", 1, 20, 5)
            use_llm_rerank = st.checkbox("Use LLM Re-ranking", value=bool(groq_api_key))
            
            if use_llm_rerank and not groq_api_key:
                st.warning("LLM re-ranking requires Groq API key")
                use_llm_rerank = False
        
        # Filters
        st.subheader("🔧 Filters")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            categories = ['All'] + sorted(df['category'].unique().tolist())
            selected_category = st.selectbox("Category", categories)
        
        with filter_col2:
            min_price = st.number_input("Min Price ($)", min_value=0.0, value=0.0)
            max_price = st.number_input("Max Price ($)", min_value=0.0, value=float(df['discounted_price'].max()))
        
        with filter_col3:
            min_rating = st.slider("Minimum Rating", 1.0, 5.0, 1.0, 0.1)
        
        # Search button
        if st.button("🔍 Get Recommendations", type="primary"):
            if not query_text.strip():
                st.warning("Please enter a search query.")
            else:
                # Prepare filters
                filters = {
                    'category': None if selected_category == 'All' else selected_category,
                    'price_range': (min_price, max_price) if max_price > min_price else None,
                    'min_rating': min_rating if min_rating > 1.0 else None
                }
                
                # Clean filters (remove None values)
                filters = {k: v for k, v in filters.items() if v is not None}
                
                with st.spinner("Generating recommendations..."):
                    recommendations = recommender.get_hybrid_recommendations(
                        query_text=query_text,
                        filters=filters if filters else None,
                        k=num_recommendations,
                        use_llm_rerank=use_llm_rerank
                    )
                
                if len(recommendations) > 0:
                    st.success(f"Found {len(recommendations)} recommendations!")
                    
                    # Display recommendations
                    st.subheader("🎯 Recommended Products")
                    
                    for idx, (_, product) in enumerate(recommendations.iterrows()):
                        with st.expander(f"{idx+1}. {product['product_name']}", expanded=(idx < 3)):
                            display_product_card(product)
                    
                    # Evaluation metrics
                    if len(recommendations) > 0:
                        metrics = evaluate_recommendations(recommendations)
                        st.subheader("📈 Recommendation Quality")
                        
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        with metric_col1:
                            st.metric("Avg Rating", f"{metrics['avg_rating']:.2f}")
                        with metric_col2:
                            st.metric("Avg Price", f"${metrics['avg_price']:.2f}")
                        with metric_col3:
                            st.metric("Categories", metrics['category_diversity'])
                        with metric_col4:
                            st.metric("Total Items", metrics['total_recommendations'])
                
                else:
                    st.warning("No recommendations found. Try adjusting your query or filters.")
    
    with tab2:
        st.header("Product-Based Recommendations")
        st.write("Find similar products based on a specific product")
        
        # Product selection
        product_search = st.text_input(
            "Search for a product:",
            placeholder="Enter product name to search..."
        )
        
        if product_search:
            # Filter products by name
            matching_products = df[
                df['product_name'].str.contains(product_search, case=False, na=False)
            ].head(10)
            
            if len(matching_products) > 0:
                # Let user select a product
                selected_product = st.selectbox(
                    "Select a product:",
                    options=matching_products['product_id'].tolist(),
                    format_func=lambda x: matching_products[matching_products['product_id'] == x]['product_name'].iloc[0]
                )
                
                if st.button("Find Similar Products"):
                    with st.spinner("Finding similar products..."):
                        similar_products = recommender.get_product_based_recommendations(
                            product_id=selected_product,
                            k=5
                        )
                    
                    if len(similar_products) > 0:
                        # Display original product
                        st.subheader("📱 Original Product")
                        original_product = df[df['product_id'] == selected_product].iloc[0]
                        display_product_card(original_product, show_similarity=False)
                        
                        # Display similar products
                        st.subheader("🔗 Similar Products")
                        for idx, (_, product) in enumerate(similar_products.iterrows()):
                            with st.expander(f"{idx+1}. {product['product_name']}", expanded=(idx < 2)):
                                display_product_card(product)
                    else:
                        st.warning("No similar products found.")
            else:
                st.info("No products found matching your search.")
    
    with tab3:
        st.header("Analytics Dashboard")
        
        # Get sample recommendations for analytics
        sample_query = st.text_input(
            "Enter a query for analytics:",
            value="electronics",
            help="This will generate sample recommendations for analytics"
        )
        
        if st.button("Generate Analytics"):
            with st.spinner("Generating sample recommendations..."):
                sample_recs = recommender.get_hybrid_recommendations(
                    query_text=sample_query,
                    k=20,
                    use_llm_rerank=False
                )
            
            create_analytics_dashboard(df, sample_recs)
        else:
            create_analytics_dashboard(df, pd.DataFrame())
    
    with tab4:
        st.header("System Settings & Information")
        
        # System information
        st.subheader("🖥️ System Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.write("**Dataset Information:**")
            st.json({
                "Total Products": len(df),
                "Features": len(df.columns),
                "Categories": df['category'].nunique(),
                "Embedding Dimension": embeddings.shape[1] if embeddings is not None else "N/A"
            })
        
        with info_col2:
            st.write("**API Status:**")
            status_data = {
                "Groq API": "✅ Connected" if groq_api_key else "❌ Not configured",
                "Embeddings": "✅ Loaded" if embeddings is not None else "❌ Not loaded",
                "FAISS Index": "✅ Built" if recommender else "❌ Not built"
            }
            for key, value in status_data.items():
                st.write(f"- {key}: {value}")
        
        # Model settings
        st.subheader("🤖 Model Configuration")
        
        st.write("**Embedding Model:** all-MiniLM-L6-v2 (SentenceTransformer)")
        st.write("**Similarity Index:** FAISS IndexFlatIP")
        st.write("**LLM Model:** Llama-3-8B (Groq API)")
        
        # Cache management
        st.subheader("💾 Cache Management")
        
        if st.button("Clear Embedding Cache"):
            # Clear cache files
            cache_files = ["cache/product_embeddings.npy", "cache/embeddings_metadata.pkl"]
            for file in cache_files:
                if os.path.exists(file):
                    os.remove(file)
            st.success("Embedding cache cleared!")
            st.info("Please restart the app to regenerate embeddings.")
        
        # Export functionality
        st.subheader("📤 Export Data")
        
        if st.button("Export Sample Recommendations"):
            sample_recs = recommender.get_hybrid_recommendations(
                query_text="popular products",
                k=50,
                use_llm_rerank=False
            )
            
            if len(sample_recs) > 0:
                csv_data = sample_recs.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="sample_recommendations.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()