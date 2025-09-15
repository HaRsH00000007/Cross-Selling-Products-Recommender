# Amazon GenAI Recommender System

A sophisticated product recommendation system powered by Groq API and advanced machine learning techniques. This system combines collaborative filtering, content-based filtering, and LLM re-ranking to provide personalized product recommendations.

## 🚀 Features

- **Hybrid Recommendation Engine**: Combines multiple recommendation approaches
  - Content-based filtering using semantic embeddings
  - Collaborative filtering for personalized recommendations
  - Popularity-based recommendations for cold-start scenarios
  - LLM re-ranking using Groq API for enhanced relevance

- **Advanced Search & Filtering**
  - Semantic search using sentence transformers
  - Category, price, and rating filters
  - FAISS-powered fast similarity search

- **Interactive Streamlit Interface**
  - User-friendly web interface
  - Real-time recommendations
  - Product analytics dashboard
  - Configurable search parameters

- **Scalable Architecture**
  - Modular codebase for easy extension
  - Caching for improved performance
  - Support for large datasets

## 📁 Project Structure

```
amazon-genai-recommender/
├── data/
│   └── amazon_products.csv          # Your dataset
├── src/
│   ├── data_loader.py              # Data loading and preprocessing
│   ├── embeddings.py               # Embedding generation using Groq/SentenceTransformer
│   ├── retriever.py                # FAISS-based similarity search
│   ├── recommender.py              # Hybrid recommendation system
│   └── app.py                      # Streamlit web application
├── cache/                          # Generated embeddings cache
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
└── README.md                       # This file
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd amazon-genai-recommender
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your Groq API key
   ```

5. **Prepare your dataset**
   - Place your `amazon_products.csv` file in the `data/` directory
   - Ensure it contains the required columns (see Dataset Format below)

## 📊 Dataset Format

Your CSV file should contain the following columns:

| Column | Description | Required |
|--------|-------------|----------|
| `product_id` | Unique product identifier | ✅ |
| `product_name` | Product name/title | ✅ |
| `category` | Product category | ✅ |
| `discounted_price` | Current/discounted price | ✅ |
| `actual_price` | Original price | ✅ |
| `discount_percentage` | Discount percentage | ✅ |
| `rating` | Average rating (1-5) | ✅ |
| `rating_count` | Number of ratings | ✅ |
| `about_product` | Product description | ✅ |
| `user_id` | User identifier | Optional |
| `user_name` | User name | Optional |
| `review_id` | Review identifier | Optional |
| `review_title` | Review title | Optional |
| `review_content` | Review text | Optional |
| `img_link` | Product image URL | Optional |
| `product_link` | Product page URL | Optional |

## 🚀 Usage

### Running the Streamlit App

```bash
streamlit run src/app.py
```

The app will be available at `http://localhost:8501`

### Using Individual Components

#### 1. Data Loading
```python
from src.data_loader import DataLoader

loader = DataLoader("data/amazon_products.csv")
df = loader.load_data()
clean_df = loader.clean_data(df)
```

#### 2. Generating Embeddings
```python
from src.embeddings import EmbeddingGenerator

generator = EmbeddingGenerator(groq_api_key="your-api-key")
embeddings = generator.generate_embeddings_for_dataframe(clean_df)
```

#### 3. Setting Up Retriever
```python
from src.retriever import ProductRetriever

retriever = ProductRetriever(embeddings, clean_df)
similar_products = retriever.get_similar_products_info(query_embedding, k=5)
```

#### 4. Hybrid Recommendations
```python
from src.recommender import HybridRecommendationSystem

recommender = HybridRecommendationSystem(clean_df, embeddings, groq_api_key)
recommendations = recommender.get_hybrid_recommendations(
    query_text="wireless headphones",
    k=5,
    use_llm_rerank=True
)
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### API Key Setup

1. **Get Groq API Key**:
   - Visit [Groq Console](https://console.groq.com/)
   - Sign up and create an API key
   - Add it to your `.env` file

2. **Alternative: Use without Groq**:
   - The system works without Groq API key
   - LLM re-ranking will be disabled
   - All other features remain functional

## 📈 How It Works

### 1. Content-Based Filtering
- Generates embeddings for product descriptions using SentenceTransformer
- Uses FAISS for fast similarity search
- Matches products based on semantic similarity

### 2. Collaborative Filtering
- Analyzes user rating patterns (if user data available)
- Implements both user-based and item-based collaborative filtering
- Falls back to popularity-based recommendations

### 3. LLM Re-ranking
- Uses Groq API with Llama models
- Re-ranks candidate recommendations for better relevance
- Considers context and user intent

### 4. Hybrid Approach
- Combines multiple recommendation signals
- Weighted scoring system
- Diversity and quality optimization

## 🎯 Features in Detail

### Search & Recommendations
- **Semantic Search**: Natural language queries
- **Advanced Filters**: Category, price range, minimum rating
- **Hybrid Scoring**: Multiple algorithms combined
- **LLM Enhancement**: Context-aware re-ranking

### Analytics Dashboard
- **Dataset Statistics**: Overview of your data
- **Recommendation Quality**: Metrics and evaluation
- **Category Distribution**: Visual insights
- **Price Analysis**: Recommendation pricing patterns

### Product-Based Recommendations
- **Similar Products**: Find products like a specific item
- **Cross-selling**: Related product suggestions
- **Category-aware**: Maintains relevance within categories

## 🔧 Customization

### Adding New Recommendation Algorithms

1. **Extend the HybridRecommendationSystem class**:
```python
def custom_recommendation_method(self, query, k=10):
    # Your custom logic here
    return recommendations
```

2. **Integrate into hybrid scoring**:
```python
# Add to get_hybrid_recommendations method
custom_recs = self.custom_recommendation_method(query_text, k=k*2)
custom_recs['source'] = 'custom'
custom_recs['base_score'] = custom_recs['custom_score'] * 0.2
all_recommendations.append(custom_recs)
```

### Modifying the UI

The Streamlit app is highly customizable:
- Edit `src/app.py` to modify the interface
- Add new tabs for additional features
- Customize the styling in the CSS section

### Using Different Embedding Models

```python
# In embeddings.py, modify the model initialization
self.sentence_model = SentenceTransformer('your-preferred-model')
```

Popular alternatives:
- `all-MiniLM-L6-v2` (default, fast and efficient)
- `all-mpnet-base-v2` (better quality, slower)
- `multi-qa-MiniLM-L6-cos-v1` (optimized for Q&A)

## 📊 Performance Optimization

### Caching
- Embeddings are automatically cached
- FAISS indices can be saved/loaded
- Streamlit caching for data loading

### Scaling Tips
1. **Large Datasets**: Use `IndexIVFFlat` instead of `IndexFlatIP`
2. **Memory**: Process embeddings in batches
3. **Speed**: Pre-compute embeddings offline
4. **API Limits**: Implement rate limiting for Groq calls

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found" errors**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/amazon-genai-recommender"
   ```

2. **FAISS installation issues**
   ```bash
   pip install faiss-cpu --no-cache-dir
   ```

3. **Groq API errors**
   - Check your API key
   - Verify rate limits
   - System works without Groq (with reduced functionality)

4. **Memory issues with large datasets**
   - Reduce batch size in embeddings generation
   - Use IVF indices for FAISS
   - Process data in chunks

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Groq** for providing fast LLM inference
- **Sentence Transformers** for semantic embeddings
- **FAISS** for efficient similarity search
- **Streamlit** for the web interface
- **Amazon** for inspiration (this is an educational project)

## 📞 Support

For questions and support:
1. Check the troubleshooting section
2. Review the code comments
3. Create an issue on GitHub
4. Refer to the documentation of used libraries

---

**Note**: This is an educational/demonstration project. For production use, ensure proper data privacy, API rate limiting, and scalability considerations.
