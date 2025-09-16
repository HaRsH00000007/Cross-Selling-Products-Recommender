import os
import re
import logging
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Groq may not be installed in every env — handle gracefully
try:
    from groq import Groq
except Exception:
    Groq = None

# Project-specific imports (assume these modules exist in the same package)
from retriever import ProductRetriever, CategoryRetriever, PriceBasedRetriever
from embeddings import EmbeddingGenerator

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRecommendationSystem:
    """
    Hybrid recommendation system combining:
      - Content-based filtering (embeddings)
      - Collaborative filtering (user-item matrix)
      - Popularity-based fallback
      - Optional LLM re-ranking (via Groq)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        groq_api_key: Optional[str] = None,
    ):
        self.df = df.copy()
        self.embeddings = embeddings
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")

        # Initialize components
        self.product_retriever = ProductRetriever(embeddings, self.df)
        self.category_retriever = CategoryRetriever(self.df)
        self.price_retriever = PriceBasedRetriever(self.df)
        self.embedding_generator = EmbeddingGenerator(groq_api_key=self.groq_api_key)

        # Initialize Groq client if available
        if self.groq_api_key and Groq is not None:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}")
                self.groq_client = None
        else:
            self.groq_client = None
            if not self.groq_api_key:
                logger.info("GROQ_API_KEY not provided — LLM re-ranking disabled.")
            else:
                logger.info("Groq SDK not available — LLM re-ranking disabled.")

        # Build collaborative features
        self._build_collaborative_features()
        logger.info("HybridRecommendationSystem initialized")

    # ---------- COLLABORATIVE FEATURE BUILDING ---------- #
    def _build_collaborative_features(self):
        try:
            if "user_id" in self.df.columns and "rating" in self.df.columns:
                self.user_item_matrix = self.df.pivot_table(
                    index="user_id", columns="product_id", values="rating", fill_value=0
                )
                logger.info(f"User-item matrix built: {self.user_item_matrix.shape}")
            else:
                self.user_item_matrix = None
                logger.info("User-item matrix not built (missing user_id or rating).")

            # Category popularity metrics (safe aggregation)
            if "category" in self.df.columns:
                self.category_popularity = (
                    self.df.groupby("category")
                    .agg(
                        rating=("rating", "mean"),
                        rating_count=("rating_count", "sum"),
                        product_count=("product_id", "count"),
                    )
                    .reset_index()
                )
                self.category_popularity["popularity_score"] = (
                    self.category_popularity["rating"].fillna(0) * 0.4
                    + np.log1p(self.category_popularity["rating_count"].fillna(0)) * 0.3
                    + np.log1p(self.category_popularity["product_count"].fillna(0)) * 0.3
                )
            else:
                self.category_popularity = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error building collaborative features: {e}")
            self.user_item_matrix = None
            self.category_popularity = pd.DataFrame()

    # ---------- CONTENT-BASED ---------- #
    def content_based_recommendations(
        self, query_text: str, k: int = 10, filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Generate content-based recommendations using embeddings."""
        try:
            query_embedding = self.embedding_generator.get_text_embedding(query_text)
            if filters:
                return self.product_retriever.hybrid_search(
                    query_embedding=query_embedding,
                    category_filter=filters.get("category"),
                    price_range=filters.get("price_range"),
                    min_rating=filters.get("min_rating"),
                    k=k,
                )
            return self.product_retriever.get_similar_products_info(query_embedding, k=k)
        except Exception as e:
            logger.error(f"Error in content_based_recommendations: {e}")
            return pd.DataFrame()

    # ---------- COLLABORATIVE FILTERING ---------- #
    def collaborative_filtering_recommendations(
        self, user_id: Optional[str] = None, product_id: Optional[str] = None, k: int = 10
    ) -> pd.DataFrame:
        """User-based or item-based collaborative filtering (fallbacks to popularity)."""
        if self.user_item_matrix is None:
            logger.warning("Collaborative filtering unavailable — falling back to popularity.")
            return self.popularity_based_recommendations(k=k)

        try:
            if user_id and user_id in self.user_item_matrix.index:
                return self._user_based_cf(user_id, k)
            if product_id:
                return self._item_based_cf(product_id, k)
            logger.warning("No user_id or product_id given for CF — returning popularity results.")
            return self.popularity_based_recommendations(k=k)
        except Exception as e:
            logger.error(f"Error in collaborative_filtering_recommendations: {e}")
            return self.popularity_based_recommendations(k=k)

    def _user_based_cf(self, user_id: str, k: int) -> pd.DataFrame:
        user_ratings = self.user_item_matrix.loc[user_id]
        user_similarities = cosine_similarity([user_ratings], self.user_item_matrix)[0]
        similar_user_indices = np.argsort(user_similarities)[::-1][1:11]  # top 10 similar users

        recommendations = {}
        for idx in similar_user_indices:
            sim_user = self.user_item_matrix.index[idx]
            sim_ratings = self.user_item_matrix.loc[sim_user]
            for pid, rating in sim_ratings.items():
                if user_ratings.get(pid, 0) == 0 and rating > 3:
                    recommendations.setdefault(pid, []).append(rating * user_similarities[idx])

        product_scores = {pid: np.mean(scores) for pid, scores in recommendations.items()} if recommendations else {}
        top_product_ids = sorted(product_scores, key=product_scores.get, reverse=True)[:k]
        return self.df[self.df["product_id"].isin(top_product_ids)]

    def _item_based_cf(self, product_id: str, k: int) -> pd.DataFrame:
        if product_id not in self.user_item_matrix.columns:
            logger.warning(f"Product {product_id} not in user-item matrix.")
            return self.popularity_based_recommendations(k=k)

        product_ratings = self.user_item_matrix[product_id]
        item_similarities = cosine_similarity(self.user_item_matrix.T, product_ratings.values.reshape(1, -1)).flatten()
        similar_indices = np.argsort(item_similarities)[::-1][1 : k + 1]
        similar_product_ids = [self.user_item_matrix.columns[idx] for idx in similar_indices]

        result = self.df[self.df["product_id"].isin(similar_product_ids)].copy()
        result["similarity_score"] = [item_similarities[idx] for idx in similar_indices]
        return result

    # ---------- POPULARITY-BASED ---------- #
    def popularity_based_recommendations(self, category: Optional[str] = None, k: int = 10) -> pd.DataFrame:
        try:
            df_filtered = self.df.copy()
            if category and "category" in df_filtered.columns:
                df_filtered = df_filtered[df_filtered["category"].str.contains(category, case=False, na=False)]

            if df_filtered.empty:
                return pd.DataFrame()

            # Safely compute popularity score
            if "rating" not in df_filtered.columns:
                df_filtered["rating"] = 0
            if "rating_count" not in df_filtered.columns:
                df_filtered["rating_count"] = 0

            df_filtered["popularity_score"] = df_filtered["rating"] * 0.6 + np.log1p(df_filtered["rating_count"]) * 0.4
            return df_filtered.nlargest(k, "popularity_score")
        except Exception as e:
            logger.error(f"Error in popularity_based_recommendations: {e}")
            return pd.DataFrame()

    # ---------- LLM RE-RANKING ---------- #
    def llm_rerank_recommendations(self, recommendations: pd.DataFrame, query_text: str, top_k: int = 5) -> pd.DataFrame:
        """Use an LLM (Groq client) to re-rank candidate recommendations. Returns top_k rows."""
        if self.groq_client is None or recommendations is None or len(recommendations) == 0:
            return recommendations.head(top_k)

        try:
            # Build compact product descriptions for the prompt
            products_text = []
            for _, row in recommendations.iterrows():
                name = row.get("product_name", "")
                category = row.get("category", "")
                price = row.get("discounted_price", None)
                rating = row.get("rating", None)
                rating_count = row.get("rating_count", None)
                desc = str(row.get("about_product", ""))[:200]
                products_text.append(
                    f"Product: {name}\nCategory: {category}\nPrice: {price}\nRating: {rating} ({rating_count} reviews)\nDescription: {desc}\n"
                )

            prompt = f"""Given the user query: "{query_text}"

Rank the following products from most relevant (1) to least relevant ({len(products_text)}) based on relevance to the query, rating, value for money, and user satisfaction.

Products:
{chr(10).join([f'{i+1}. {p}' for i, p in enumerate(products_text)])}

Reply with a comma-separated ranking (e.g., "3,1,4,2,...")."""

            # Call Groq chat completion (API may differ between versions; this tries the common pattern)
            try:
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-8b-8192",
                    temperature=0.1,
                    max_tokens=200,
                )
                ranking_text = ""
                # Different SDK versions return different structures — try to extract text safely
                if hasattr(response, "choices") and len(response.choices) > 0:
                    # e.g., response.choices[0].message.content
                    try:
                        ranking_text = response.choices[0].message.content.strip()
                    except Exception:
                        # fallback to str(response)
                        ranking_text = str(response)
                else:
                    ranking_text = str(response)
            except Exception as e:
                logger.warning(f"Groq call failed: {e}")
                return recommendations.head(top_k)

            # Try to parse the ranking — extract integers in order
            nums = re.findall(r"\d+", ranking_text)
            if len(nums) >= len(products_text):
                rankings = [int(n) - 1 for n in nums[: len(products_text)]]
                # Validate and reorder
                if all(0 <= r < len(recommendations) for r in rankings):
                    reordered = recommendations.iloc[rankings].copy()
                    reordered["llm_rank"] = range(1, len(reordered) + 1)
                    return reordered.head(top_k)
                else:
                    logger.warning("Parsed rankings invalid — falling back to original order.")
            else:
                logger.warning("Could not parse full ranking from LLM — using original order.")

        except Exception as e:
            logger.error(f"Error in llm_rerank_recommendations: {e}")

        return recommendations.head(top_k)

    # ---------- HYBRID ---------- #
    def get_hybrid_recommendations(
        self,
        query_text: str,
        user_id: Optional[str] = None,
        filters: Optional[Dict] = None,
        k: int = 10,
        use_llm_rerank: bool = True,
    ) -> pd.DataFrame:
        """Combine content, collaborative and popularity signals and optionally LLM re-rank."""
        try:
            sources = []

            # Content-based (40% weight)
            cb = self.content_based_recommendations(query_text, k=k * 2, filters=filters)
            if cb is not None and not cb.empty:
                cb = cb.copy()
                cb["source"] = "content"
                if "similarity_score" in cb.columns:
                    cb["base_score"] = cb["similarity_score"] * 0.4
                else:
                    cb["base_score"] = 0.5 * 0.4
                sources.append(cb)

            # Collaborative filtering (30% weight)
            if user_id:
                cf = self.collaborative_filtering_recommendations(user_id=user_id, k=k)
                if cf is not None and not cf.empty:
                    cf = cf.copy()
                    cf["source"] = "collaborative"
                    cf["base_score"] = cf.get("similarity_score", 0.4) * 0.3
                    sources.append(cf)

            # Popularity-based (30% weight)
            category_filter = filters.get("category") if filters else None
            pop = self.popularity_based_recommendations(category=category_filter, k=k)
            if pop is not None and not pop.empty:
                pop = pop.copy()
                pop["source"] = "popularity"
                pop["base_score"] = (
                    (pop.get("rating", 0) / 5.0 * 0.6 + np.log1p(pop.get("rating_count", 0)) / 10.0 * 0.4)
                    * 0.3
                )
                sources.append(pop)

            if not sources:
                logger.warning("No recommendations generated from any method.")
                return pd.DataFrame()

            combined = pd.concat(sources, ignore_index=True)
            # Aggregate duplicate products by summing base_score and keeping first values for other fields
            agg = {
                "product_name": "first",
                "category": "first",
                "discounted_price": "first",
                "actual_price": "first",
                "rating": "first",
                "rating_count": "first",
                "about_product": "first",
                "product_link": "first",
                "img_link": "first",
                "base_score": "sum",
                "source": lambda x: ",".join(x.unique()),
            }
            unique_recs = combined.groupby("product_id").agg(agg).reset_index()
            unique_recs = unique_recs.sort_values("base_score", ascending=False)

            # Candidate pool for re-ranking
            candidates = unique_recs.head(min(k * 2, len(unique_recs)))

            # LLM re-ranking (optional)
            if use_llm_rerank and self.groq_client is not None:
                final_recs = self.llm_rerank_recommendations(candidates, query_text, top_k=k)
            else:
                final_recs = candidates.head(k)

            final_recs = final_recs.copy()
            final_recs["final_rank"] = range(1, len(final_recs) + 1)
            return final_recs

        except Exception as e:
            logger.error(f"Error in get_hybrid_recommendations: {e}")
            return pd.DataFrame()

    # ---------- PRODUCT-BASED ---------- #
    def get_product_based_recommendations(self, product_id: str, k: int = 5) -> pd.DataFrame:
        try:
            product_info = self.product_retriever.get_product_by_id(product_id)
            if not product_info:
                return pd.DataFrame()
            query_text = f"{product_info.get('product_name', '')} {product_info.get('about_product', '')}"
            recs = self.content_based_recommendations(query_text, k=k + 1, filters={"category": product_info.get("category")})
            recs = recs[recs["product_id"] != product_id]
            return recs.head(k)
        except Exception as e:
            logger.error(f"Error in get_product_based_recommendations: {e}")
            return pd.DataFrame()


# ---------- UTILITY EVALUATION FUNCTION (EXPORTABLE) ---------- #
def evaluate_recommendations(recommendations: pd.DataFrame, ground_truth: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Evaluate recommendation quality. Returns basic metrics:
      - avg_rating
      - avg_price
      - category_diversity
      - total_recommendations
      - precision, recall, f1_score (if ground_truth provided)
    """
    metrics: Dict[str, float] = {}

    if recommendations is None or len(recommendations) == 0:
        metrics["total_recommendations"] = 0
        return metrics

    # Safe column access
    metrics["avg_rating"] = float(recommendations["rating"].mean()) if "rating" in recommendations.columns else 0.0
    metrics["avg_price"] = float(recommendations["discounted_price"].mean()) if "discounted_price" in recommendations.columns else 0.0
    metrics["category_diversity"] = int(recommendations["category"].nunique()) if "category" in recommendations.columns else 0
    metrics["total_recommendations"] = int(len(recommendations))

    if ground_truth:
        rec_ids = set(recommendations["product_id"].tolist()) if "product_id" in recommendations.columns else set()
        relevant_ids = set(ground_truth)
        if rec_ids:
            precision = len(rec_ids.intersection(relevant_ids)) / len(rec_ids)
            metrics["precision"] = float(precision)
        if relevant_ids:
            recall = len(rec_ids.intersection(relevant_ids)) / len(relevant_ids)
            metrics["recall"] = float(recall)
        if metrics.get("precision") is not None and metrics.get("recall") is not None and (metrics["precision"] + metrics["recall"]) > 0:
            p = metrics["precision"]
            r = metrics["recall"]
            metrics["f1_score"] = 2 * p * r / (p + r)

    return metrics


# ---------- QUICK SELF-TEST ---------- #
if __name__ == "__main__":
    print("Testing recommender module (lightweight)...")
    sample_data = {
        "product_id": [f"P{i:03d}" for i in range(20)],
        "product_name": [f"Product {i}" for i in range(20)],
        "category": ["Electronics", "Clothing", "Books", "Home", "Sports"] * 4,
        "discounted_price": np.random.uniform(10, 500, 20),
        "actual_price": np.random.uniform(10, 600, 20),
        "rating": np.random.uniform(1, 5, 20),
        "rating_count": np.random.randint(1, 500, 20),
        "about_product": [f"Product {i} description" for i in range(20)],
        "product_link": [f"https://example.com/p{i}" for i in range(20)],
        "img_link": [f"https://example.com/img{i}.jpg" for i in range(20)],
        "user_id": ["U1", "U2", "U3", "U4"] * 5,
        "review_content": [f"Review {i}" for i in range(20)],
    }

    df = pd.DataFrame(sample_data)
    embeddings = np.random.rand(20, 384).astype(np.float32)

    recommender = HybridRecommendationSystem(df, embeddings)
    res = recommender.get_hybrid_recommendations("cheap electronics", k=5, use_llm_rerank=False)
    print(res[["product_id", "product_name", "category", "rating", "base_score"]])
