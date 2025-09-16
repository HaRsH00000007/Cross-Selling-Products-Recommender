import os
import pandas as pd
import numpy as np
import logging
from ast import literal_eval

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    @staticmethod
    def _safe_convert(value):
        """Safely convert a cell value to float if possible."""
        if pd.isna(value):
            return np.nan

        # Try direct float conversion
        try:
            return float(value)
        except (ValueError, TypeError):
            pass

        # Try parsing list-like strings e.g. "['4.2']"
        try:
            parsed = literal_eval(str(value))
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                return float(parsed[0])
        except Exception:
            pass

        return np.nan

    def load_data(self) -> pd.DataFrame:
        """Load raw CSV data."""
        if not os.path.exists(self.file_path):
            logger.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")
        df = pd.read_csv(self.file_path)
        logger.info(f"Data loaded successfully from {self.file_path}. Shape: {df.shape}")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by handling missing, duplicate, and invalid values."""
        logger.info("Starting data cleaning...")

        # Drop duplicates
        df = df.drop_duplicates()

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Columns to clean
        numeric_columns = ["discounted_price", "actual_price", "rating", "rating_count"]
        for col in numeric_columns:
            if col in df.columns:
                logger.info(f"Cleaning column: {col}")
                df[col] = df[col].apply(DataLoader._safe_convert)

        # Drop rows with too many missing values
        df = df.dropna(thresh=len(df.columns) - 2)

        # Create combined_text column
        logger.info("Creating combined_text column.")
        df['combined_text'] = df['product_name'].fillna('') + ' ' + df['about_product'].fillna('')

        # Reset index
        df = df.reset_index(drop=True)

        logger.info("Data cleaning completed successfully.")
        return df

    def get_data(self) -> pd.DataFrame:
        """Load and clean data in one go."""
        df = self.load_data()
        df = self.clean_data(df)
        return df

    def get_data_info(self, df: pd.DataFrame):
        """Return dataset summary info for the sidebar metrics."""
        return {
            "total_products": len(df),
            "unique_categories": df["category"].nunique() if "category" in df.columns else 0,
            "price_range": {
                "min_price": df["discounted_price"].min() if "discounted_price" in df.columns else 0,
                "max_price": df["discounted_price"].max() if "discounted_price" in df.columns else 0,
            }
        }

# --- Wrapper function for app.py ---
def get_data(file_path: str) -> pd.DataFrame:
    """Convenience function to load and clean data directly."""
    loader = DataLoader(file_path)
    return loader.get_data()
