"""
Predictive Models: Classification and Regression
Predict playlist characteristics and track popularity

Author: Adarsh Singh
Date: November 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'predictive_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PredictiveModels:
    """Build classification and regression models for playlist analysis."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.classifier = None
        self.regressor = None
    
    def load_data(self):
        """Load features."""
        logger.info("Loading data...")
        
        track_features = pd.read_parquet("data/processed/track_features_full.parquet")
        playlist_features = pd.read_parquet("data/processed/playlist_features_full.parquet")
        
        logger.info(f"Loaded {len(track_features):,} tracks")
        logger.info(f"Loaded {len(playlist_features):,} playlists")
        
        return track_features, playlist_features
    
    def build_classification_model(self, track_features):
        """
        Classification: Predict if a track is 'popular' (in 1000+ playlists)
        """
        logger.info("\n" + "="*60)
        logger.info("Building Classification Model: Track Popularity")
        logger.info("="*60)
        
        # Create binary target: popular (1) vs not popular (0)
        threshold = 1000
        track_features['is_popular'] = (track_features['playlist_count'] >= threshold).astype(int)
        
        logger.info(f"Popular tracks (>={threshold} playlists): {track_features['is_popular'].sum():,}")
        logger.info(f"Non-popular tracks: {(~track_features['is_popular'].astype(bool)).sum():,}")
        
        # Features for classification
        feature_cols = [
            'avg_position', 'std_position', 'position_consistency',
            'artist_popularity', 'album_popularity', 'duration_normalized'
        ]
        
        available_cols = [col for col in feature_cols if col in track_features.columns]
        X = track_features[available_cols].fillna(0).values
        y = track_features['is_popular'].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")
        
        # Train Random Forest Classifier
        logger.info("Training Random Forest Classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.classifier.score(X_train, y_train)
        test_score = self.classifier.score(X_test, y_test)
        
        logger.info(f"\nClassification Results:")
        logger.info(f"  Training Accuracy: {train_score:.4f}")
        logger.info(f"  Test Accuracy: {test_score:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': available_cols,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop 3 Most Important Features:")
        for idx, row in feature_importance.head(3).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self.classifier, {'train_accuracy': train_score, 'test_accuracy': test_score}
    
    def build_regression_model(self, track_features):
        """
        Regression: Predict track popularity (playlist count)
        """
        logger.info("\n" + "="*60)
        logger.info("Building Regression Model: Predict Playlist Count")
        logger.info("="*60)
        
        # Log-transform target for better distribution
        track_features['log_playlist_count'] = np.log1p(track_features['playlist_count'])
        
        # Features for regression
        feature_cols = [
            'avg_position', 'std_position', 'position_consistency',
            'artist_popularity', 'album_popularity', 'duration_normalized'
        ]
        
        available_cols = [col for col in feature_cols if col in track_features.columns]
        X = track_features[available_cols].fillna(0).values
        y = track_features['log_playlist_count'].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set: {len(X_train):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")
        
        # Train Random Forest Regressor
        logger.info("Training Random Forest Regressor...")
        self.regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.regressor.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.regressor.predict(X_train)
        test_pred = self.regressor.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        logger.info(f"\nRegression Results:")
        logger.info(f"  Training R²: {train_r2:.4f}")
        logger.info(f"  Test R²: {test_r2:.4f}")
        logger.info(f"  Test RMSE: {test_rmse:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': available_cols,
            'importance': self.regressor.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop 3 Most Important Features:")
        for idx, row in feature_importance.head(3).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self.regressor, {'train_r2': train_r2, 'test_r2': test_r2, 'test_rmse': test_rmse}
    
    def save_models(self):
        """Save models."""
        logger.info("\nSaving models...")
        
        # Save classifier
        classifier_file = self.models_dir / "track_popularity_classifier.pkl"
        with open(classifier_file, 'wb') as f:
            pickle.dump(self.classifier, f)
        logger.info(f"Saved classifier: {classifier_file}")
        
        # Save regressor
        regressor_file = self.models_dir / "track_count_regressor.pkl"
        with open(regressor_file, 'wb') as f:
            pickle.dump(self.regressor, f)
        logger.info(f"Saved regressor: {regressor_file}")

def main():
    """Main execution."""
    
    OUTPUT_DIR = "data/processed"
    
    pm = PredictiveModels(output_dir=OUTPUT_DIR)
    
    # Load data
    track_features, playlist_features = pm.load_data()
    
    # Build classification model
    classifier, class_results = pm.build_classification_model(track_features)
    
    # Build regression model
    regressor, reg_results = pm.build_regression_model(track_features)
    
    # Save models
    pm.save_models()
    
    logger.info("\n" + "="*60)
    logger.info("Predictive Models Complete!")
    logger.info("="*60)
    logger.info(f"Classification Accuracy: {class_results['test_accuracy']:.4f}")
    logger.info(f"Regression R²: {reg_results['test_r2']:.4f}")
    logger.info("✅ Both models trained and saved!")

if __name__ == "__main__":
    main()