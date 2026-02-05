"""
Phase 2: Continuous Learning System
Implements online model updating and performance feedback for adaptive trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import logging
from collections import deque
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Track model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    mae: float
    r2_score: float
    prediction_count: int
    last_updated: str


@dataclass
class TradeOutcome:
    """Track trade outcomes for learning"""
    symbol: str
    prediction: float
    actual_outcome: float
    features: List[float]
    timestamp: str
    profit_loss: float
    hold_duration: int


class ContinuousLearningSystem:
    """
    Implements online learning with performance feedback and model adaptation
    """

    def __init__(self, model_save_path: str = "models/"):
        self.model_save_path = model_save_path
        self.models = {}
        self.scalers = {}
        self.performance_history = {}
        self.trade_outcomes = deque(maxlen=1000)  # Keep last 1000 trades

        # Learning parameters
        self.min_samples_for_update = 10
        self.performance_window = 50
        self.model_retrain_threshold = 0.1  # Retrain if performance drops 10%
        self.learning_rate = 0.01

        # Model ensemble
        self.ensemble_weights = {}
        self.model_types = ["rf", "gb", "sgd"]  # Random Forest, Gradient Boosting, SGD

        # Performance tracking
        self.baseline_performance = {}
        self.current_performance = {}
        self.adaptation_history = []

        self._initialize_models()
        logger.info("✅ Continuous Learning System initialized")

    def _initialize_models(self):
        """Initialize the ensemble of models"""
        try:
            # Random Forest - good for feature importance and non-linear patterns
            self.models["rf"] = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )

            # Gradient Boosting - good for sequential learning
            self.models["gb"] = GradientBoostingRegressor(
                n_estimators=50, learning_rate=0.1, max_depth=6, random_state=42
            )

            # SGD Regressor - good for online learning
            self.models["sgd"] = SGDRegressor(learning_rate="adaptive", eta0=0.01, random_state=42)

            # Initialize scalers and equal weights
            for model_type in self.model_types:
                self.scalers[model_type] = StandardScaler()
                self.ensemble_weights[model_type] = 1.0 / len(self.model_types)

            logger.info(f"Initialized {len(self.models)} models in ensemble")

        except Exception as e:
            logger.error(f"Error initializing models: {e}")

    def train_initial_models(self, training_data: pd.DataFrame, target_column: str):
        """Train initial models with historical data"""
        try:
            if len(training_data) < 100:
                logger.warning(f"Insufficient training data: {len(training_data)} samples")
                return False

            # Prepare features and target
            feature_columns = [col for col in training_data.columns if col != target_column]
            X = training_data[feature_columns].fillna(0)
            y = training_data[target_column].fillna(0)

            # Split for validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Train each model
            for model_type in self.model_types:
                try:
                    X_train_scaled = self.scalers[model_type].fit_transform(X_train)
                    X_val_scaled = self.scalers[model_type].transform(X_val)

                    self.models[model_type].fit(X_train_scaled, y_train)
                    y_pred = self.models[model_type].predict(X_val_scaled)
                    performance = self._calculate_performance_metrics(y_val, y_pred)

                    self.baseline_performance[model_type] = performance
                    self.current_performance[model_type] = performance

                    logger.info(
                        f"Model {model_type} trained - R²: {performance.r2_score:.3f}, MSE: {performance.mse:.3f}"
                    )

                except Exception as e:
                    logger.error(f"Error training {model_type}: {e}")
                    continue

            self._save_models()
            logger.info("✅ Initial model training completed")
            return True

        except Exception as e:
            logger.error(f"Error in initial training: {e}")
            return False

    def predict_ensemble(self, features: List[float]) -> Dict:
        """Make prediction using ensemble of models"""
        try:
            features_array = np.array(features).reshape(1, -1)
            predictions = {}
            weights = []
            valid_predictions = []

            for model_type in self.model_types:
                try:
                    features_scaled = self.scalers[model_type].transform(features_array)
                    pred = self.models[model_type].predict(features_scaled)[0]
                    predictions[model_type] = pred

                    weight = self.ensemble_weights[model_type]
                    weights.append(weight)
                    valid_predictions.append(pred)

                except Exception as e:
                    logger.warning(f"Error predicting with {model_type}: {e}")
                    continue

            if not valid_predictions:
                logger.error("No valid predictions from ensemble")
                return {"ensemble_prediction": 0.0, "model_predictions": {}, "confidence": 0.0}

            weights = np.array(weights)
            weights = weights / weights.sum()
            ensemble_pred = float(np.average(valid_predictions, weights=weights))
            pred_std = float(np.std(valid_predictions))
            confidence = max(0.0, 1.0 - pred_std)

            return {
                "ensemble_prediction": ensemble_pred,
                "model_predictions": predictions,
                "confidence": confidence,
                "prediction_count": len(valid_predictions),
            }

        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {"ensemble_prediction": 0.0, "model_predictions": {}, "confidence": 0.0}

    def update_with_outcome(self, trade_outcome: TradeOutcome):
        """Update models with actual trade outcome"""
        try:
            self.trade_outcomes.append(trade_outcome)

            if len(self.trade_outcomes) < self.min_samples_for_update:
                return

            recent_outcomes = list(self.trade_outcomes)[-self.performance_window:]
            X_new = np.array([outcome.features for outcome in recent_outcomes])
            y_new = np.array([outcome.actual_outcome for outcome in recent_outcomes])

            try:
                if len(X_new) > 0:
                    X_new_scaled = self.scalers["sgd"].transform(X_new)
                    self.models["sgd"].partial_fit(X_new_scaled, y_new)

                    y_pred = self.models["sgd"].predict(X_new_scaled)
                    performance = self._calculate_performance_metrics(y_new, y_pred)
                    self.current_performance["sgd"] = performance

            except Exception as e:
                logger.warning(f"Error updating SGD model: {e}")

            self._check_model_performance()
            self._update_ensemble_weights()

            logger.debug(f"Updated models with trade outcome: {trade_outcome.symbol}")

        except Exception as e:
            logger.error(f"Error updating with outcome: {e}")

    def _calculate_performance_metrics(self, y_true, y_pred) -> ModelPerformance:
        """Calculate comprehensive performance metrics"""
        try:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            y_true_binary = (y_true > 0).astype(int)
            y_pred_binary = (y_pred > 0).astype(int)

            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = float(np.mean(y_true_binary == y_pred_binary))

            return ModelPerformance(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                mse=mse,
                mae=mae,
                r2_score=r2,
                prediction_count=len(y_true),
                last_updated=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return ModelPerformance(0, 0, 0, 0, float("inf"), float("inf"), -1, 0, datetime.now().isoformat())

    def _check_model_performance(self):
        """Check if models need retraining due to performance degradation"""
        try:
            for model_type in self.model_types:
                if model_type not in self.baseline_performance or model_type not in self.current_performance:
                    continue

                baseline_r2 = self.baseline_performance[model_type].r2_score
                current_r2 = self.current_performance[model_type].r2_score

                if baseline_r2 > 0 and (baseline_r2 - current_r2) > self.model_retrain_threshold:
                    logger.warning(
                        f"Model {model_type} performance degraded: {baseline_r2:.3f} → {current_r2:.3f}"
                    )
                    self._schedule_model_retrain(model_type)

        except Exception as e:
            logger.error(f"Error checking model performance: {e}")

    def _schedule_model_retrain(self, model_type: str):
        """Schedule model retraining (simplified implementation)"""
        try:
            logger.info(f"Scheduling retrain for {model_type} model")
            if model_type in self.current_performance:
                self.baseline_performance[model_type] = self.current_performance[model_type]

        except Exception as e:
            logger.error(f"Error scheduling retrain: {e}")

    def _update_ensemble_weights(self):
        """Update ensemble weights based on recent performance"""
        try:
            total_performance = 0
            model_scores = {}

            for model_type in self.model_types:
                if model_type in self.current_performance:
                    perf = self.current_performance[model_type]
                    score = (perf.accuracy + perf.f1_score + max(0, perf.r2_score)) / 3
                    model_scores[model_type] = max(0.01, score)
                    total_performance += model_scores[model_type]
                else:
                    model_scores[model_type] = 0.01

            if total_performance > 0:
                for model_type in self.model_types:
                    self.ensemble_weights[model_type] = model_scores[model_type] / total_performance

            logger.debug(f"Updated ensemble weights: {self.ensemble_weights}")

        except Exception as e:
            logger.error(f"Error updating ensemble weights: {e}")

    def get_learning_insights(self) -> Dict:
        """Get insights about the learning system's performance"""
        try:
            insights = {
                "total_trades_learned": len(self.trade_outcomes),
                "models_performance": {},
                "ensemble_weights": self.ensemble_weights.copy(),
                "recent_accuracy": self._calculate_recent_accuracy(),
                "learning_trend": self._analyze_learning_trend(),
                "feature_importance": self._get_feature_importance(),
            }

            for model_type in self.model_types:
                if model_type in self.current_performance:
                    insights["models_performance"][model_type] = asdict(self.current_performance[model_type])

            return insights

        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {}

    def _calculate_recent_accuracy(self) -> float:
        """Calculate accuracy on recent predictions"""
        try:
            if len(self.trade_outcomes) < 10:
                return 0.0

            recent_outcomes = list(self.trade_outcomes)[-20:]
            correct_predictions = 0

            for outcome in recent_outcomes:
                predicted_direction = 1 if outcome.prediction > 0 else -1
                actual_direction = 1 if outcome.actual_outcome > 0 else -1

                if predicted_direction == actual_direction:
                    correct_predictions += 1

            return correct_predictions / len(recent_outcomes)

        except Exception as e:
            logger.error(f"Error calculating recent accuracy: {e}")
            return 0.0

    def _analyze_learning_trend(self) -> str:
        """Analyze if the model is improving over time"""
        try:
            if len(self.trade_outcomes) < 20:
                return "Insufficient data"

            all_outcomes = list(self.trade_outcomes)
            mid_point = len(all_outcomes) // 2

            first_half = all_outcomes[:mid_point]
            second_half = all_outcomes[mid_point:]

            first_accuracy = self._calculate_accuracy_for_outcomes(first_half)
            second_accuracy = self._calculate_accuracy_for_outcomes(second_half)

            if second_accuracy > first_accuracy + 0.05:
                return "Improving"
            elif second_accuracy < first_accuracy - 0.05:
                return "Declining"
            else:
                return "Stable"

        except Exception as e:
            logger.error(f"Error analyzing learning trend: {e}")
            return "Unknown"

    def _calculate_accuracy_for_outcomes(self, outcomes: List[TradeOutcome]) -> float:
        """Calculate accuracy for a list of trade outcomes"""
        if not outcomes:
            return 0.0

        correct = 0
        for outcome in outcomes:
            predicted_direction = 1 if outcome.prediction > 0 else -1
            actual_direction = 1 if outcome.actual_outcome > 0 else -1
            if predicted_direction == actual_direction:
                correct += 1

        return correct / len(outcomes)

    def _get_feature_importance(self) -> Dict:
        """Get feature importance from tree-based models"""
        try:
            importance_dict = {}

            if "rf" in self.models and hasattr(self.models["rf"], "feature_importances_"):
                rf_importance = self.models["rf"].feature_importances_
                importance_dict["random_forest"] = rf_importance.tolist()

            if "gb" in self.models and hasattr(self.models["gb"], "feature_importances_"):
                gb_importance = self.models["gb"].feature_importances_
                importance_dict["gradient_boosting"] = gb_importance.tolist()

            return importance_dict

        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}

    def _save_models(self):
        """Save models and scalers to disk"""
        try:
            import os
            os.makedirs(self.model_save_path, exist_ok=True)

            for model_type, model in self.models.items():
                model_path = f"{self.model_save_path}/{model_type}_model.joblib"
                joblib.dump(model, model_path)

            for scaler_type, scaler in self.scalers.items():
                scaler_path = f"{self.model_save_path}/{scaler_type}_scaler.joblib"
                joblib.dump(scaler, scaler_path)

            performance_data = {
                "baseline_performance": {k: asdict(v) for k, v in self.baseline_performance.items()},
                "current_performance": {k: asdict(v) for k, v in self.current_performance.items()},
                "ensemble_weights": self.ensemble_weights,
            }

            with open(f"{self.model_save_path}/performance_history.json", "w") as f:
                json.dump(performance_data, f, indent=2)

            logger.info("Models and performance data saved")

        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def load_models(self):
        """Load models and scalers from disk"""
        try:
            import os

            for model_type in self.model_types:
                model_path = f"{self.model_save_path}/{model_type}_model.joblib"
                if os.path.exists(model_path):
                    self.models[model_type] = joblib.load(model_path)

            for scaler_type in self.model_types:
                scaler_path = f"{self.model_save_path}/{scaler_type}_scaler.joblib"
                if os.path.exists(scaler_path):
                    self.scalers[scaler_type] = joblib.load(scaler_path)

            performance_path = f"{self.model_save_path}/performance_history.json"
            if os.path.exists(performance_path):
                with open(performance_path, "r") as f:
                    performance_data = json.load(f)

                for model_type, perf_dict in performance_data.get("baseline_performance", {}).items():
                    self.baseline_performance[model_type] = ModelPerformance(**perf_dict)

                for model_type, perf_dict in performance_data.get("current_performance", {}).items():
                    self.current_performance[model_type] = ModelPerformance(**perf_dict)

                self.ensemble_weights = performance_data.get("ensemble_weights", self.ensemble_weights)

            logger.info("Models and performance data loaded")
            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def reset_learning(self):
        """Reset the learning system"""
        try:
            self.trade_outcomes.clear()
            self.current_performance.clear()
            self.baseline_performance.clear()

            for model_type in self.model_types:
                self.ensemble_weights[model_type] = 1.0 / len(self.model_types)

            logger.info("Learning system reset")

        except Exception as e:
            logger.error(f"Error resetting learning system: {e}")


# Global instance
_learning_system = None


def get_learning_system() -> ContinuousLearningSystem:
    """Get the global continuous learning system instance"""
    global _learning_system
    if _learning_system is None:
        _learning_system = ContinuousLearningSystem()
    return _learning_system
"""
Phase 2: Continuous Learning System
Implements online model updating and performance feedback for adaptive trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import logging
from collections import deque
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Track model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    mae: float
    r2_score: float
    prediction_count: int
    last_updated: str


@dataclass
class TradeOutcome:
    """Track trade outcomes for learning"""
    symbol: str
    prediction: float
    actual_outcome: float
    features: List[float]
    timestamp: str
    profit_loss: float
    hold_duration: int


class ContinuousLearningSystem:
    """
    Implements online learning with performance feedback and model adaptation
    """
    
    def __init__(self, model_save_path: str = "models/"):
        self.model_save_path = model_save_path
        self.models = {}
        self.scalers = {}
        self.performance_history = {}
        self.trade_outcomes = deque(maxlen=1000)  # Keep last 1000 trades
        
        # Learning parameters
        self.min_samples_for_update = 10
        self.performance_window = 50
        self.model_retrain_threshold = 0.1  # Retrain if performance drops 10%
        self.learning_rate = 0.01
        
        # Model ensemble
        self.ensemble_weights = {}
        self.model_types = ['rf', 'gb', 'sgd']  # Random Forest, Gradient Boosting, SGD
        
        # Performance tracking
        self.baseline_performance = {}
        self.current_performance = {}
        self.adaptation_history = []
        
        self._initialize_models()
        logger.info("✅ Continuous Learning System initialized")
    
    def _initialize_models(self):
        """Initialize the ensemble of models"""
        try:
            # Random Forest - good for feature importance and non-linear patterns
            self.models['rf'] = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Gradient Boosting - good for sequential learning
            self.models['gb'] = GradientBoostingRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            # SGD Regressor - good for online learning
            self.models['sgd'] = SGDRegressor(
                learning_rate='adaptive',
                eta0=0.01,
                random_state=42
            )
            
            # Initialize scalers
            for model_type in self.model_types:
                self.scalers[model_type] = StandardScaler()
                self.ensemble_weights[model_type] = 1.0 / len(self.model_types)
            
            logger.info(f"Initialized {len(self.models)} models in ensemble")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def train_initial_models(self, training_data: pd.DataFrame, target_column: str):
        """Train initial models with historical data"""
        try:
            if len(training_data) < 100:
                logger.warning(f"Insufficient training data: {len(training_data)} samples")
                return False
            
            # Prepare features and target
            feature_columns = [col for col in training_data.columns if col != target_column]
            X = training_data[feature_columns].fillna(0)
            y = training_data[target_column].fillna(0)
            
            # Split for validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train each model
            for model_type in self.model_types:
                try:
                    # Scale features
                    X_train_scaled = self.scalers[model_type].fit_transform(X_train)
                    X_val_scaled = self.scalers[model_type].transform(X_val)
                    
                    # Train model
                    self.models[model_type].fit(X_train_scaled, y_train)
                    
                    # Evaluate on validation set
                    y_pred = self.models[model_type].predict(X_val_scaled)
                    performance = self._calculate_performance_metrics(y_val, y_pred)
                    
                    self.baseline_performance[model_type] = performance
                    self.current_performance[model_type] = performance
                    
                    logger.info(f"Model {model_type} trained - R²: {performance.r2_score:.3f}, MSE: {performance.mse:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_type}: {e}")
                    continue
            
            # Save initial models
            self._save_models()
            
            logger.info("✅ Initial model training completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in initial training: {e}")
            return False
    
    def predict_ensemble(self, features: List[float]) -> Dict:
        """Make prediction using ensemble of models"""
        try:
            features_array = np.array(features).reshape(1, -1)
            predictions = {}
            weights = []
            valid_predictions = []
            
            # Get predictions from each model
            for model_type in self.model_types:
                try:
                    # Scale features
                    features_scaled = self.scalers[model_type].transform(features_array)
                    
                    # Make prediction
                    pred = self.models[model_type].predict(features_scaled)[0]
                    predictions[model_type] = pred
                    
                    # Weight by current performance
                    weight = self.ensemble_weights[model_type]
                    weights.append(weight)
                    valid_predictions.append(pred)
                    
                except Exception as e:
                    logger.warning(f"Error predicting with {model_type}: {e}")
                    continue
            
            if not valid_predictions:
                logger.error("No valid predictions from ensemble")
                return {'ensemble_prediction': 0.0, 'model_predictions': {}, 'confidence': 0.0}
            
            # Calculate weighted ensemble prediction
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights
            
            ensemble_pred = np.average(valid_predictions, weights=weights)
            
            # Calculate confidence based on agreement between models
            pred_std = np.std(valid_predictions)
            confidence = max(0.0, 1.0 - pred_std)  # Higher agreement = higher confidence
            
            return {
                'ensemble_prediction': float(ensemble_pred),
                'model_predictions': predictions,
                'confidence': float(confidence),
                'prediction_count': len(valid_predictions)
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {'ensemble_prediction': 0.0, 'model_predictions': {}, 'confidence': 0.0}
    
    def update_with_outcome(self, trade_outcome: TradeOutcome):
        """Update models with actual trade outcome"""
        try:
            # Store trade outcome
            self.trade_outcomes.append(trade_outcome)

            # Check if we have enough samples for updating
            if len(self.trade_outcomes) < self.min_samples_for_update:
                return

            # Prepare recent training data
            recent_outcomes = list(self.trade_outcomes)[-self.performance_window:]

            X_new = np.array([outcome.features for outcome in recent_outcomes])
            y_new = np.array([outcome.actual_outcome for outcome in recent_outcomes])

            # Online learning update for SGD model
            try:
                if len(X_new) > 0:
                    X_new_scaled = self.scalers['sgd'].transform(X_new)
                    self.models['sgd'].partial_fit(X_new_scaled, y_new)

                    # Update performance
                    y_pred = self.models['sgd'].predict(X_new_scaled)
                    performance = self._calculate_performance_metrics(y_new, y_pred)
                    self.current_performance['sgd'] = performance

            except Exception as e:
                logger.warning(f"Error updating SGD model: {e}")

            # Check if models need retraining
            self._check_model_performance()

            # Update ensemble weights based on recent performance
            self._update_ensemble_weights()

            logger.debug(f"Updated models with trade outcome: {trade_outcome.symbol}")

        except Exception as e:
            logger.error(f"Error updating with outcome: {e}")
    
    def _calculate_performance_metrics(self, y_true, y_pred) -> ModelPerformance:
        """Calculate comprehensive performance metrics"""
        try:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # For classification metrics, convert to binary
            y_true_binary = (y_true > 0).astype(int)
            y_pred_binary = (y_pred > 0).astype(int)
            
            # Calculate precision, recall, f1
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = np.mean(y_true_binary == y_pred_binary)
            
            return ModelPerformance(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                mse=mse,
                mae=mae,
                r2_score=r2,
                prediction_count=len(y_true),
                last_updated=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return ModelPerformance(0, 0, 0, 0, float('inf'), float('inf'), -1, 0, datetime.now().isoformat())
    
    def _check_model_performance(self):
        """Check if models need retraining due to performance degradation"""
        try:
            for model_type in self.model_types:
                if model_type not in self.baseline_performance or model_type not in self.current_performance:
                    continue
                
                baseline_r2 = self.baseline_performance[model_type].r2_score
                current_r2 = self.current_performance[model_type].r2_score
                
                # Check for significant performance drop
                if baseline_r2 > 0 and (baseline_r2 - current_r2) > self.model_retrain_threshold:
                    logger.warning(f"Model {model_type} performance degraded: {baseline_r2:.3f} → {current_r2:.3f}")
                    self._schedule_model_retrain(model_type)
                    
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
    
    def _schedule_model_retrain(self, model_type: str):
        """Schedule model retraining (simplified implementation)"""
        try:
            # In a full implementation, this would trigger async retraining
            logger.info(f"Scheduling retrain for {model_type} model")
            
            # For now, just reset the baseline to current performance
            # to avoid repeated retraining notifications
            if model_type in self.current_performance:
                self.baseline_performance[model_type] = self.current_performance[model_type]
                
        except Exception as e:
            logger.error(f"Error scheduling retrain: {e}")
    
    def _update_ensemble_weights(self):
        """Update ensemble weights based on recent performance"""
        try:
            total_performance = 0
            model_scores = {}
            
            # Calculate performance scores (higher is better)
            for model_type in self.model_types:
                if model_type in self.current_performance:
                    # Use combination of metrics for scoring
                    perf = self.current_performance[model_type]
                    score = (perf.accuracy + perf.f1_score + max(0, perf.r2_score)) / 3
                    model_scores[model_type] = max(0.01, score)  # Minimum weight
                    total_performance += model_scores[model_type]
                else:
                    model_scores[model_type] = 0.01
            
            # Normalize weights
            if total_performance > 0:
                for model_type in self.model_types:
                    self.ensemble_weights[model_type] = model_scores[model_type] / total_performance
            
            logger.debug(f"Updated ensemble weights: {self.ensemble_weights}")
            
        except Exception as e:
            logger.error(f"Error updating ensemble weights: {e}")
    
    def get_learning_insights(self) -> Dict:
        """Get insights about the learning system's performance"""
        try:
            insights = {
                'total_trades_learned': len(self.trade_outcomes),
                'models_performance': {},
                'ensemble_weights': self.ensemble_weights.copy(),
                'recent_accuracy': self._calculate_recent_accuracy(),
                'learning_trend': self._analyze_learning_trend(),
                'feature_importance': self._get_feature_importance()
            }
            
            # Add performance metrics for each model
            for model_type in self.model_types:
                if model_type in self.current_performance:
                    insights['models_performance'][model_type] = asdict(self.current_performance[model_type])
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {}
    
    def _calculate_recent_accuracy(self) -> float:
        """Calculate accuracy on recent predictions"""
        try:
            if len(self.trade_outcomes) < 10:
                return 0.0
            
            recent_outcomes = list(self.trade_outcomes)[-20:]
            correct_predictions = 0
            
            for outcome in recent_outcomes:
                predicted_direction = 1 if outcome.prediction > 0 else -1
                actual_direction = 1 if outcome.actual_outcome > 0 else -1
                
                if predicted_direction == actual_direction:
                    correct_predictions += 1
            
            return correct_predictions / len(recent_outcomes)
            
        except Exception as e:
            logger.error(f"Error calculating recent accuracy: {e}")
            return 0.0
    
    def _analyze_learning_trend(self) -> str:
        """Analyze if the model is improving over time"""
        try:
            if len(self.trade_outcomes) < 20:
                return "Insufficient data"
            
            # Compare first half vs second half accuracy
            all_outcomes = list(self.trade_outcomes)
            mid_point = len(all_outcomes) // 2
            
            first_half = all_outcomes[:mid_point]
            second_half = all_outcomes[mid_point:]
            
            first_accuracy = self._calculate_accuracy_for_outcomes(first_half)
            second_accuracy = self._calculate_accuracy_for_outcomes(second_half)
            
            if second_accuracy > first_accuracy + 0.05:
                return "Improving"
            elif second_accuracy < first_accuracy - 0.05:
                return "Declining"
            else:
                return "Stable"
                
        except Exception as e:
            logger.error(f"Error analyzing learning trend: {e}")
            return "Unknown"
    
    def _calculate_accuracy_for_outcomes(self, outcomes: List[TradeOutcome]) -> float:
        """Calculate accuracy for a list of trade outcomes"""
        if not outcomes:
            return 0.0
        
        correct = 0
        for outcome in outcomes:
            predicted_direction = 1 if outcome.prediction > 0 else -1
            actual_direction = 1 if outcome.actual_outcome > 0 else -1
            if predicted_direction == actual_direction:
                correct += 1
        
        return correct / len(outcomes)
    
    def _get_feature_importance(self) -> Dict:
        """Get feature importance from tree-based models"""
        try:
            importance_dict = {}
            
            # Get importance from Random Forest
            if 'rf' in self.models and hasattr(self.models['rf'], 'feature_importances_'):
                rf_importance = self.models['rf'].feature_importances_
                importance_dict['random_forest'] = rf_importance.tolist()
            
            # Get importance from Gradient Boosting
            if 'gb' in self.models and hasattr(self.models['gb'], 'feature_importances_'):
                gb_importance = self.models['gb'].feature_importances_
                importance_dict['gradient_boosting'] = gb_importance.tolist()
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def _save_models(self):
        """Save models and scalers to disk"""
        try:
            import os
            os.makedirs(self.model_save_path, exist_ok=True)
            
            # Save models
            for model_type, model in self.models.items():
                model_path = f"{self.model_save_path}/{model_type}_model.joblib"
                joblib.dump(model, model_path)
            
            # Save scalers
            for scaler_type, scaler in self.scalers.items():
                scaler_path = f"{self.model_save_path}/{scaler_type}_scaler.joblib"
                joblib.dump(scaler, scaler_path)
            
            # Save performance history
            performance_data = {
                'baseline_performance': {k: asdict(v) for k, v in self.baseline_performance.items()},
                'current_performance': {k: asdict(v) for k, v in self.current_performance.items()},
                'ensemble_weights': self.ensemble_weights
            }
            
            with open(f"{self.model_save_path}/performance_history.json", 'w') as f:
                json.dump(performance_data, f, indent=2)
            
            logger.info("Models and performance data saved")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load models and scalers from disk"""
        try:
            import os
            
            # Load models
            for model_type in self.model_types:
                model_path = f"{self.model_save_path}/{model_type}_model.joblib"
                if os.path.exists(model_path):
                    self.models[model_type] = joblib.load(model_path)
            
            # Load scalers
            for scaler_type in self.model_types:
                scaler_path = f"{self.model_save_path}/{scaler_type}_scaler.joblib"
                if os.path.exists(scaler_path):
                    self.scalers[scaler_type] = joblib.load(scaler_path)
            
            # Load performance history
            performance_path = f"{self.model_save_path}/performance_history.json"
            if os.path.exists(performance_path):
                with open(performance_path, 'r') as f:
                    performance_data = json.load(f)
                
                # Reconstruct performance objects
                for model_type, perf_dict in performance_data.get('baseline_performance', {}).items():
                    self.baseline_performance[model_type] = ModelPerformance(**perf_dict)
                
                for model_type, perf_dict in performance_data.get('current_performance', {}).items():
                    self.current_performance[model_type] = ModelPerformance(**perf_dict)
                
                self.ensemble_weights = performance_data.get('ensemble_weights', self.ensemble_weights)
            
            logger.info("Models and performance data loaded")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def reset_learning(self):
        """Reset the learning system"""
        try:
            self.trade_outcomes.clear()
            self.current_performance.clear()
            self.baseline_performance.clear()
            
            # Reset ensemble weights to equal
            for model_type in self.model_types:
                self.ensemble_weights[model_type] = 1.0 / len(self.model_types)
            
            logger.info("Learning system reset")
            
        except Exception as e:
            logger.error(f"Error resetting learning system: {e}")


# Global instance
_learning_system = None

def get_learning_system() -> ContinuousLearningSystem:
    """Get the global continuous learning system instance"""
    global _learning_system
    if _learning_system is None:
        _learning_system = ContinuousLearningSystem()
    return _learning_system
"""
Phase 2: Continuous Learning System
Implements online model updating and performance feedback for adaptive trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import logging
from collections import deque
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Track model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    mae: float
    r2_score: float
    prediction_count: int
    last_updated: str


@dataclass
class TradeOutcome:
    """Track trade outcomes for learning"""
    symbol: str
    prediction: float
    actual_outcome: float
    features: List[float]
    timestamp: str
    profit_loss: float
    hold_duration: int


class ContinuousLearningSystem:
    """
    Implements online learning with performance feedback and model adaptation
    """
    
    def __init__(self, model_save_path: str = "models/"):
        self.model_save_path = model_save_path
        self.models = {}
        self.scalers = {}
        self.performance_history = {}
        self.trade_outcomes = deque(maxlen=1000)  # Keep last 1000 trades
        
        # Learning parameters
        self.min_samples_for_update = 10
        self.performance_window = 50
        self.model_retrain_threshold = 0.1  # Retrain if performance drops 10%
        self.learning_rate = 0.01
        
        # Model ensemble
        self.ensemble_weights = {}
        self.model_types = ['rf', 'gb', 'sgd']  # Random Forest, Gradient Boosting, SGD
        
        # Performance tracking
        self.baseline_performance = {}
        self.current_performance = {}
        self.adaptation_history = []
        
        self._initialize_models()
        logger.info("✅ Continuous Learning System initialized")
    
    def _initialize_models(self):
        """Initialize the ensemble of models"""
        try:
            # Random Forest - good for feature importance and non-linear patterns
            self.models['rf'] = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Gradient Boosting - good for sequential learning
            self.models['gb'] = GradientBoostingRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            # SGD Regressor - good for online learning
            self.models['sgd'] = SGDRegressor(
                learning_rate='adaptive',
                eta0=0.01,
                random_state=42
            )
            
            # Initialize scalers
            for model_type in self.model_types:
                self.scalers[model_type] = StandardScaler()
                self.ensemble_weights[model_type] = 1.0 / len(self.model_types)
            
            logger.info(f"Initialized {len(self.models)} models in ensemble")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def train_initial_models(self, training_data: pd.DataFrame, target_column: str):
        """Train initial models with historical data"""
        try:
            if len(training_data) < 100:
                logger.warning(f"Insufficient training data: {len(training_data)} samples")
                return False
            
            # Prepare features and target
            feature_columns = [col for col in training_data.columns if col != target_column]
            X = training_data[feature_columns].fillna(0)
            y = training_data[target_column].fillna(0)
            
            # Split for validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train each model
            for model_type in self.model_types:
                try:
                    # Scale features
                    X_train_scaled = self.scalers[model_type].fit_transform(X_train)
                    X_val_scaled = self.scalers[model_type].transform(X_val)
                    
                    # Train model
                    self.models[model_type].fit(X_train_scaled, y_train)
                    
                    # Evaluate on validation set
                    y_pred = self.models[model_type].predict(X_val_scaled)
                    performance = self._calculate_performance_metrics(y_val, y_pred)
                    
                    self.baseline_performance[model_type] = performance
                    self.current_performance[model_type] = performance
                    
                    logger.info(f"Model {model_type} trained - R²: {performance.r2_score:.3f}, MSE: {performance.mse:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_type}: {e}")
                    continue
            
            # Save initial models
            self._save_models()
            
            logger.info("✅ Initial model training completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in initial training: {e}")
            return False
    
    def predict_ensemble(self, features: List[float]) -> Dict:
        """Make prediction using ensemble of models"""
        try:
            features_array = np.array(features).reshape(1, -1)
            predictions = {}
            weights = []
            valid_predictions = []
            
            # Get predictions from each model
            for model_type in self.model_types:
                try:
                    # Scale features
                    features_scaled = self.scalers[model_type].transform(features_array)
                    
                    # Make prediction
                    pred = self.models[model_type].predict(features_scaled)[0]
                    predictions[model_type] = pred
                    
                    # Weight by current performance
                    weight = self.ensemble_weights[model_type]
                    weights.append(weight)
                    valid_predictions.append(pred)
                    
                except Exception as e:
                    logger.warning(f"Error predicting with {model_type}: {e}")
                    continue
            
            if not valid_predictions:
                logger.error("No valid predictions from ensemble")
                return {'ensemble_prediction': 0.0, 'model_predictions': {}, 'confidence': 0.0}
            
            # Calculate weighted ensemble prediction
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights
            
            ensemble_pred = np.average(valid_predictions, weights=weights)
            
            # Calculate confidence based on agreement between models
            pred_std = np.std(valid_predictions)
            confidence = max(0.0, 1.0 - pred_std)  # Higher agreement = higher confidence
            
            return {
                'ensemble_prediction': float(ensemble_pred),
                'model_predictions': predictions,
                'confidence': float(confidence),
                'prediction_count': len(valid_predictions)
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {'ensemble_prediction': 0.0, 'model_predictions': {}, 'confidence': 0.0}
    
    def update_with_outcome(self, trade_outcome: TradeOutcome):
        """Update models with actual trade outcome"""
        try:
            # Store trade outcome
            self.trade_outcomes.append(trade_outcome)
            
            # Check if we have enough samples for updating
            if len(self.trade_outcomes) < self.min_samples_for_update:
                return
            
            # Prepare recent training data
            recent_outcomes = list(self.trade_outcomes)[-self.performance_window:]
            
            X_new = np.array([outcome.features for outcome in recent_outcomes])
            y_new = np.array([outcome.actual_outcome for outcome in recent_outcomes])
            
            # Online learning update for SGD model
            try:               
                X_new_scaled = self.scalers['sgd'].transform(X_new)             
                self.models['sgd'].partial_fit(X_new_scaled, y_new)
                # Update performance\n                
                y_pred = self.models['sgd'].predict(X_new_scaled)
                performance = self._calculate_performance_metrics(y_new, y_pred)
                self.current_performance['sgd'] = performance
            except Exception as e:
                logger.warning()
        except Exception as e:
                logger.warning()