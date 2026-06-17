import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib
import os

class CognitiveLoadModel:
    def __init__(self):
        # ✅ Create data directory
        os.makedirs('data', exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = [
            'total_session_time', 'runs_per_minute', 'avg_typing_speed', 'typing_variance',
            'avg_code_length', 'code_length_std', 'max_code_length', 'error_rate',
            'success_rate', 'lines_of_code', 'unique_tokens', 'function_count',
            'class_count', 'control_flow', 'halstead_volume', 'fixation_ratio', 'pause_frequency'
        ]
        self.evaluation_metrics = {}
        self._load_or_train_model()
    
    def _generate_synthetic_dataset(self, n_samples=5000) -> pd.DataFrame:
        """Generate realistic synthetic dataset"""
        np.random.seed(42)
        
        data = {
            'total_session_time': np.random.exponential(300, n_samples),
            'runs_per_minute': np.random.uniform(0.5, 5, n_samples),
            'avg_typing_speed': np.random.uniform(20, 200, n_samples),
            'typing_variance': np.random.uniform(0.1, 2, n_samples),
            'avg_code_length': np.random.uniform(50, 500, n_samples),
            'code_length_std': np.random.uniform(10, 100, n_samples),
            'max_code_length': np.random.uniform(100, 1000, n_samples),
            'error_rate': np.random.uniform(0, 0.8, n_samples),
            'success_rate': np.random.uniform(0.2, 1.0, n_samples),
            'lines_of_code': np.random.uniform(10, 100, n_samples),
            'unique_tokens': np.random.uniform(20, 200, n_samples),
            'function_count': np.random.poisson(1, n_samples),
            'class_count': np.random.poisson(0.3, n_samples),
            'control_flow': np.random.poisson(2, n_samples),
            'halstead_volume': np.random.uniform(100, 2000, n_samples),
            'fixation_ratio': np.random.uniform(0, 0.5, n_samples),
            'pause_frequency': np.random.uniform(0, 0.3, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Realistic label generation
        conditions = [
            (df['error_rate'] < 0.1) & (df['pause_frequency'] < 0.05) & (df['avg_typing_speed'] > 120),
            (df['error_rate'] > 0.4) & (df['pause_frequency'] > 0.15) & (df['avg_typing_speed'] < 80),
        ]
        choices = [0, 2]  # Low=0, High=2
        
        df['cognitive_load_raw'] = np.select(conditions, choices, default=1)
        df['cognitive_load'] = df['cognitive_load_raw'].astype(int)
        
        return df
    
    def _train_model(self):
        """Train XGBoost model"""
        print("Training cognitive load model...")
        
        df = self._generate_synthetic_dataset(5000)
        X = df[self.feature_names]
        y = df['cognitive_load']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        self.evaluation_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Save model files
        joblib.dump(self.model, 'data/model.pkl')
        joblib.dump(self.scaler, 'data/scaler.pkl')
        joblib.dump(self.label_encoder.fit(['Low', 'Medium', 'High']), 'data/label_encoder.pkl')
        
        print(f"✅ Model trained - Test Accuracy: {self.evaluation_metrics['accuracy']:.4f}")
    
    def _load_or_train_model(self):
        """Load model or train new one"""
        if os.path.exists('data/model.pkl'):
            try:
                self.model = joblib.load('data/model.pkl')
                self.scaler = joblib.load('data/scaler.pkl')
                self.label_encoder = joblib.load('data/label_encoder.pkl')
                print("✅ Model loaded successfully")
            except:
                print("⚠️  Failed to load model, training new one...")
                self._train_model()
        else:
            self._train_model()
    
    def predict(self, features: dict) -> tuple:
        """Predict cognitive load"""
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names]).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        prediction = self.model.predict(feature_vector_scaled)[0]
        probabilities = self.model.predict_proba(feature_vector_scaled)[0]
        confidence = np.max(probabilities)
        
        load_level = self.label_encoder.inverse_transform([int(prediction)])[0]
        return load_level, confidence
    
def get_evaluation_metrics(self) -> dict:
    """Get model evaluation metrics - robust version"""
    if hasattr(self, 'evaluation_metrics') and self.evaluation_metrics:
        return self.evaluation_metrics.copy()
    
    # Fallback: Quick evaluation on synthetic test set
    print("🔄 Running quick evaluation...")
    df = self._generate_synthetic_dataset(1000)
    X = df[self.feature_names]
    y = df['cognitive_load']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = self.scaler.transform(X_test)
    
    y_pred = self.model.predict(X_test_scaled)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    self.evaluation_metrics = metrics
    return metrics

    # ✅ ADDED: Comprehensive evaluation method
    def comprehensive_evaluation(self, test_size=0.2, n_trials=5):
        """Research-grade cross-validation evaluation"""
        print(f"Running {n_trials}-trial cross-validation...")
        
        results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'confusion_matrices': []
        }
        
        for i in range(n_trials):
            df = self._generate_synthetic_dataset(2000)
            X = df[self.feature_names]
            y = df['cognitive_load']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42+i, stratify=y
            )
            
            scaler_trial = StandardScaler()
            X_train_scaled = scaler_trial.fit_transform(X_train)
            X_test_scaled = scaler_trial.transform(X_test)
            
            trial_model = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42+i
            )
            trial_model.fit(X_train_scaled, y_train)
            
            y_pred = trial_model.predict(X_test_scaled)
            
            results['accuracy'].append(accuracy_score(y_test, y_pred))
            results['precision'].append(precision_score(y_test, y_pred, average='weighted'))
            results['recall'].append(recall_score(y_test, y_pred, average='weighted'))
            results['f1'].append(f1_score(y_test, y_pred, average='weighted'))
            results['confusion_matrices'].append(confusion_matrix(y_test, y_pred).tolist())
            
            print(f"  Trial {i+1}: Accuracy = {results['accuracy'][-1]:.4f}")
        
        return {
            'mean_accuracy': np.mean(results['accuracy']),
            'std_accuracy': np.std(results['accuracy']),
            'mean_f1': np.mean(results['f1']),
            'detailed_results': results
        }

# ✅ Test the model
if __name__ == "__main__":
    model = CognitiveLoadModel()
    print("✅ CognitiveLoadModel ready!")
    print(f"Feature count: {len(model.feature_names)}")