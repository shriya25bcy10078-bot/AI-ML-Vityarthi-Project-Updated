import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Import scikit-learn modules
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
    print("✓ scikit-learn imported successfully")
except ImportError as e:
    SKLEARN_AVAILABLE = False
    print(f"✗ scikit-learn not available: {e}")

class ExpensePredictor:
    def __init__(self):
        if SKLEARN_AVAILABLE:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.vectorizer = TfidfVectorizer(max_features=100)
            self.scaler = StandardScaler()
        else:
            self.model = None
            self.vectorizer = None
            self.scaler = None
            
        self.categories = ['Food', 'Transportation', 'Entertainment', 'Shopping', 'Bills', 'Healthcare', 'Other']
        self.model_path = 'data/models/expense_classifier.pkl'
        self.load_model()
    
    def load_model(self):
        """Load trained model or create new one"""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print("Model loaded successfully")
            except:
                if SKLEARN_AVAILABLE:
                    self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    self.model = None
        else:
            if SKLEARN_AVAILABLE:
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            os.makedirs('data/models', exist_ok=True)
    
    def prepare_features(self, descriptions, amounts):
        """Prepare features for model training/prediction"""
        if not SKLEARN_AVAILABLE:
            return np.array(amounts).reshape(-1, 1)
            
        # Text features
        if len(descriptions) > 0:
            text_features = self.vectorizer.fit_transform(descriptions).toarray()
        else:
            text_features = np.zeros((len(descriptions), 100))
        
        # Numerical features
        amount_features = self.scaler.fit_transform(np.array(amounts).reshape(-1, 1))
        
        # Combine features
        features = np.hstack([text_features, amount_features])
        return features
    
    def train_model(self, expenses_df):
        """Train the category prediction model"""
        try:
            if len(expenses_df) < 5:
                print("Insufficient data for training")
                return
            
            if not SKLEARN_AVAILABLE:
                print("scikit-learn not available - using rule-based classification")
                return
            
            # Prepare features
            descriptions = expenses_df['description'].fillna('').astype(str).tolist()
            amounts = expenses_df['amount'].tolist()
            features = self.prepare_features(descriptions, amounts)
            
            # Prepare labels
            labels = expenses_df['category'].apply(
                lambda x: x if x in self.categories else 'Other'
            ).tolist()
            
            # Train model
            self.model.fit(features, labels)
            
            # Save model
            joblib.dump(self.model, self.model_path)
            print("Model trained and saved successfully")
            
        except Exception as e:
            print(f"Error training model: {e}")
    
    def predict_category(self, description, amount):
        """Predict category for new expense"""
        try:
            if not SKLEARN_AVAILABLE:
                return self._rule_based_prediction(description, amount)
            
            if self.model is None:
                return 'Other'
            
            # Prepare features for single prediction
            text_features = self.vectorizer.transform([description]).toarray()
            amount_features = self.scaler.transform([[amount]])
            features = np.hstack([text_features, amount_features])
            
            prediction = self.model.predict(features)[0]
            return prediction
        except:
            return self._rule_based_prediction(description, amount)
    
    def _rule_based_prediction(self, description, amount):
        """Rule-based fallback when ML is not available"""
        description_lower = description.lower()
        
        category_keywords = {
            'Food': ['restaurant', 'lunch', 'dinner', 'food', 'groceries', 'coffee', 'cafe', 'meal', 'breakfast'],
            'Transportation': ['uber', 'taxi', 'gas', 'bus', 'train', 'transport', 'fuel', 'metro', 'subway', 'parking'],
            'Entertainment': ['movie', 'netflix', 'game', 'concert', 'entertainment', 'theater', 'cinema', 'music'],
            'Shopping': ['shop', 'store', 'buy', 'purchase', 'mall', 'amazon', 'ebay', 'walmart', 'target'],
            'Bills': ['bill', 'invoice', 'payment', 'subscription', 'electricity', 'water', 'internet', 'phone', 'rent'],
            'Healthcare': ['doctor', 'hospital', 'medicine', 'pharmacy', 'medical', 'clinic', 'dental']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                return category
        
        # Amount-based fallback
        if amount > 200:
            return 'Shopping'
        elif amount > 100:
            return 'Bills'
        elif amount > 50:
            return 'Food'
        else:
            return 'Other'
    
    def retrain_model(self):
        """Retrain model with latest data"""
        print("Retraining model...")
        # Implementation would load latest data and retrain
    
    def generate_insights(self, expenses_df):
        """Generate spending insights"""
        insights = {}
        
        try:
            # Basic statistics
            total_spent = expenses_df['amount'].sum()
            avg_daily = expenses_df.groupby('date')['amount'].sum().mean()
            most_expensive = expenses_df.loc[expenses_df['amount'].idxmax()]
            
            # Category analysis
            category_stats = expenses_df.groupby('category')['amount'].agg(['sum', 'count', 'mean'])
            
            insights = {
                'total_spent': total_spent,
                'avg_daily': avg_daily,
                'most_expensive_category': category_stats['sum'].idxmax(),
                'most_frequent_category': category_stats['count'].idxmax(),
                'category_breakdown': category_stats.to_dict(),
                'suggestions': self.generate_suggestions(expenses_df)
            }
        except Exception as e:
            insights = {'error': str(e)}
        
        return insights
    
    def generate_suggestions(self, expenses_df):
        """Generate personalized suggestions"""
        suggestions = []
        
        try:
            # Analyze spending patterns
            daily_spending = expenses_df.groupby('date')['amount'].sum()
            if len(daily_spending) > 7:
                recent_avg = daily_spending.tail(7).mean()
                previous_avg = daily_spending.head(len(daily_spending)-7).mean()
                
                if recent_avg > previous_avg * 1.2:
                    suggestions.append("Your spending has increased by 20% in the last week. Consider reviewing your expenses.")
                elif recent_avg < previous_avg * 0.8:
                    suggestions.append("Great! Your spending has decreased by 20% in the last week.")
            
            # Category-based suggestions
            category_totals = expenses_df.groupby('category')['amount'].sum()
            if 'Entertainment' in category_totals and category_totals['Entertainment'] > category_totals.get('Food', 0):
                suggestions.append("You're spending more on entertainment than food. Consider balancing your budget.")
            
            # High spending alert
            if total_spent := expenses_df['amount'].sum() > 1000:
                suggestions.append(f"You've spent ${total_spent:.2f} this period. Consider setting a budget.")
                
        except Exception as e:
            suggestions = [f"Error generating suggestions: {e}"]
        
        return suggestions if suggestions else ["Your spending patterns look healthy! Keep tracking your expenses."]
    
    def analyze_patterns(self, expenses_df):
        """Analyze spending patterns using clustering"""
        try:
            if len(expenses_df) < 10:
                return {"message": "Need more data for pattern analysis"}
            
            if not SKLEARN_AVAILABLE:
                return {"message": "scikit-learn required for pattern analysis"}
            
            # Feature engineering for clustering
            expenses_df['day_of_week'] = pd.to_datetime(expenses_df['date']).dt.dayofweek
            expenses_df['is_weekend'] = expenses_df['day_of_week'].isin([5, 6]).astype(int)
            
            features = pd.get_dummies(expenses_df[['category', 'day_of_week']])
            features['amount'] = self.scaler.fit_transform(expenses_df[['amount']])
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=min(3, len(expenses_df)), random_state=42)
            clusters = kmeans.fit_predict(features)
            
            expenses_df['cluster'] = clusters
            
            # Analyze clusters
            cluster_analysis = expenses_df.groupby('cluster').agg({
                'amount': ['mean', 'count'],
                'category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
                'is_weekend': 'mean'
            }).round(2)
            
            return {
                'clusters_found': len(cluster_analysis),
                'cluster_details': cluster_analysis.to_dict(),
                'patterns': self.interpret_clusters(cluster_analysis)
            }
            
        except Exception as e:
            return {'error': f"Pattern analysis failed: {e}"}
    
    def interpret_clusters(self, cluster_analysis):
        """Interpret clustering results"""
        patterns = []
        for cluster_id in cluster_analysis.index:
            cluster_data = cluster_analysis.loc[cluster_id]
            avg_amount = cluster_data[('amount', 'mean')]
            weekend_ratio = cluster_data[('is_weekend', 'mean')]
            main_category = cluster_data[('category', '<lambda>')]
            
            pattern_desc = f"Cluster {cluster_id}: Mostly {main_category} expenses "
            pattern_desc += f"(avg: ${avg_amount:.2f})"
            
            if weekend_ratio > 0.6:
                pattern_desc += " - Primarily weekend spending"
            elif weekend_ratio < 0.4:
                pattern_desc += " - Primarily weekday spending"
                
            patterns.append(pattern_desc)
        
        return patterns

class AnomalyDetector:
    def __init__(self):
        if SKLEARN_AVAILABLE:
            self.model = IsolationForest(contamination=0.1, random_state=42)
            self.scaler = StandardScaler()
        else:
            self.model = None
            self.scaler = None
    
    def detect_anomalies(self, expenses_df):
        """Detect anomalous expenses"""
        try:
            if len(expenses_df) < 10:
                return {"message": "Need at least 10 expenses for anomaly detection"}
            
            if not SKLEARN_AVAILABLE:
                return self._statistical_anomaly_detection(expenses_df)
            
            # Prepare features for anomaly detection
            features = expenses_df[['amount']].copy()
            features['day_of_week'] = pd.to_datetime(expenses_df['date']).dt.dayofweek
            
            # One-hot encode categories
            category_dummies = pd.get_dummies(expenses_df['category'])
            features = pd.concat([features, category_dummies], axis=1)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Detect anomalies
            anomalies = self.model.fit_predict(scaled_features)
            expenses_df['is_anomaly'] = anomalies == -1
            
            anomalous_expenses = expenses_df[expenses_df['is_anomaly']].to_dict('records')
            
            return {
                'total_anomalies': len(anomalous_expenses),
                'anomalous_expenses': anomalous_expenses,
                'anomaly_rate': len(anomalous_expenses) / len(expenses_df),
                'method_used': 'isolation_forest'
            }
            
        except Exception as e:
            return {'error': f"Anomaly detection failed: {e}"}
    
    def _statistical_anomaly_detection(self, expenses_df):
        """Statistical method for anomaly detection when scikit-learn is not available"""
        amounts = expenses_df['amount'].values
        mean_amount = np.mean(amounts)
        std_amount = np.std(amounts)
        
        # Mark amounts > 2 standard deviations as anomalies
        z_scores = np.abs((amounts - mean_amount) / std_amount)
        anomalies = z_scores > 2
        
        anomalous_expenses = expenses_df[anomalies].to_dict('records')
        
        return {
            'total_anomalies': len(anomalous_expenses),
            'anomalous_expenses': anomalous_expenses,
            'anomaly_rate': len(anomalous_expenses) / len(expenses_df),
            'method_used': 'statistical_zscore'
        }

class ForecastModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
    def prepare_forecast_data(self, expenses_df):
        """Prepare data for time series forecasting"""
        try:
            # Aggregate daily spending
            daily_spending = expenses_df.groupby('date')['amount'].sum().sort_index()
            
            # Create features for forecasting
            dates = pd.to_datetime(daily_spending.index)
            spending_values = daily_spending.values
            
            if len(spending_values) < 7:
                return None, None
            
            # Create sequences for forecasting
            sequence_length = 7
            X, y = [], []
            
            for i in range(len(spending_values) - sequence_length):
                X.append(spending_values[i:i + sequence_length])
                y.append(spending_values[i + sequence_length])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            print(f"Error preparing forecast data: {e}")
            return None, None
    
    def predict_spending(self, expenses_df, days=7):
        """Predict future spending using available methods"""
        try:
            if len(expenses_df) < 7:
                return {"message": "Need at least 1 week of data for forecasting"}
            
            if not SKLEARN_AVAILABLE:
                return self._simple_forecast(expenses_df, days)
            
            return self._ml_forecast(expenses_df, days)
            
        except Exception as e:
            return {'error': f"Forecasting failed: {e}"}
    
    def _simple_forecast(self, expenses_df, days):
        """Simple forecasting using moving averages"""
        daily_spending = expenses_df.groupby('date')['amount'].sum()
        values = daily_spending.values
        
        # Use weighted average (recent days have more weight)
        if len(values) >= 7:
            weights = np.arange(1, len(values) + 1)
            recent_avg = np.average(values[-7:], weights=weights[-7:])
        else:
            recent_avg = np.mean(values)
        
        # Add some random variation
        variation = recent_avg * 0.1  # 10% variation
        predictions = [max(0, recent_avg + np.random.uniform(-variation, variation)) for _ in range(days)]
        
        trend = 'increasing' if predictions[-1] > predictions[0] else 'decreasing'
        if abs(predictions[-1] - predictions[0]) < recent_avg * 0.05:
            trend = 'stable'
        
        return {
            'predictions': predictions,
            'next_week_total': sum(predictions),
            'confidence': 0.7,
            'trend': trend,
            'method_used': 'moving_average'
        }
    
    def _ml_forecast(self, expenses_df, days):
        """Machine learning forecasting using scikit-learn"""
        try:
            X, y = self.prepare_forecast_data(expenses_df)
            if X is None:
                return self._simple_forecast(expenses_df, days)
            
            daily_spending = expenses_df.groupby('date')['amount'].sum()
            values = daily_spending.values
            
            # Use ensemble of models
            models = [
                LinearRegression(),
                MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42, early_stopping=True),
                SVR(kernel='rbf')
            ]
            
            predictions_list = []
            
            for model in models:
                try:
                    model.fit(X, y)
                    last_sequence = values[-7:]
                    model_predictions = []
                    
                    for _ in range(days):
                        pred = model.predict(last_sequence.reshape(1, -1))[0]
                        model_predictions.append(max(0, pred))  # Ensure non-negative
                        last_sequence = np.roll(last_sequence, -1)
                        last_sequence[-1] = pred
                    
                    predictions_list.append(model_predictions)
                except Exception as e:
                    print(f"Model {type(model).__name__} failed: {e}")
                    continue
            
            if not predictions_list:
                return self._simple_forecast(expenses_df, days)
            
            # Ensemble average
            predictions = np.mean(predictions_list, axis=0)
            
            # Calculate confidence based on model agreement
            if len(predictions_list) > 1:
                std_predictions = np.std(predictions_list, axis=0)
                avg_std = np.mean(std_predictions)
                confidence = max(0.5, 1 - (avg_std / np.mean(predictions)))
            else:
                confidence = 0.7
            
            trend = 'increasing' if predictions[-1] > predictions[0] else 'decreasing'
            if abs(predictions[-1] - predictions[0]) < np.mean(predictions) * 0.05:
                trend = 'stable'
            
            return {
                'predictions': predictions.tolist(),
                'next_week_total': sum(predictions),
                'confidence': round(confidence, 2),
                'trend': trend,
                'method_used': 'ensemble'
            }
            
        except Exception as e:
            print(f"ML forecast failed: {e}")
            return self._simple_forecast(expenses_df, days)
