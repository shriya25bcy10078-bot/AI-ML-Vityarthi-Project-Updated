import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self):
        self.supported_categories = ['Food', 'Transportation', 'Entertainment', 
                                   'Shopping', 'Bills', 'Healthcare', 'Other']
    
    def clean_expense_data(self, expenses_df):
        """Clean and preprocess expense data"""
        try:
            df = expenses_df.copy()
            
            # Handle missing values
            df['description'] = df['description'].fillna('')
            df['category'] = df['category'].fillna('Other')
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df = df.dropna(subset=['amount'])
            
            # Validate categories
            df['category'] = df['category'].apply(
                lambda x: x if x in self.supported_categories else 'Other'
            )
            
            # Ensure date format
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            return df
            
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return expenses_df
    
    def extract_features(self, expenses_df):
        """Extract additional features from expense data"""
        try:
            df = expenses_df.copy()
            
            # Time-based features
            df['datetime'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['datetime'].dt.day_name()
            df['month'] = df['datetime'].dt.month_name()
            df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6])
            
            # Amount-based features
            df['amount_category'] = pd.cut(df['amount'], 
                                         bins=[0, 10, 50, 100, float('inf')],
                                         labels=['Small', 'Medium', 'Large', 'Very Large'])
            
            # Text-based features
            df['description_length'] = df['description'].str.len()
            df['has_digits'] = df['description'].str.contains(r'\d')
            
            return df
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return expenses_df
    
    def calculate_metrics(self, expenses_df):
        """Calculate various spending metrics"""
        try:
            metrics = {}
            
            # Basic metrics
            metrics['total_expenses'] = len(expenses_df)
            metrics['total_amount'] = expenses_df['amount'].sum()
            metrics['average_expense'] = expenses_df['amount'].mean()
            metrics['median_expense'] = expenses_df['amount'].median()
            
            # Time-based metrics
            daily_spending = expenses_df.groupby('date')['amount'].sum()
            metrics['avg_daily_spending'] = daily_spending.mean()
            metrics['max_daily_spending'] = daily_spending.max()
            
            # Category metrics
            category_stats = expenses_df.groupby('category').agg({
                'amount': ['sum', 'mean', 'count']
            }).round(2)
            metrics['category_breakdown'] = category_stats.to_dict()
            
            # Trend metrics
            if len(daily_spending) > 7:
                recent_week = daily_spending.tail(7).mean()
                previous_week = daily_spending.head(len(daily_spending)-7).mean()
                metrics['weekly_trend'] = (recent_week - previous_week) / previous_week
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {}
    
    def detect_seasonality(self, expenses_df):
        """Detect seasonal patterns in spending"""
        try:
            if len(expenses_df) < 30:
                return {"message": "Need at least 30 days of data for seasonality analysis"}
            
            df = expenses_df.copy()
            df['datetime'] = pd.to_datetime(df['date'])
            daily_spending = df.groupby('datetime')['amount'].sum()
            
            # Weekly seasonality
            weekly_pattern = daily_spending.groupby(daily_spending.index.dayofweek).mean()
            
            # Monthly seasonality (if enough data)
            monthly_pattern = None
            if len(daily_spending) > 90:
                monthly_pattern = daily_spending.groupby(daily_spending.index.month).mean()
            
            return {
                'weekly_seasonality': weekly_pattern.to_dict(),
                'monthly_seasonality': monthly_pattern.to_dict() if monthly_pattern is not None else None,
                'peak_day': weekly_pattern.idxmax(),
                'lowest_day': weekly_pattern.idxmin()
            }
            
        except Exception as e:
            return {'error': f"Seasonality detection failed: {e}"}
