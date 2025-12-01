from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import os

from models.expense_model import Expense, init_db
from models.ml_models import ExpensePredictor, AnomalyDetector, ForecastModel
from utils.data_processor import DataProcessor
from utils.visualization import create_spending_chart, create_trend_analysis, create_analytics_dashboard, create_text_summary

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['DATABASE'] = 'expenses.db'

# Initialize components
init_db()
ml_predictor = ExpensePredictor()
anomaly_detector = AnomalyDetector()
forecast_model = ForecastModel()
data_processor = DataProcessor()

@app.route('/')
def index():
    """Main dashboard with overview and analytics"""
    try:
        expenses = Expense.get_all()
        df = pd.DataFrame([exp.to_dict() for exp in expenses])
        
        # Basic statistics
        total_spent = df['amount'].sum() if not df.empty else 0
        avg_daily = df.groupby('date')['amount'].sum().mean() if not df.empty else 0
        category_summary = df.groupby('category')['amount'].sum().to_dict() if not df.empty else {}
        
        # ML Insights
        if len(df) >= 10:  # Only generate insights with sufficient data
            insights = ml_predictor.generate_insights(df)
            chart_html = create_spending_chart(df)
        else:
            insights = {"message": "Add more expenses to see insights"}
            chart_html = "<p>Add more data to see charts</p>"
            
        return render_template('index.html', 
                             total_spent=total_spent,
                             avg_daily=avg_daily,
                             category_summary=category_summary,
                             insights=insights,
                             chart_html=chart_html)
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/expenses')
def expenses():
    """View all expenses with filtering and pagination"""
    page = request.args.get('page', 1, type=int)
    category = request.args.get('category', '')
    
    expenses_data = Expense.get_filtered(category=category if category else None)
    return render_template('expenses.html', expenses=expenses_data, category=category)

@app.route('/add', methods=['GET', 'POST'])
def add_expense():
    """Add new expense with AI-powered category suggestion"""
    if request.method == 'POST':
        try:
            amount = float(request.form['amount'])
            description = request.form['description']
            category = request.form['category']
            date = request.form['date'] or datetime.now().strftime('%Y-%m-%d')
            
            # AI-powered category suggestion if not provided
            if not category:
                category = ml_predictor.predict_category(description, amount)
            
            expense = Expense(amount=amount, description=description, 
                            category=category, date=date)
            expense.save()
            
            # Retrain model with new data periodically
            if Expense.count() % 10 == 0:
                ml_predictor.retrain_model()
                
            return redirect(url_for('expenses'))
        except Exception as e:
            return render_template('add_expense.html', error=str(e))
    
    return render_template('add_expense.html')

@app.route('/edit/<int:expense_id>', methods=['GET', 'POST'])
def edit_expense(expense_id):
    """Edit existing expense"""
    expense = Expense.get_by_id(expense_id)
    if not expense:
        return redirect(url_for('expenses'))
    
    if request.method == 'POST':
        try:
            expense.amount = float(request.form['amount'])
            expense.description = request.form['description']
            expense.category = request.form['category']
            expense.date = request.form['date']
            expense.update()
            return redirect(url_for('expenses'))
        except Exception as e:
            return render_template('edit_expense.html', expense=expense, error=str(e))
    
    return render_template('edit_expense.html', expense=expense)

@app.route('/delete/<int:expense_id>')
def delete_expense(expense_id):
    """Delete expense"""
    expense = Expense.get_by_id(expense_id)
    if expense:
        expense.delete()
    return redirect(url_for('expenses'))

@app.route('/analytics')
def analytics():
    """Advanced analytics with ML insights"""
    try:
        expenses = Expense.get_all()
        if len(expenses) < 5:
            return render_template('analytics.html', 
                                 error="Need at least 5 expenses for analytics")
        
        df = pd.DataFrame([exp.to_dict() for exp in expenses])
        
        # ML-powered analytics
        anomalies = anomaly_detector.detect_anomalies(df)
        forecast = forecast_model.predict_spending(df)
        patterns = ml_predictor.analyze_patterns(df)
        
        # Visualizations
        trend_chart = create_trend_analysis(df)
        anomaly_chart = create_spending_chart(df, highlight_anomalies=True)
        
        return render_template('analytics.html',
                             anomalies=anomalies,
                             forecast=forecast,
                             patterns=patterns,
                             trend_chart=trend_chart,
                             anomaly_chart=anomaly_chart)
    except Exception as e:
        return render_template('analytics.html', error=str(e))

@app.route('/api/expenses', methods=['GET'])
def api_get_expenses():
    """REST API to get expenses"""
    expenses = Expense.get_all()
    return jsonify([exp.to_dict() for exp in expenses])

@app.route('/api/predict-category', methods=['POST'])
def api_predict_category():
    """API for category prediction"""
    data = request.get_json()
    description = data.get('description', '')
    amount = data.get('amount', 0)
    
    category = ml_predictor.predict_category(description, amount)
    return jsonify({'category': category})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
