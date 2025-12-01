import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def create_spending_chart(expenses_df, highlight_anomalies=False):
    """Create interactive spending chart using Plotly"""
    try:
        if expenses_df.empty:
            return "<p>No data available for chart</p>"
        
        category_totals = expenses_df.groupby('category')['amount'].sum().reset_index()
        
        fig = px.bar(category_totals, 
                    x='category', 
                    y='amount',
                    title='Spending by Category',
                    labels={'amount': 'Total Amount ($)', 'category': 'Category'})
        
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            height=400
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        return f"<p>Error creating chart: {str(e)}</p>"

def create_trend_analysis(expenses_df):
    """Create trend analysis using Plotly"""
    try:
        if expenses_df.empty:
            return "<p>No data available for trend analysis</p>"
        
        daily_spending = expenses_df.groupby('date')['amount'].sum().reset_index()
        daily_spending['date'] = pd.to_datetime(daily_spending['date'])
        daily_spending = daily_spending.sort_values('date')
        
        fig = px.line(daily_spending, 
                     x='date', 
                     y='amount',
                     title='Daily Spending Trend',
                     labels={'amount': 'Amount ($)', 'date': 'Date'})
        
        fig.update_layout(height=400)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        return f"<p>Error creating trend chart: {str(e)}</p>"

def create_analytics_dashboard(expenses_df):
    """Create comprehensive analytics dashboard"""
    try:
        if len(expenses_df) < 5:
            return "<p>Need more data for analytics dashboard</p>"
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Spending by Category', 'Daily Trend', 
                          'Expense Distribution', 'Category Percentage'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "pie"}]]
        )
        
        # Chart 1: Category spending
        category_totals = expenses_df.groupby('category')['amount'].sum()
        fig.add_trace(go.Bar(x=category_totals.index, y=category_totals.values), row=1, col=1)
        
        # Chart 2: Daily trend
        daily_spending = expenses_df.groupby('date')['amount'].sum()
        fig.add_trace(go.Scatter(x=daily_spending.index, y=daily_spending.values, mode='lines+markers'), row=1, col=2)
        
        # Chart 3: Distribution
        fig.add_trace(go.Histogram(x=expenses_df['amount'], nbinsx=20), row=2, col=1)
        
        # Chart 4: Pie chart
        fig.add_trace(go.Pie(labels=category_totals.index, values=category_totals.values), row=2, col=2)
        
        fig.update_layout(height=800, title_text="Expense Analytics Dashboard")
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
        
    except Exception as e:
        return f"<p>Error creating dashboard: {str(e)}</p>"

def create_text_summary(expenses_df):
    """Create text-based summary as fallback"""
    try:
        summary = []
        summary.append("<div class='text-summary'>")
        summary.append("<h4>Expense Summary</h4>")
        
        total_spent = expenses_df['amount'].sum()
        avg_expense = expenses_df['amount'].mean()
        total_expenses = len(expenses_df)
        
        summary.append(f"<p>Total Spent: ${total_spent:.2f}</p>")
        summary.append(f"<p>Average Expense: ${avg_expense:.2f}</p>")
        summary.append(f"<p>Total Expenses: {total_expenses}</p>")
        
        # Category breakdown
        summary.append("<h5>Category Breakdown:</h5>")
        category_totals = expenses_df.groupby('category')['amount'].sum()
        for category, amount in category_totals.items():
            percentage = (amount / total_spent) * 100 if total_spent > 0 else 0
            summary.append(f"<p>{category}: ${amount:.2f} ({percentage:.1f}%)</p>")
        
        summary.append("</div>")
        return "\n".join(summary)
        
    except Exception as e:
        return f"<p>Error creating summary: {str(e)}</p>"
