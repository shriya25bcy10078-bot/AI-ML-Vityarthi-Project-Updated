import sqlite3
from datetime import datetime
import os

class Expense:
    def __init__(self, id=None, amount=0.0, description="", category="Other", date=None):
        self.id = id
        self.amount = amount
        self.description = description
        self.category = category
        self.date = date or datetime.now().strftime('%Y-%m-%d')
    
    def save(self):
        """Save expense to database"""
        conn = sqlite3.connect('expenses.db')
        cursor = conn.cursor()
        
        if self.id is None:
            cursor.execute('''
                INSERT INTO expenses (amount, description, category, date)
                VALUES (?, ?, ?, ?)
            ''', (self.amount, self.description, self.category, self.date))
            self.id = cursor.lastrowid
        else:
            cursor.execute('''
                UPDATE expenses 
                SET amount=?, description=?, category=?, date=?
                WHERE id=?
            ''', (self.amount, self.description, self.category, self.date, self.id))
        
        conn.commit()
        conn.close()
    
    def delete(self):
        """Delete expense from database"""
        if self.id is not None:
            conn = sqlite3.connect('expenses.db')
            cursor = conn.cursor()
            cursor.execute('DELETE FROM expenses WHERE id=?', (self.id,))
            conn.commit()
            conn.close()
    
    def update(self):
        """Update existing expense"""
        self.save()
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'amount': self.amount,
            'description': self.description,
            'category': self.category,
            'date': self.date
        }
    
    @classmethod
    def get_by_id(cls, expense_id):
        """Get expense by ID"""
        conn = sqlite3.connect('expenses.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM expenses WHERE id=?', (expense_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return cls(id=row[0], amount=row[1], description=row[2], category=row[3], date=row[4])
        return None
    
    @classmethod
    def get_all(cls):
        """Get all expenses"""
        conn = sqlite3.connect('expenses.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM expenses ORDER BY date DESC')
        rows = cursor.fetchall()
        conn.close()
        
        expenses = []
        for row in rows:
            expenses.append(cls(id=row[0], amount=row[1], description=row[2], category=row[3], date=row[4]))
        return expenses
    
    @classmethod
    def get_filtered(cls, category=None, start_date=None, end_date=None):
        """Get filtered expenses"""
        conn = sqlite3.connect('expenses.db')
        cursor = conn.cursor()
        
        query = 'SELECT * FROM expenses WHERE 1=1'
        params = []
        
        if category:
            query += ' AND category = ?'
            params.append(category)
        
        if start_date:
            query += ' AND date >= ?'
            params.append(start_date)
            
        if end_date:
            query += ' AND date <= ?'
            params.append(end_date)
            
        query += ' ORDER BY date DESC'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        expenses = []
        for row in rows:
            expenses.append(cls(id=row[0], amount=row[1], description=row[2], category=row[3], date=row[4]))
        return expenses
    
    @classmethod
    def count(cls):
        """Get total number of expenses"""
        conn = sqlite3.connect('expenses.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM expenses')
        count = cursor.fetchone()[0]
        conn.close()
        return count

def init_db():
    """Initialize database with required tables"""
    conn = sqlite3.connect('expenses.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL NOT NULL,
            description TEXT,
            category TEXT,
            date TEXT
        )
    ''')
    
    # Insert sample data if empty
    cursor.execute('SELECT COUNT(*) FROM expenses')
    if cursor.fetchone()[0] == 0:
        sample_expenses = [
            (25.50, "Lunch at restaurant", "Food", "2024-01-15"),
            (45.00, "Gasoline", "Transportation", "2024-01-14"),
            (12.99, "Netflix subscription", "Entertainment", "2024-01-13"),
            (89.99, "Groceries", "Food", "2024-01-12"),
            (15.00, "Uber ride", "Transportation", "2024-01-11")
        ]
        cursor.executemany('''
            INSERT INTO expenses (amount, description, category, date)
            VALUES (?, ?, ?, ?)
        ''', sample_expenses)
    
    conn.commit()
    conn.close()
