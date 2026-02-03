"""
Shoe Database Management with SQLite

This module provides database operations for the shoe department assistant.
"""

import sqlite3
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class ShoeDatabase:
    """SQLite database manager for shoe inventory"""
    
    def __init__(self, db_path: str = "shoe_inventory.db"):
        """Initialize database connection and create tables if needed"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()
        self.populate_sample_data()
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Shoes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand TEXT NOT NULL,
                model TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                sizes TEXT NOT NULL,
                colors TEXT NOT NULL,
                description TEXT,
                features TEXT,
                in_stock INTEGER DEFAULT 1
            )
        """)
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_visit TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # User preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                preference_type TEXT NOT NULL,
                preference_value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        self.conn.commit()
    
    def populate_sample_data(self):
        """Populate database with sample shoe data if empty"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM shoes")
        count = cursor.fetchone()[0]
        
        if count == 0:
            sample_shoes = [
                {
                    "brand": "Nike",
                    "model": "Air Max 270",
                    "category": "Running",
                    "price": 150.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]),
                    "colors": json.dumps(["Black", "White", "Blue", "Red"]),
                    "description": "Comfortable running shoes with excellent cushioning",
                    "features": json.dumps(["Air cushioning", "Breathable mesh", "Durable rubber outsole"]),
                    "in_stock": 1
                },
                {
                    "brand": "Adidas",
                    "model": "Ultraboost 22",
                    "category": "Running",
                    "price": 180.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]),
                    "colors": json.dumps(["Black", "White", "Grey"]),
                    "description": "Premium running shoes with responsive Boost cushioning",
                    "features": json.dumps(["Boost technology", "Primeknit upper", "Continental rubber outsole"]),
                    "in_stock": 1
                },
                {
                    "brand": "New Balance",
                    "model": "Fresh Foam 1080v12",
                    "category": "Running",
                    "price": 160.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black", "Blue", "Grey"]),
                    "description": "Plush cushioning for long-distance running",
                    "features": json.dumps(["Fresh Foam cushioning", "Engineered mesh", "Wide sizes available"]),
                    "in_stock": 1
                },
                {
                    "brand": "Converse",
                    "model": "Chuck Taylor All Star",
                    "category": "Casual",
                    "price": 60.00,
                    "sizes": json.dumps([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]),
                    "colors": json.dumps(["Black", "White", "Red", "Navy", "Pink"]),
                    "description": "Classic canvas sneakers for everyday wear",
                    "features": json.dumps(["Canvas upper", "Rubber sole", "Iconic design"]),
                    "in_stock": 1
                },
                {
                    "brand": "Vans",
                    "model": "Old Skool",
                    "category": "Casual",
                    "price": 70.00,
                    "sizes": json.dumps([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]),
                    "colors": json.dumps(["Black/White", "Navy", "Red", "Checkerboard"]),
                    "description": "Skate-inspired casual shoes with signature side stripe",
                    "features": json.dumps(["Suede and canvas upper", "Padded collar", "Waffle outsole"]),
                    "in_stock": 1
                },
                {
                    "brand": "Clarks",
                    "model": "Desert Boot",
                    "category": "Casual",
                    "price": 140.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]),
                    "colors": json.dumps(["Beeswax", "Sand", "Black"]),
                    "description": "Classic desert boots with premium suede",
                    "features": json.dumps(["Suede leather", "Crepe sole", "Ankle height"]),
                    "in_stock": 1
                },
                {
                    "brand": "Asics",
                    "model": "Gel-Kayano 29",
                    "category": "Running",
                    "price": 160.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]),
                    "colors": json.dumps(["Black", "Blue", "White"]),
                    "description": "Stability running shoes for overpronators",
                    "features": json.dumps(["GEL cushioning", "Dynamic DuoMax support", "Breathable mesh"]),
                    "in_stock": 1
                },
                {
                    "brand": "Skechers",
                    "model": "Go Walk 6",
                    "category": "Walking",
                    "price": 85.00,
                    "sizes": json.dumps([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black", "Grey", "Navy", "White"]),
                    "description": "Lightweight walking shoes with excellent comfort",
                    "features": json.dumps(["Ultra Go cushioning", "Air-cooled memory foam", "Slip-on design"]),
                    "in_stock": 1
                },
                {
                    "brand": "Timberland",
                    "model": "6-Inch Premium Boot",
                    "category": "Boots",
                    "price": 200.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Wheat", "Black", "Brown"]),
                    "description": "Waterproof leather boots for all weather",
                    "features": json.dumps(["Waterproof leather", "Padded collar", "Rubber lug outsole"]),
                    "in_stock": 1
                },
                {
                    "brand": "Dr. Martens",
                    "model": "1460 8-Eye Boot",
                    "category": "Boots",
                    "price": 170.00,
                    "sizes": json.dumps([6, 7, 8, 9, 10, 11, 12]),
                    "colors": json.dumps(["Black", "Cherry Red", "White"]),
                    "description": "Iconic leather boots with air-cushioned sole",
                    "features": json.dumps(["Smooth leather", "AirWair sole", "Yellow stitching"]),
                    "in_stock": 1
                },
                {
                    "brand": "Allbirds",
                    "model": "Tree Runners",
                    "category": "Casual",
                    "price": 98.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]),
                    "colors": json.dumps(["Natural Grey", "Blizzard", "Thunder"]),
                    "description": "Sustainable shoes made from eucalyptus tree fiber",
                    "features": json.dumps(["Eucalyptus fiber upper", "Merino wool insole", "Carbon neutral"]),
                    "in_stock": 1
                },
                {
                    "brand": "Hoka",
                    "model": "Clifton 9",
                    "category": "Running",
                    "price": 145.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black/White", "Blue", "Grey"]),
                    "description": "Maximalist cushioning for comfortable long runs",
                    "features": json.dumps(["Maximum cushioning", "Lightweight", "Early stage Meta-Rocker"]),
                    "in_stock": 1
                }
            ]
            
            cursor.executemany("""
                INSERT INTO shoes (brand, model, category, price, sizes, colors, description, features, in_stock)
                VALUES (:brand, :model, :category, :price, :sizes, :colors, :description, :features, :in_stock)
            """, sample_shoes)
            
            self.conn.commit()
    
    def search_shoes(self, 
                     brand: Optional[str] = None,
                     category: Optional[str] = None,
                     max_price: Optional[float] = None,
                     min_price: Optional[float] = None) -> List[Dict]:
        """Search for shoes based on criteria"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM shoes WHERE in_stock = 1"
        params = []
        
        if brand:
            query += " AND LOWER(brand) LIKE LOWER(?)"
            params.append(f"%{brand}%")
        
        if category:
            query += " AND LOWER(category) LIKE LOWER(?)"
            params.append(f"%{category}%")
        
        if max_price:
            query += " AND price <= ?"
            params.append(max_price)
        
        if min_price:
            query += " AND price >= ?"
            params.append(min_price)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            shoe = dict(row)
            shoe['sizes'] = json.loads(shoe['sizes'])
            shoe['colors'] = json.loads(shoe['colors'])
            shoe['features'] = json.loads(shoe['features'])
            results.append(shoe)
        
        return results
    
    def get_shoe_by_id(self, shoe_id: int) -> Optional[Dict]:
        """Get a specific shoe by ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM shoes WHERE id = ?", (shoe_id,))
        row = cursor.fetchone()
        
        if row:
            shoe = dict(row)
            shoe['sizes'] = json.loads(shoe['sizes'])
            shoe['colors'] = json.loads(shoe['colors'])
            shoe['features'] = json.loads(shoe['features'])
            return shoe
        return None
    
    def get_all_brands(self) -> List[str]:
        """Get list of all available brands"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT brand FROM shoes WHERE in_stock = 1 ORDER BY brand")
        return [row[0] for row in cursor.fetchall()]
    
    def get_all_categories(self) -> List[str]:
        """Get list of all available categories"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT category FROM shoes WHERE in_stock = 1 ORDER BY category")
        return [row[0] for row in cursor.fetchall()]
    
    def find_user_by_name(self, first_name: str, last_name: str) -> Optional[Dict]:
        """Find a user by first and last name"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, first_name, last_name, created_at, last_visit 
            FROM users 
            WHERE LOWER(first_name) = LOWER(?) AND LOWER(last_name) = LOWER(?)
        """, (first_name, last_name))
        row = cursor.fetchone()
        
        if row:
            return {
                'id': row[0],
                'first_name': row[1],
                'last_name': row[2],
                'created_at': row[3],
                'last_visit': row[4]
            }
        return None
    
    def create_user(self, first_name: str, last_name: str) -> int:
        """Create a new user and return their ID"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO users (first_name, last_name)
            VALUES (?, ?)
        """, (first_name, last_name))
        self.conn.commit()
        return cursor.lastrowid
    
    def update_user_last_visit(self, user_id: int) -> None:
        """Update the last visit timestamp for a user"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE users 
            SET last_visit = CURRENT_TIMESTAMP 
            WHERE id = ?
        """, (user_id,))
        self.conn.commit()
    
    def add_user_preference(self, user_id: int, preference_type: str, preference_value: str) -> None:
        """Add a preference for a user"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO user_preferences (user_id, preference_type, preference_value)
            VALUES (?, ?, ?)
        """, (user_id, preference_type, preference_value))
        self.conn.commit()
    
    def get_user_preferences(self, user_id: int) -> List[Dict]:
        """Get all preferences for a user"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT preference_type, preference_value, created_at
            FROM user_preferences
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, (user_id,))
        
        preferences = []
        for row in cursor.fetchall():
            preferences.append({
                'type': row[0],
                'value': row[1],
                'created_at': row[2]
            })
        return preferences
    
    def get_user_preferences_summary(self, user_id: int) -> Dict:
        """Get a summary of user preferences grouped by type"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT preference_type, GROUP_CONCAT(preference_value, ', ') as values
            FROM user_preferences
            WHERE user_id = ?
            GROUP BY preference_type
        """, (user_id,))
        
        summary = {}
        for row in cursor.fetchall():
            summary[row[0]] = row[1]
        return summary
    
    def close(self):
        """Close database connection"""
        self.conn.close()
