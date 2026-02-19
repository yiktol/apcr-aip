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
    
    def __init__(self, db_path: str = None):
        """Initialize database connection and create tables if needed"""
        # If no path provided, use default path in session3 directory
        if db_path is None:
            # Get the directory where this file is located (utils/)
            current_dir = Path(__file__).parent
            # Go up one level to session3/ and set the db path
            db_path = str(current_dir.parent / "shoe_inventory.db")
        
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
                # Running Shoes
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
                    "brand": "Nike",
                    "model": "Pegasus 40",
                    "category": "Running",
                    "price": 140.00,
                    "sizes": json.dumps([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black/White", "Blue", "Pink", "Volt"]),
                    "description": "Versatile daily running shoes with ReactX foam",
                    "features": json.dumps(["ReactX foam", "Waffle outsole", "Engineered mesh upper", "13% more energy return"]),
                    "in_stock": 1
                },
                {
                    "brand": "Nike",
                    "model": "Vaporfly 3",
                    "category": "Running",
                    "price": 260.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]),
                    "colors": json.dumps(["Volt/Black", "White/Blue", "Pink"]),
                    "description": "Elite racing shoes with carbon fiber plate",
                    "features": json.dumps(["ZoomX foam", "Carbon fiber plate", "Flyknit upper", "Race day performance"]),
                    "in_stock": 1
                },
                {
                    "brand": "Adidas",
                    "model": "Ultraboost 23",
                    "category": "Running",
                    "price": 190.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]),
                    "colors": json.dumps(["Black", "White", "Grey", "Solar Red"]),
                    "description": "Premium running shoes with responsive Boost cushioning",
                    "features": json.dumps(["Boost technology", "Primeknit+ upper", "Continental rubber outsole", "Linear Energy Push"]),
                    "in_stock": 1
                },
                {
                    "brand": "Adidas",
                    "model": "Adizero Boston 12",
                    "category": "Running",
                    "price": 160.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]),
                    "colors": json.dumps(["Black/White", "Solar Yellow", "Blue"]),
                    "description": "Lightweight tempo running shoes for speed training",
                    "features": json.dumps(["Lightstrike Pro cushioning", "Energy rods", "Continental rubber", "Race-ready design"]),
                    "in_stock": 1
                },
                {
                    "brand": "New Balance",
                    "model": "Fresh Foam 1080v13",
                    "category": "Running",
                    "price": 165.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13, 14]),
                    "colors": json.dumps(["Black", "Blue", "Grey", "White"]),
                    "description": "Plush cushioning for long-distance running",
                    "features": json.dumps(["Fresh Foam X cushioning", "Hypoknit upper", "Wide sizes available", "Rocker geometry"]),
                    "in_stock": 1
                },
                {
                    "brand": "Asics",
                    "model": "Gel-Kayano 30",
                    "category": "Running",
                    "price": 170.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]),
                    "colors": json.dumps(["Black", "Blue", "White", "Grey"]),
                    "description": "Stability running shoes for overpronators",
                    "features": json.dumps(["FF Blast Plus cushioning", "4D Guidance System", "PureGEL technology", "Breathable mesh"]),
                    "in_stock": 1
                },
                {
                    "brand": "Hoka",
                    "model": "Clifton 9",
                    "category": "Running",
                    "price": 145.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black/White", "Blue", "Grey", "Coral"]),
                    "description": "Maximalist cushioning for comfortable long runs",
                    "features": json.dumps(["Maximum cushioning", "Lightweight", "Early stage Meta-Rocker", "Compression molded EVA"]),
                    "in_stock": 1
                },
                {
                    "brand": "Hoka",
                    "model": "Speedgoat 5",
                    "category": "Trail Running",
                    "price": 155.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black/Orange", "Blue", "Green"]),
                    "description": "Technical trail running shoes for rugged terrain",
                    "features": json.dumps(["Vibram Megagrip outsole", "Cushioned midsole", "Protective toe cap", "Trail-specific traction"]),
                    "in_stock": 1
                },
                {
                    "brand": "Saucony",
                    "model": "Endorphin Speed 3",
                    "category": "Running",
                    "price": 170.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]),
                    "colors": json.dumps(["White/Black", "Vizipro", "Blue"]),
                    "description": "Tempo running shoes with nylon plate",
                    "features": json.dumps(["PWRRUN PB cushioning", "Nylon plate", "Speedroll technology", "Lightweight design"]),
                    "in_stock": 1
                },
                {
                    "brand": "Brooks",
                    "model": "Ghost 17",
                    "category": "Running",
                    "price": 140.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13, 14]),
                    "colors": json.dumps(["Black", "White", "Blue", "Grey", "Pink"]),
                    "description": "Neutral running shoes for everyday training with improved cushioning",
                    "features": json.dumps(["DNA Loft v3 cushioning", "Segmented Crash Pad", "3D Fit Print upper", "Smooth transitions", "67% recycled upper"]),
                    "in_stock": 1
                },
                {
                    "brand": "Asics",
                    "model": "Gel-Nimbus 28",
                    "category": "Running",
                    "price": 170.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black", "White", "Blue", "Grey", "Pink"]),
                    "description": "Maximum cushioned running shoes with lighter weight design",
                    "features": json.dumps(["FF Blast Plus Eco cushioning", "PureGEL technology", "43.5mm heel stack", "30g lighter than v27", "20% bio-based foam"]),
                    "in_stock": 1
                },
                {
                    "brand": "Hoka",
                    "model": "Clifton 10",
                    "category": "Running",
                    "price": 145.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black/White", "Blue", "Grey", "Coral", "Green"]),
                    "description": "Maximalist cushioning with improved responsiveness",
                    "features": json.dumps(["Compression molded EVA", "Early stage Meta-Rocker", "Lightweight design", "Plush cushioning platform"]),
                    "in_stock": 1
                },
                {
                    "brand": "Mizuno",
                    "model": "Wave Rider 29",
                    "category": "Running",
                    "price": 150.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black", "White", "Blue", "Grey"]),
                    "description": "Versatile daily trainer with nitrogen-infused cushioning",
                    "features": json.dumps(["Mizuno Enerzy NXT", "Mizuno Wave plate", "Smooth weight transfer", "Moderate cushioning"]),
                    "in_stock": 1
                },
                {
                    "brand": "Karhu",
                    "model": "Ikoni 3.0",
                    "category": "Running",
                    "price": 160.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black/White", "Blue", "Red", "Grey"]),
                    "description": "Finnish-engineered daily trainer with balanced cushioning",
                    "features": json.dumps(["AHAR Plus cushioning", "Fulcrum technology", "M-Lock midfoot support", "Breathable mesh upper"]),
                    "in_stock": 1
                },
                {
                    "brand": "Altra",
                    "model": "Lone Peak 9",
                    "category": "Trail Running",
                    "price": 145.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13, 14]),
                    "colors": json.dumps(["Black", "Dusty Olive", "Teal/Black", "Grey"]),
                    "description": "Zero-drop trail running shoes with roomy toe box",
                    "features": json.dumps(["Zero drop platform", "Roomy toe box", "StoneGuard rockplate", "Ripstop mesh upper", "MaxTrac outsole"]),
                    "in_stock": 1
                },
                {
                    "brand": "Altra",
                    "model": "Lone Peak 9+ Vibram",
                    "category": "Trail Running",
                    "price": 155.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13, 14]),
                    "colors": json.dumps(["Black", "Teal/Black", "Grey"]),
                    "description": "Enhanced trail runner with Vibram Megagrip outsole",
                    "features": json.dumps(["Vibram Megagrip outsole", "Zero drop", "Roomy toe box", "StoneGuard rockplate", "Enhanced traction"]),
                    "in_stock": 1
                },
                {
                    "brand": "On",
                    "model": "Cloudmonster 2",
                    "category": "Running",
                    "price": 180.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Nimbus/Arctic", "Black/White", "Blue", "Grey"]),
                    "description": "Maximum cushioned running shoes with CloudTec technology",
                    "features": json.dumps(["CloudTec cushioning", "Helion superfoam", "Speedboard technology", "High-stack platform"]),
                    "in_stock": 1
                },
                {
                    "brand": "Saucony",
                    "model": "Ride 18",
                    "category": "Running",
                    "price": 140.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black/White", "Blue", "Grey", "Vizipro"]),
                    "description": "Versatile daily trainer with balanced cushioning",
                    "features": json.dumps(["PWRRUN cushioning", "PWRRUN+ sockliner", "Durable XT-900 outsole", "Smooth ride"]),
                    "in_stock": 1
                },
                {
                    "brand": "Adidas",
                    "model": "Adizero Evo SL",
                    "category": "Running",
                    "price": 130.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black/White", "Solar Yellow", "Blue", "Pink"]),
                    "description": "Lightweight everyday running shoes with race-day feel",
                    "features": json.dumps(["Lightstrike cushioning", "Continental rubber outsole", "Lightweight design", "Breathable upper"]),
                    "in_stock": 1
                },
                {
                    "brand": "Asics",
                    "model": "Metaspeed Sky Tokyo",
                    "category": "Running",
                    "price": 275.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]),
                    "colors": json.dumps(["White/Red", "Black/Gold", "Blue"]),
                    "description": "Elite carbon-plated racing shoes for all distances",
                    "features": json.dumps(["FF Turbo Plus cushioning", "Carbon plate", "Motion Wrap upper", "Race-day performance"]),
                    "in_stock": 1
                },
                
                # Basketball Shoes
                {
                    "brand": "Nike",
                    "model": "LeBron 21",
                    "category": "Basketball",
                    "price": 200.00,
                    "sizes": json.dumps([8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 14, 15]),
                    "colors": json.dumps(["Black/Gold", "White/Red", "Purple/Yellow"]),
                    "description": "Signature basketball shoes with Zoom Air cushioning",
                    "features": json.dumps(["Zoom Air units", "Battleknit 2.0 upper", "Carbon fiber shank", "Multi-directional traction"]),
                    "in_stock": 1
                },
                {
                    "brand": "Nike",
                    "model": "KD 16",
                    "category": "Basketball",
                    "price": 160.00,
                    "sizes": json.dumps([8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 14]),
                    "colors": json.dumps(["Black/White", "Blue", "Aunt Pearl"]),
                    "description": "Lightweight basketball shoes for guards and forwards",
                    "features": json.dumps(["Cushlon 3.0 foam", "Zoom Air Strobel", "Lightweight design", "Herringbone traction"]),
                    "in_stock": 1
                },
                {
                    "brand": "Adidas",
                    "model": "Dame 8",
                    "category": "Basketball",
                    "price": 120.00,
                    "sizes": json.dumps([8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 14]),
                    "colors": json.dumps(["Black/Red", "White/Blue", "Oakland"]),
                    "description": "Damian Lillard signature shoes with Bounce Pro",
                    "features": json.dumps(["Bounce Pro cushioning", "Textile upper", "Rubber outsole", "Lockdown fit"]),
                    "in_stock": 1
                },
                {
                    "brand": "Under Armour",
                    "model": "Curry 11",
                    "category": "Basketball",
                    "price": 160.00,
                    "sizes": json.dumps([8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 14]),
                    "colors": json.dumps(["White/Gold", "Black/Blue", "Championship"]),
                    "description": "Stephen Curry signature shoes for shooters",
                    "features": json.dumps(["Flow cushioning", "UA Warp upper", "Traction pattern", "Lightweight construction"]),
                    "in_stock": 1
                },
                {
                    "brand": "Jordan",
                    "model": "Luka 2",
                    "category": "Basketball",
                    "price": 130.00,
                    "sizes": json.dumps([8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 14]),
                    "colors": json.dumps(["White/Blue", "Black/Green", "Bred"]),
                    "description": "Luka Doncic signature shoes with Formula 23 foam",
                    "features": json.dumps(["Formula 23 foam", "IsoPlate technology", "Data-informed traction", "Supportive fit"]),
                    "in_stock": 1
                },
                
                # Soccer/Football Shoes
                {
                    "brand": "Nike",
                    "model": "Mercurial Vapor 15",
                    "category": "Soccer",
                    "price": 270.00,
                    "sizes": json.dumps([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]),
                    "colors": json.dumps(["Volt/Black", "White/Blue", "Pink"]),
                    "description": "Speed-focused soccer cleats for explosive acceleration",
                    "features": json.dumps(["Vaporposite+ upper", "Air Zoom unit", "Tri-star studs", "Speed cage"]),
                    "in_stock": 1
                },
                {
                    "brand": "Adidas",
                    "model": "Predator Edge",
                    "category": "Soccer",
                    "price": 250.00,
                    "sizes": json.dumps([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]),
                    "colors": json.dumps(["Black/Red", "White/Blue", "Solar Yellow"]),
                    "description": "Control-focused soccer cleats with rubber elements",
                    "features": json.dumps(["Facet frame", "Primeknit collar", "Controlframe outsole", "Rubber strike zones"]),
                    "in_stock": 1
                },
                {
                    "brand": "Puma",
                    "model": "Future Ultimate",
                    "category": "Soccer",
                    "price": 230.00,
                    "sizes": json.dumps([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]),
                    "colors": json.dumps(["Black/Yellow", "White/Red", "Blue"]),
                    "description": "Adaptive fit soccer cleats with FUZIONFIT+ technology",
                    "features": json.dumps(["FUZIONFIT+ compression band", "Dynamic Motion System", "Nano Grip technology", "Lightweight design"]),
                    "in_stock": 1
                },
                
                # Tennis Shoes
                {
                    "brand": "Nike",
                    "model": "Court Zoom Vapor Pro 2",
                    "category": "Tennis",
                    "price": 140.00,
                    "sizes": json.dumps([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["White/Black", "Blue", "Pink"]),
                    "description": "Lightweight tennis shoes for quick movements",
                    "features": json.dumps(["Zoom Air cushioning", "Durable rubber outsole", "Breathable upper", "Court-specific traction"]),
                    "in_stock": 1
                },
                {
                    "brand": "Adidas",
                    "model": "Barricade 13",
                    "category": "Tennis",
                    "price": 150.00,
                    "sizes": json.dumps([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["White/Black", "Blue", "Red"]),
                    "description": "Durable tennis shoes for aggressive players",
                    "features": json.dumps(["Boost cushioning", "Adituff toe protection", "Adiwear outsole", "TPU midfoot shank"]),
                    "in_stock": 1
                },
                {
                    "brand": "Asics",
                    "model": "Gel-Resolution 9",
                    "category": "Tennis",
                    "price": 160.00,
                    "sizes": json.dumps([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["White/Blue", "Black/Red", "Clay"]),
                    "description": "Stability tennis shoes for baseline players",
                    "features": json.dumps(["GEL cushioning", "DYNAWRAP technology", "AHARPLUS outsole", "PGuard toe protector"]),
                    "in_stock": 1
                },
                
                # Training/CrossFit Shoes
                {
                    "brand": "Nike",
                    "model": "Metcon 9",
                    "category": "Training",
                    "price": 150.00,
                    "sizes": json.dumps([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13, 14]),
                    "colors": json.dumps(["Black/White", "Blue", "Pink", "Camo"]),
                    "description": "Versatile training shoes for CrossFit and HIIT",
                    "features": json.dumps(["Hyperlift plate", "React foam", "Rope wrap", "Wide heel base"]),
                    "in_stock": 1
                },
                {
                    "brand": "Reebok",
                    "model": "Nano X3",
                    "category": "Training",
                    "price": 150.00,
                    "sizes": json.dumps([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13, 14]),
                    "colors": json.dumps(["Black/White", "Blue", "Red", "Camo"]),
                    "description": "CrossFit training shoes with Flexweave upper",
                    "features": json.dumps(["Flexweave Knit upper", "Floatride Energy Foam", "Lift and Run Chassis", "Meta-split outsole"]),
                    "in_stock": 1
                },
                {
                    "brand": "Nobull",
                    "model": "Trainer+",
                    "category": "Training",
                    "price": 159.00,
                    "sizes": json.dumps([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13, 14]),
                    "colors": json.dumps(["Black", "White", "Grey", "Camo"]),
                    "description": "Durable training shoes with SuperFabric upper",
                    "features": json.dumps(["SuperFabric upper", "Stable platform", "Rope guard", "Versatile traction"]),
                    "in_stock": 1
                },
                
                # Hiking Shoes
                {
                    "brand": "Salomon",
                    "model": "X Ultra 4 GTX",
                    "category": "Hiking",
                    "price": 150.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black/Grey", "Brown", "Blue"]),
                    "description": "Waterproof hiking shoes for technical trails",
                    "features": json.dumps(["Gore-Tex waterproof", "Contagrip outsole", "Advanced Chassis", "Quicklace system"]),
                    "in_stock": 1
                },
                {
                    "brand": "Salomon",
                    "model": "X Ultra 5 Mid GTX",
                    "category": "Hiking",
                    "price": 180.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black/Magnet", "Brown", "Blue/Grey"]),
                    "description": "Mid-height waterproof hiking boots with enhanced stability",
                    "features": json.dumps(["Gore-Tex waterproof", "Advanced Chassis", "Contagrip MA outsole", "Quicklace system", "Ankle support"]),
                    "in_stock": 1
                },
                {
                    "brand": "Merrell",
                    "model": "Moab 3 Mid GTX",
                    "category": "Hiking",
                    "price": 155.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13, 14]),
                    "colors": json.dumps(["Brown", "Black", "Grey"]),
                    "description": "Classic hiking boots with Gore-Tex protection",
                    "features": json.dumps(["Gore-Tex waterproof", "Vibram TC5+ outsole", "Air cushion heel", "Protective toe cap", "Bellows tongue"]),
                    "in_stock": 1
                },
                {
                    "brand": "The North Face",
                    "model": "Vectiv Fastpack Futurelight",
                    "category": "Hiking",
                    "price": 170.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black/Grey", "Blue", "Green"]),
                    "description": "Fast hiking shoes with rocker technology",
                    "features": json.dumps(["Futurelight waterproof", "VECTIV technology", "Surface Control outsole", "Lightweight design"]),
                    "in_stock": 1
                },
                {
                    "brand": "LOWA",
                    "model": "Renegade EVO GTX Mid",
                    "category": "Hiking",
                    "price": 280.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Brown", "Black", "Grey"]),
                    "description": "Premium leather hiking boots with enhanced stability",
                    "features": json.dumps(["Nubuck leather upper", "Gore-Tex waterproof", "Monowrap frame", "Vibram outsole", "DuraPU midsole"]),
                    "in_stock": 1
                },
                {
                    "brand": "La Sportiva",
                    "model": "Ultra Raptor II Mid Leather GTX",
                    "category": "Hiking",
                    "price": 219.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Brown/Black", "Grey", "Blue"]),
                    "description": "Technical hiking boots with leather upper",
                    "features": json.dumps(["Leather upper", "Gore-Tex waterproof", "FriXion XT outsole", "Impact Brake System", "STB Control"]),
                    "in_stock": 1
                },
                {
                    "brand": "La Sportiva",
                    "model": "Prodigio Max",
                    "category": "Hiking",
                    "price": 185.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black/Yellow", "Grey/Blue", "Brown"]),
                    "description": "Maximum cushioned hiking shoes for long distances",
                    "features": json.dumps(["InFuse midsole", "FriXion XT outsole", "Breathable mesh", "Rock guard", "Ortholite insole"]),
                    "in_stock": 1
                },
                {
                    "brand": "Scarpa",
                    "model": "Rush Trek GTX",
                    "category": "Hiking",
                    "price": 200.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Brown", "Black/Grey", "Blue"]),
                    "description": "Versatile hiking boots for varied terrain",
                    "features": json.dumps(["Gore-Tex waterproof", "Suede leather upper", "Vibram Megagrip", "Dual-density EVA", "Ankle support"]),
                    "in_stock": 1
                },
                {
                    "brand": "Scarpa",
                    "model": "Zodiac Plus GTX",
                    "category": "Hiking",
                    "price": 240.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Grey/Orange", "Black", "Blue"]),
                    "description": "Technical approach shoes for alpine hiking",
                    "features": json.dumps(["Gore-Tex waterproof", "Suede leather", "Vibram Megagrip", "Climbing zone", "Sock-Fit construction"]),
                    "in_stock": 1
                },
                {
                    "brand": "Hoka",
                    "model": "Speedgoat 6",
                    "category": "Trail Running",
                    "price": 155.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black/Orange", "Blue", "Green", "Grey"]),
                    "description": "Technical trail running shoes for rugged terrain",
                    "features": json.dumps(["Vibram Megagrip outsole", "Cushioned midsole", "Protective toe cap", "Trail-specific traction", "Improved fit"]),
                    "in_stock": 1
                },
                {
                    "brand": "Arc'teryx",
                    "model": "Aerios FL Mid GTX",
                    "category": "Hiking",
                    "price": 220.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black", "Grey/Blue", "Brown"]),
                    "description": "Lightweight fast-hiking boots with Gore-Tex",
                    "features": json.dumps(["Gore-Tex waterproof", "Adaptive Fit Lite", "Vibram Megagrip", "EVA midsole", "Lightweight design"]),
                    "in_stock": 1
                },
                {
                    "brand": "Oboz",
                    "model": "Sawtooth X Mid",
                    "category": "Hiking",
                    "price": 185.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13, 14]),
                    "colors": json.dumps(["Brown", "Black", "Grey"]),
                    "description": "Comfortable hiking boots with excellent support",
                    "features": json.dumps(["B-DRY waterproof", "O FIT Insole", "Sawtooth outsole", "Nubuck leather", "Adaptive Cushioning"]),
                    "in_stock": 1
                },
                {
                    "brand": "Keen",
                    "model": "Targhee III Mid WP",
                    "category": "Hiking",
                    "price": 165.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13, 14, 15]),
                    "colors": json.dumps(["Brown", "Black/Grey", "Olive"]),
                    "description": "Waterproof hiking boots with roomy toe box",
                    "features": json.dumps(["KEEN.DRY waterproof", "Leather upper", "External Support Shank", "Metatomical footbed", "All-terrain outsole"]),
                    "in_stock": 1
                },
                {
                    "brand": "Danner",
                    "model": "Trail 2650 Mid GTX",
                    "category": "Hiking",
                    "price": 190.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Brown", "Black", "Grey/Blue"]),
                    "description": "Lightweight hiking boots inspired by Pacific Crest Trail",
                    "features": json.dumps(["Gore-Tex waterproof", "Vibram 460 outsole", "EXO Heel System", "Ortholite footbed", "Lightweight design"]),
                    "in_stock": 1
                },
                
                # Casual/Lifestyle Shoes
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
                    "brand": "On",
                    "model": "Cloud 5",
                    "category": "Casual",
                    "price": 140.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["All White", "Black/White", "Grey"]),
                    "description": "Swiss-engineered lifestyle shoes with CloudTec cushioning",
                    "features": json.dumps(["CloudTec cushioning", "Speed lacing system", "Lightweight design", "Versatile style"]),
                    "in_stock": 1
                },
                
                # Walking Shoes
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
                    "brand": "New Balance",
                    "model": "990v6",
                    "category": "Walking",
                    "price": 185.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13, 14]),
                    "colors": json.dumps(["Grey", "Black", "Navy"]),
                    "description": "Premium Made in USA walking shoes",
                    "features": json.dumps(["FuelCell cushioning", "ENCAP midsole", "Pigskin/mesh upper", "Made in USA"]),
                    "in_stock": 1
                },
                
                # Boots
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
                    "brand": "Red Wing",
                    "model": "Iron Ranger",
                    "category": "Boots",
                    "price": 350.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Amber Harness", "Black", "Copper"]),
                    "description": "Heritage work boots with premium leather",
                    "features": json.dumps(["Full-grain leather", "Cork midsole", "Vibram outsole", "Goodyear welt construction"]),
                    "in_stock": 1
                },
                
                # Skateboarding Shoes
                {
                    "brand": "Nike",
                    "model": "SB Dunk Low",
                    "category": "Skateboarding",
                    "price": 110.00,
                    "sizes": json.dumps([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black/White", "University Red", "Navy"]),
                    "description": "Classic skateboarding shoes with Zoom Air",
                    "features": json.dumps(["Zoom Air cushioning", "Padded tongue", "Rubber outsole", "Durable leather"]),
                    "in_stock": 1
                },
                {
                    "brand": "Vans",
                    "model": "Sk8-Hi Pro",
                    "category": "Skateboarding",
                    "price": 75.00,
                    "sizes": json.dumps([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black/White", "Navy", "Checkerboard"]),
                    "description": "High-top skate shoes with enhanced durability",
                    "features": json.dumps(["Duracap reinforcement", "PopCush insole", "Waffle outsole", "Padded collar"]),
                    "in_stock": 1
                },
                {
                    "brand": "Adidas",
                    "model": "Busenitz Pro",
                    "category": "Skateboarding",
                    "price": 90.00,
                    "sizes": json.dumps([6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["Black/White", "Navy/Gold", "Grey"]),
                    "description": "Pro skate shoes with Adiprene cushioning",
                    "features": json.dumps(["Adiprene cushioning", "Suede upper", "Geofit collar", "Grippy outsole"]),
                    "in_stock": 1
                },
                
                # Cycling Shoes
                {
                    "brand": "Shimano",
                    "model": "RC300",
                    "category": "Cycling",
                    "price": 150.00,
                    "sizes": json.dumps([39, 40, 41, 42, 43, 44, 45, 46, 47]),
                    "colors": json.dumps(["Black", "White", "Blue"]),
                    "description": "Road cycling shoes with carbon-reinforced sole",
                    "features": json.dumps(["Carbon-reinforced sole", "BOA dial", "Synthetic leather upper", "SPD-SL compatible"]),
                    "in_stock": 1
                },
                {
                    "brand": "Giro",
                    "model": "Empire E70 Knit",
                    "category": "Cycling",
                    "price": 250.00,
                    "sizes": json.dumps([39, 40, 41, 42, 43, 44, 45, 46, 47]),
                    "colors": json.dumps(["Black", "Grey", "Blue"]),
                    "description": "Premium road cycling shoes with knit upper",
                    "features": json.dumps(["Xnetic Knit upper", "EC70 carbon outsole", "Lace closure", "Lightweight design"]),
                    "in_stock": 1
                },
                
                # Golf Shoes
                {
                    "brand": "FootJoy",
                    "model": "Pro SL Carbon",
                    "category": "Golf",
                    "price": 200.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["White", "Black", "Grey"]),
                    "description": "Premium golf shoes with carbon fiber plate",
                    "features": json.dumps(["Carbon fiber plate", "Infinity outsole", "Waterproof leather", "Athletic fit"]),
                    "in_stock": 1
                },
                {
                    "brand": "Adidas",
                    "model": "CodeChaos 22",
                    "category": "Golf",
                    "price": 180.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["White/Black", "Grey", "Navy"]),
                    "description": "Spikeless golf shoes with Boost cushioning",
                    "features": json.dumps(["Boost cushioning", "Primeknit upper", "Spikeless Twist Grip", "Waterproof"]),
                    "in_stock": 1
                },
                {
                    "brand": "Nike",
                    "model": "Air Zoom Infinity Tour NEXT%",
                    "category": "Golf",
                    "price": 220.00,
                    "sizes": json.dumps([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13]),
                    "colors": json.dumps(["White", "Black", "Blue"]),
                    "description": "Tour-level golf shoes with Zoom Air",
                    "features": json.dumps(["Zoom Air units", "Flyknit upper", "Integrated traction", "Waterproof"]),
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
            SELECT preference_type, GROUP_CONCAT(preference_value, ', ') as pref_values
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
