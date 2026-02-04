#!/usr/bin/env python3
"""
Test script for Strands Agent tools
"""

import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

print("=== Testing Database Connection ===")
from shoe_database import ShoeDatabase

db = ShoeDatabase()
print(f"✓ Database initialized at: {db.db_path}")
print(f"✓ Total shoes: {len(db.search_shoes())}")
print(f"✓ Total brands: {len(db.get_all_brands())}")
print(f"✓ Categories: {', '.join(db.get_all_categories())}")

print("\n=== Testing Shoe Tools (without Strands decorator) ===")

# Test search functions directly on database
print("\n1. Search by brand (Nike):")
nike_shoes = db.search_shoes(brand="Nike")
print(f"   Found {len(nike_shoes)} Nike shoes")
if nike_shoes:
    print(f"   Example: {nike_shoes[0]['brand']} {nike_shoes[0]['model']} - ${nike_shoes[0]['price']}")

print("\n2. Search by category (Running):")
running_shoes = db.search_shoes(category="Running")
print(f"   Found {len(running_shoes)} running shoes")

print("\n3. Search by price range ($100-$150):")
price_range = db.search_shoes(min_price=100, max_price=150)
print(f"   Found {len(price_range)} shoes in range")

print("\n4. Get all brands:")
brands = db.get_all_brands()
print(f"   {len(brands)} brands: {', '.join(brands[:10])}...")

print("\n5. Get all categories:")
categories = db.get_all_categories()
print(f"   Categories: {', '.join(categories)}")

print("\n=== Testing User Management ===")

print("\n6. Find or create user:")
user = db.find_user_by_name("John", "Doe")
if not user:
    user_id = db.create_user("John", "Doe")
    print(f"   Created new user with ID: {user_id}")
    user = db.find_user_by_name("John", "Doe")
else:
    print(f"   Found existing user: {user['first_name']} {user['last_name']} (ID: {user['id']})")

print("\n7. Save user preference:")
db.add_user_preference(user['id'], "brand", "Nike")
db.add_user_preference(user['id'], "size", "10")
print("   Saved preferences: brand=Nike, size=10")

print("\n8. Get user preferences:")
prefs = db.get_user_preferences_summary(user['id'])
print(f"   User preferences: {prefs}")

print("\n=== Testing Strands Tools Import ===")
try:
    from shoe_tools import (
        search_shoes_by_brand,
        list_all_brands,
        find_or_create_user
    )
    print("✓ Successfully imported Strands tools")
    
    print("\n9. Test list_all_brands tool:")
    result = list_all_brands()
    print(f"   Result type: {type(result)}")
    print(f"   Result: {result[:100]}...")
    
    print("\n10. Test search_shoes_by_brand tool:")
    result = search_shoes_by_brand("Nike")
    print(f"   Result type: {type(result)}")
    print(f"   Result length: {len(result)}")
    print(f"   First 200 chars: {result[:200]}...")
    
    print("\n11. Test find_or_create_user tool:")
    result = find_or_create_user("Jane", "Smith")
    print(f"   Result type: {type(result)}")
    print(f"   Result: {result[:200]}...")
    
    print("\n✅ All tests passed!")
    
except ImportError as e:
    print(f"✗ Failed to import Strands tools: {e}")
    print("\nNote: This is expected if strands-agents is not installed.")
    print("Install with: pip install strands-agents")
except Exception as e:
    print(f"✗ Error testing tools: {e}")
    import traceback
    traceback.print_exc()
