"""
Test script for user management functionality
"""

import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from utils.shoe_database import ShoeDatabase

def test_user_management():
    """Test user management features"""
    print("Testing User Management Features\n")
    print("=" * 50)
    
    # Initialize database
    db = ShoeDatabase("test_shoe_inventory.db")
    
    # Test 1: Create new user
    print("\n1. Creating new user: John Smith")
    user_id = db.create_user("John", "Smith")
    print(f"   ✓ User created with ID: {user_id}")
    
    # Test 2: Find existing user
    print("\n2. Finding user: John Smith")
    user = db.find_user_by_name("John", "Smith")
    if user:
        print(f"   ✓ User found: {user['first_name']} {user['last_name']}")
        print(f"   - User ID: {user['id']}")
        print(f"   - Created: {user['created_at']}")
    
    # Test 3: Add preferences
    print("\n3. Adding preferences for John Smith")
    db.add_user_preference(user_id, "brand", "Nike")
    db.add_user_preference(user_id, "category", "Running")
    db.add_user_preference(user_id, "size", "10")
    db.add_user_preference(user_id, "color", "Black")
    db.add_user_preference(user_id, "price_range", "under $150")
    print("   ✓ Preferences added")
    
    # Test 4: Get preferences
    print("\n4. Retrieving preferences for John Smith")
    preferences = db.get_user_preferences(user_id)
    for pref in preferences:
        print(f"   - {pref['type']}: {pref['value']}")
    
    # Test 5: Get preferences summary
    print("\n5. Getting preferences summary")
    summary = db.get_user_preferences_summary(user_id)
    for pref_type, values in summary.items():
        print(f"   - {pref_type}: {values}")
    
    # Test 6: Create another user
    print("\n6. Creating another user: Jane Doe")
    user_id_2 = db.create_user("Jane", "Doe")
    print(f"   ✓ User created with ID: {user_id_2}")
    
    # Test 7: Add preferences for Jane
    print("\n7. Adding preferences for Jane Doe")
    db.add_user_preference(user_id_2, "brand", "Adidas")
    db.add_user_preference(user_id_2, "brand", "New Balance")
    db.add_user_preference(user_id_2, "category", "Walking")
    db.add_user_preference(user_id_2, "size", "8.5")
    print("   ✓ Preferences added")
    
    # Test 8: Get Jane's preferences
    print("\n8. Retrieving preferences for Jane Doe")
    summary_2 = db.get_user_preferences_summary(user_id_2)
    for pref_type, values in summary_2.items():
        print(f"   - {pref_type}: {values}")
    
    # Test 9: Update last visit
    print("\n9. Updating last visit for John Smith")
    db.update_user_last_visit(user_id)
    user_updated = db.find_user_by_name("John", "Smith")
    print(f"   ✓ Last visit updated: {user_updated['last_visit']}")
    
    print("\n" + "=" * 50)
    print("All tests completed successfully! ✓")
    
    db.close()

if __name__ == "__main__":
    test_user_management()
