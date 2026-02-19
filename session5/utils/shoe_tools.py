"""
Strands Agent Tools for Shoe Department Assistant

This module provides tools for the Strands agent to interact with the shoe database.
"""

from strands import tool
from typing import Optional, List, Dict
from .shoe_database import ShoeDatabase
import json


# Initialize database connection
db = ShoeDatabase()


@tool
def search_shoes_by_brand(brand: str) -> str:
    """Search for shoes by brand name.
    
    Args:
        brand: The brand name to search for (e.g., "Nike", "Adidas", "Converse")
    
    Returns:
        JSON string with list of matching shoes including model, price, sizes, and colors
    """
    results = db.search_shoes(brand=brand)
    
    if not results:
        return f"No shoes found for brand '{brand}'. Available brands: {', '.join(db.get_all_brands())}"
    
    # Format results for better readability
    formatted_results = []
    for shoe in results:
        formatted_results.append({
            "id": shoe["id"],
            "brand": shoe["brand"],
            "model": shoe["model"],
            "category": shoe["category"],
            "price": f"${shoe['price']:.2f}",
            "available_sizes": shoe["sizes"],
            "available_colors": shoe["colors"],
            "description": shoe["description"]
        })
    
    return json.dumps(formatted_results, indent=2)


@tool
def search_shoes_by_category(category: str) -> str:
    """Search for shoes by category.
    
    Args:
        category: The category to search for (e.g., "Running", "Casual", "Boots", "Walking")
    
    Returns:
        JSON string with list of matching shoes including brand, model, price, and features
    """
    results = db.search_shoes(category=category)
    
    if not results:
        return f"No shoes found in category '{category}'. Available categories: {', '.join(db.get_all_categories())}"
    
    # Format results for better readability
    formatted_results = []
    for shoe in results:
        formatted_results.append({
            "id": shoe["id"],
            "brand": shoe["brand"],
            "model": shoe["model"],
            "price": f"${shoe['price']:.2f}",
            "available_sizes": shoe["sizes"],
            "available_colors": shoe["colors"],
            "features": shoe["features"],
            "description": shoe["description"]
        })
    
    return json.dumps(formatted_results, indent=2)


@tool
def search_shoes_by_price_range(min_price: float, max_price: float) -> str:
    """Search for shoes within a specific price range.
    
    Args:
        min_price: Minimum price in dollars (e.g., 50.0)
        max_price: Maximum price in dollars (e.g., 150.0)
    
    Returns:
        JSON string with list of matching shoes sorted by price
    """
    results = db.search_shoes(min_price=min_price, max_price=max_price)
    
    if not results:
        return f"No shoes found in price range ${min_price:.2f} - ${max_price:.2f}"
    
    # Sort by price
    results.sort(key=lambda x: x['price'])
    
    # Format results for better readability
    formatted_results = []
    for shoe in results:
        formatted_results.append({
            "id": shoe["id"],
            "brand": shoe["brand"],
            "model": shoe["model"],
            "category": shoe["category"],
            "price": f"${shoe['price']:.2f}",
            "available_sizes": shoe["sizes"],
            "available_colors": shoe["colors"]
        })
    
    return json.dumps(formatted_results, indent=2)


@tool
def get_shoe_details(shoe_id: int) -> str:
    """Get detailed information about a specific shoe.
    
    Args:
        shoe_id: The unique ID of the shoe to retrieve details for
    
    Returns:
        JSON string with complete shoe details including all features and specifications
    """
    shoe = db.get_shoe_by_id(shoe_id)
    
    if not shoe:
        return f"No shoe found with ID {shoe_id}"
    
    # Format complete details
    details = {
        "id": shoe["id"],
        "brand": shoe["brand"],
        "model": shoe["model"],
        "category": shoe["category"],
        "price": f"${shoe['price']:.2f}",
        "available_sizes": shoe["sizes"],
        "available_colors": shoe["colors"],
        "description": shoe["description"],
        "features": shoe["features"],
        "in_stock": bool(shoe["in_stock"])
    }
    
    return json.dumps(details, indent=2)


@tool
def list_all_brands() -> str:
    """Get a list of all available shoe brands in the store.
    
    Returns:
        Comma-separated list of brand names
    """
    brands = db.get_all_brands()
    return ", ".join(brands)


@tool
def list_all_categories() -> str:
    """Get a list of all available shoe categories in the store.
    
    Returns:
        Comma-separated list of category names
    """
    categories = db.get_all_categories()
    return ", ".join(categories)


@tool
def get_recommendations_for_activity(activity: str) -> str:
    """Get shoe recommendations based on an activity or use case.
    
    Args:
        activity: The activity or use case (e.g., "running", "walking", "casual wear", "hiking")
    
    Returns:
        JSON string with recommended shoes for the specified activity
    """
    activity_lower = activity.lower()
    
    # Map activities to categories
    category_mapping = {
        "running": "Running",
        "jogging": "Running",
        "marathon": "Running",
        "walking": "Walking",
        "casual": "Casual",
        "everyday": "Casual",
        "hiking": "Boots",
        "outdoor": "Boots",
        "work": "Boots"
    }
    
    # Find matching category
    category = None
    for key, value in category_mapping.items():
        if key in activity_lower:
            category = value
            break
    
    if category:
        results = db.search_shoes(category=category)
    else:
        # If no specific category, return popular options
        results = db.search_shoes()[:5]
    
    if not results:
        return f"No specific recommendations found for '{activity}'. Try browsing our categories: {', '.join(db.get_all_categories())}"
    
    # Format recommendations
    recommendations = []
    for shoe in results[:5]:  # Limit to top 5
        recommendations.append({
            "id": shoe["id"],
            "brand": shoe["brand"],
            "model": shoe["model"],
            "price": f"${shoe['price']:.2f}",
            "why_recommended": shoe["description"],
            "key_features": shoe["features"][:3]  # Top 3 features
        })
    
    return json.dumps(recommendations, indent=2)


@tool
def find_or_create_user(first_name: str, last_name: str) -> str:
    """Find an existing user or create a new user in the database.
    
    Args:
        first_name: User's first name
        last_name: User's last name
    
    Returns:
        JSON string with user information and their preferences if they exist
    """
    # Try to find existing user
    user = db.find_user_by_name(first_name, last_name)
    
    if user:
        # Update last visit
        db.update_user_last_visit(user['id'])
        
        # Get user preferences
        preferences = db.get_user_preferences_summary(user['id'])
        
        result = {
            "status": "existing_user",
            "user_id": user['id'],
            "first_name": user['first_name'],
            "last_name": user['last_name'],
            "member_since": user['created_at'],
            "last_visit": user['last_visit'],
            "preferences": preferences if preferences else "No preferences saved yet"
        }
    else:
        # Create new user
        user_id = db.create_user(first_name, last_name)
        
        result = {
            "status": "new_user",
            "user_id": user_id,
            "first_name": first_name,
            "last_name": last_name,
            "message": "Welcome! New user profile created."
        }
    
    return json.dumps(result, indent=2)


@tool
def save_user_preference(first_name: str, last_name: str, preference_type: str, preference_value: str) -> str:
    """Save a user's shoe preference to the database.
    
    Args:
        first_name: User's first name
        last_name: User's last name
        preference_type: Type of preference (e.g., "brand", "category", "price_range", "size", "color", "activity")
        preference_value: The preference value (e.g., "Nike", "Running", "under $150", "10", "Black", "marathon training")
    
    Returns:
        Confirmation message
    """
    # Find user
    user = db.find_user_by_name(first_name, last_name)
    
    if not user:
        return f"User {first_name} {last_name} not found. Please use find_or_create_user first."
    
    # Save preference
    db.add_user_preference(user['id'], preference_type, preference_value)
    
    return f"Preference saved: {preference_type} = {preference_value} for {first_name} {last_name}"


@tool
def get_user_preferences(first_name: str, last_name: str) -> str:
    """Get all saved preferences for a user.
    
    Args:
        first_name: User's first name
        last_name: User's last name
    
    Returns:
        JSON string with all user preferences
    """
    # Find user
    user = db.find_user_by_name(first_name, last_name)
    
    if not user:
        return f"User {first_name} {last_name} not found in database."
    
    # Get preferences
    preferences = db.get_user_preferences(user['id'])
    
    if not preferences:
        return f"No preferences saved for {first_name} {last_name} yet."
    
    result = {
        "user": f"{first_name} {last_name}",
        "user_id": user['id'],
        "preferences": preferences
    }
    
    return json.dumps(result, indent=2)
