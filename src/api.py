"""
Open Food Facts API client for fetching product data by barcode or search.
"""
import requests
from typing import Optional

BASE_URL = "https://world.openfoodfacts.org"
USER_AGENT = "SmartFoodScanner/1.0 (jbetts3@elon.edu)"


def search_products(query: str, page_size: int = 10) -> list[dict]:
    """Search Open Food Facts by product name."""
    url = f"{BASE_URL}/cgi/search.pl"
    params = {
        "search_terms": query,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": page_size,
        "fields": "code,product_name,brands,ingredients_text,nutriscore_grade,"
                  "nova_group,categories,labels,allergens,"
                  "energy-kcal_100g,fat_100g,saturated-fat_100g,trans-fat_100g,"
                  "carbohydrates_100g,sugars_100g,fiber_100g,proteins_100g,"
                  "salt_100g,sodium_100g,additives_n,nutriscore_score,"
                  "image_front_small_url,image_url"
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("products", [])
    except Exception as e:
        print(f"Search error: {e}")
        return []


def get_product_by_barcode(barcode: str) -> Optional[dict]:
    """Fetch a single product by barcode from Open Food Facts."""
    url = f"{BASE_URL}/api/v2/product/{barcode}"
    params = {
        "fields": "code,product_name,brands,ingredients_text,nutriscore_grade,"
                  "nova_group,categories,labels,allergens,"
                  "energy-kcal_100g,fat_100g,saturated-fat_100g,trans-fat_100g,"
                  "carbohydrates_100g,sugars_100g,fiber_100g,proteins_100g,"
                  "salt_100g,sodium_100g,additives_n,nutriscore_score,"
                  "image_front_small_url,image_url,serving_quantity,"
                  "vitamin-a_100g,vitamin-c_100g,calcium_100g,iron_100g,"
                  "cholesterol_100g"
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == 1:
            return data.get("product")
        return None
    except Exception as e:
        print(f"Barcode lookup error: {e}")
        return None
