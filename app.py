"""
Smart Food Scanner — Streamlit App
Predicts Nutri-Score (A-E) and NOVA Group (1-4) for food products
using NLP + structured nutrition data.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.api import search_products, get_product_by_barcode
from src.preprocessing import extract_numeric_features, extract_text, extract_features_from_manual_input
from src.predictor import FoodPredictor

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Food Scanner",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load models ──────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    pred = FoodPredictor()
    success = pred.load()
    return pred, success

predictor, models_loaded = load_predictor()

# ── Styling ──────────────────────────────────────────────────
NUTRISCORE_COLORS = {
    "A": "#038141", "B": "#85BB2F", "C": "#FECB02",
    "D": "#EE8100", "E": "#E63E11"
}
NUTRISCORE_LABELS = {
    "A": "Excellent", "B": "Good", "C": "Average",
    "D": "Poor", "E": "Bad"
}
NOVA_COLORS = {
    1: "#038141", 2: "#85BB2F", 3: "#EE8100", 4: "#E63E11"
}
NOVA_LABELS = {
    1: "Unprocessed / Minimally processed",
    2: "Processed culinary ingredients",
    3: "Processed foods",
    4: "Ultra-processed foods"
}


# ── Visualization helpers ────────────────────────────────────
def nutriscore_badge(grade: str, confidence: float, probabilities: dict):
    """Render a Nutri-Score badge with confidence gauge."""
    color = NUTRISCORE_COLORS.get(grade, "#999")
    label = NUTRISCORE_LABELS.get(grade, "Unknown")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div style="text-align:center; padding:20px; background:{color};
                    border-radius:16px; color:white; margin-bottom:10px;">
            <div style="font-size:60px; font-weight:bold;">{grade}</div>
            <div style="font-size:16px;">{label}</div>
            <div style="font-size:13px; opacity:0.85;">{confidence:.0%} confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Probability bar chart
        grades = sorted(probabilities.keys())
        probs = [probabilities[g] for g in grades]
        colors = [NUTRISCORE_COLORS.get(g, "#ccc") for g in grades]
        fig = go.Figure(go.Bar(
            x=grades, y=probs, marker_color=colors,
            text=[f"{p:.1%}" for p in probs], textposition="outside"
        ))
        fig.update_layout(
            title="Class Probabilities", yaxis_title="Probability",
            yaxis_range=[0, 1], height=250, margin=dict(t=40, b=20, l=40, r=20),
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)


def nova_badge(group: int, confidence: float, probabilities: dict):
    """Render a NOVA Group badge with confidence gauge."""
    color = NOVA_COLORS.get(group, "#999")
    label = NOVA_LABELS.get(group, "Unknown")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div style="text-align:center; padding:20px; background:{color};
                    border-radius:16px; color:white; margin-bottom:10px;">
            <div style="font-size:60px; font-weight:bold;">NOVA {group}</div>
            <div style="font-size:14px;">{label}</div>
            <div style="font-size:13px; opacity:0.85;">{confidence:.0%} confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        groups = sorted(probabilities.keys())
        probs = [probabilities[g] for g in groups]
        colors = [NOVA_COLORS.get(g, "#ccc") for g in groups]
        fig = go.Figure(go.Bar(
            x=[f"NOVA {g}" for g in groups], y=probs, marker_color=colors,
            text=[f"{p:.1%}" for p in probs], textposition="outside"
        ))
        fig.update_layout(
            title="Class Probabilities", yaxis_title="Probability",
            yaxis_range=[0, 1], height=250, margin=dict(t=40, b=20, l=40, r=20),
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)


def display_nutrition_table(product: dict):
    """Show a formatted nutrition facts table."""
    nutrients = {
        "Energy (kcal/100g)": product.get("energy-kcal_100g"),
        "Fat (g/100g)": product.get("fat_100g"),
        "Saturated Fat (g/100g)": product.get("saturated-fat_100g"),
        "Carbohydrates (g/100g)": product.get("carbohydrates_100g"),
        "Sugars (g/100g)": product.get("sugars_100g"),
        "Fiber (g/100g)": product.get("fiber_100g"),
        "Protein (g/100g)": product.get("proteins_100g"),
        "Salt (g/100g)": product.get("salt_100g"),
        "Sodium (g/100g)": product.get("sodium_100g"),
    }
    rows = []
    for name, val in nutrients.items():
        if val is not None and val != "":
            try:
                rows.append({"Nutrient": name, "Value": f"{float(val):.1f}"})
            except (ValueError, TypeError):
                pass
    if rows:
        st.table(rows)


# ── Main App ─────────────────────────────────────────────────
st.title("Smart Food Scanner")
st.markdown("Predict **Nutri-Score** and **NOVA Group** for any food product using AI/NLP.")

if not models_loaded:
    st.error(
        "**Models not found.** Please train the models first by running:\n\n"
        "```bash\n"
        "python train_models.py --off-path path/to/en.openfoodfacts.org.products.csv\n"
        "```\n\n"
        "See the README for details."
    )
    st.stop()

# Sidebar
st.sidebar.header("How to Use")
st.sidebar.markdown(
    "1. **Search** for a product by name\n"
    "2. Or enter a **barcode** number\n"
    "3. Or **manually enter** ingredients and nutrition\n\n"
    "The AI model predicts health scores based on both "
    "the ingredient text (NLP) and nutrition numbers."
)
st.sidebar.divider()
st.sidebar.markdown(
    "**About the Model**\n\n"
    "Trained on Open Food Facts data using TF-IDF text features "
    "combined with structured nutrition data. "
    "Logistic Regression with balanced class weights."
)

# Input tabs
tab_search, tab_barcode, tab_manual = st.tabs([
    "Search by Name", "Lookup by Barcode", "Manual Entry"
])

selected_product = None

with tab_search:
    query = st.text_input("Search for a food product:", placeholder="e.g. Cheerios, Nutella, KIND bar")
    if query:
        with st.spinner("Searching Open Food Facts..."):
            results = search_products(query, page_size=8)
        if results:
            options = {}
            for p in results:
                name = p.get("product_name", "Unknown")
                brand = p.get("brands", "")
                label = f"{name} — {brand}" if brand else name
                options[label] = p

            choice = st.selectbox("Select a product:", list(options.keys()))
            if choice:
                selected_product = options[choice]
        else:
            st.warning("No products found. Try a different search term.")

with tab_barcode:
    barcode = st.text_input("Enter barcode (UPC/EAN):", placeholder="e.g. 3017620422003")
    if barcode:
        with st.spinner("Looking up barcode..."):
            selected_product = get_product_by_barcode(barcode.strip())
        if selected_product is None:
            st.warning("Product not found in Open Food Facts database.")

with tab_manual:
    st.markdown("Enter product details manually:")
    m_name = st.text_input("Product name:", key="manual_name")
    m_ingredients = st.text_area(
        "Ingredients list:",
        placeholder="e.g. Whole grain oats, sugar, corn starch, honey, salt...",
        key="manual_ing"
    )
    st.markdown("**Nutrition Facts (per 100g):**")
    mc1, mc2, mc3, mc4 = st.columns(4)
    m_energy = mc1.number_input("Energy (kcal)", 0.0, 1000.0, 0.0, key="m_en")
    m_fat = mc2.number_input("Fat (g)", 0.0, 100.0, 0.0, key="m_fat")
    m_satfat = mc3.number_input("Sat. Fat (g)", 0.0, 100.0, 0.0, key="m_sf")
    m_transfat = mc4.number_input("Trans Fat (g)", 0.0, 100.0, 0.0, key="m_tf")
    mc5, mc6, mc7, mc8 = st.columns(4)
    m_carbs = mc5.number_input("Carbs (g)", 0.0, 100.0, 0.0, key="m_carb")
    m_sugars = mc6.number_input("Sugars (g)", 0.0, 100.0, 0.0, key="m_sug")
    m_fiber = mc7.number_input("Fiber (g)", 0.0, 100.0, 0.0, key="m_fib")
    m_protein = mc8.number_input("Protein (g)", 0.0, 100.0, 0.0, key="m_pro")
    mc9, mc10 = st.columns(2)
    m_salt = mc9.number_input("Salt (g)", 0.0, 100.0, 0.0, key="m_salt")
    m_sodium = mc10.number_input("Sodium (g)", 0.0, 100.0, 0.0, key="m_sod")

    if st.button("Analyze", key="manual_btn"):
        if m_ingredients.strip():
            numeric, text = extract_features_from_manual_input(
                ingredients=m_ingredients, product_name=m_name,
                energy_kcal=m_energy, fat=m_fat, saturated_fat=m_satfat,
                trans_fat=m_transfat, carbs=m_carbs, sugars=m_sugars,
                fiber=m_fiber, protein=m_protein, salt=m_salt, sodium=m_sodium
            )
            with st.spinner("Running prediction..."):
                result = predictor.predict(numeric, text)

            st.divider()
            st.subheader(f"Results for: {m_name or 'Manual Entry'}")

            r1, r2 = st.columns(2)
            with r1:
                st.markdown("### Nutri-Score")
                nutriscore_badge(
                    result["nutriscore"]["grade"],
                    result["nutriscore"]["confidence"],
                    result["nutriscore"]["probabilities"]
                )
            with r2:
                st.markdown("### NOVA Group")
                nova_badge(
                    result["nova"]["group"],
                    result["nova"]["confidence"],
                    result["nova"]["probabilities"]
                )
        else:
            st.warning("Please enter at least the ingredients list.")


# ── Display results for API-fetched product ──────────────────
if selected_product and (query or barcode):
    numeric = extract_numeric_features(selected_product)
    text = extract_text(selected_product)

    with st.spinner("Running AI prediction..."):
        result = predictor.predict(numeric, text)

    st.divider()
    prod_name = selected_product.get("product_name", "Unknown Product")
    prod_brand = selected_product.get("brands", "")
    st.subheader(f"{prod_name}" + (f" — {prod_brand}" if prod_brand else ""))

    # Product image
    img_url = selected_product.get("image_front_small_url") or selected_product.get("image_url")

    if img_url:
        col_img, col_space = st.columns([1, 3])
        with col_img:
            st.image(img_url, width=150)

    # Predictions
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("### Nutri-Score Prediction")
        nutriscore_badge(
            result["nutriscore"]["grade"],
            result["nutriscore"]["confidence"],
            result["nutriscore"]["probabilities"]
        )
        # Compare to official if available
        official_ns = selected_product.get("nutriscore_grade")
        if official_ns and official_ns.upper() in NUTRISCORE_COLORS:
            match = "matches" if official_ns.upper() == result["nutriscore"]["grade"] else "differs from"
            st.info(f"Official Nutri-Score: **{official_ns.upper()}** — our prediction {match} the official grade.")

    with r2:
        st.markdown("### NOVA Group Prediction")
        nova_badge(
            result["nova"]["group"],
            result["nova"]["confidence"],
            result["nova"]["probabilities"]
        )
        official_nova = selected_product.get("nova_group")
        if official_nova:
            try:
                official_nova = int(official_nova)
                match = "matches" if official_nova == result["nova"]["group"] else "differs from"
                st.info(f"Official NOVA Group: **{official_nova}** — our prediction {match} the official group.")
            except (ValueError, TypeError):
                pass

    # Details
    with st.expander("View Nutrition Facts"):
        display_nutrition_table(selected_product)

    with st.expander("View Ingredients"):
        ing = selected_product.get("ingredients_text", "Not available")
        st.write(ing if ing else "Not available")

    with st.expander("View Labels & Allergens"):
        labels = selected_product.get("labels", "Not available")
        allergens = selected_product.get("allergens", "Not available")
        st.markdown(f"**Labels:** {labels}")
        st.markdown(f"**Allergens:** {allergens}")
