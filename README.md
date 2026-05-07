# Smart Food Scanner

AI-powered food product analyzer that predicts **Nutri-Score** (A-E) and **NOVA Group** (1-4) using NLP + structured nutrition data.

Built as part of MSBA 6660: Deep Learning for Business (Elon University, Spring 2026).

## Features

- **Search** for any food product by name via the Open Food Facts API
- - **Scan** a barcode (UPC/EAN) to instantly fetch product data
  - - **Manual entry** for products not in the database
    - - **AI predictions** using TF-IDF text features combined with structured nutrition data
      - - **Visual confidence scores** with probability breakdowns
       
        - ## Quick Start
       
        - ```bash
          pip install -r requirements.txt
          streamlit run app.py
          ```

          ## Training Models

          ```bash
          python train_models.py --off-path path/to/en.openfoodfacts.org.products.csv
          ```

          ## Data Sources

          - [Open Food Facts](https://world.openfoodfacts.org/data)
          - - [USDA FoodData Central](https://fdc.nal.usda.gov/download-datasets)
