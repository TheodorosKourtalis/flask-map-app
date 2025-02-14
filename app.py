#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:22:29 2025

@author:
"""

import os
import pickle
import functools
import json

from flask import Flask, request, render_template_string
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.io as pio

###############################################################################
# FLASK APP SETUP
###############################################################################
app = Flask(__name__)
app.secret_key = "supersecretkey"

###############################################################################
# COLOR SCALES
###############################################################################
color_scales = ["Viridis", "Tealrose", "Inferno", "Turbo", "Plasma", "Cividis"]

###############################################################################
# PATHS (Relative to repository root)
###############################################################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GREEK_PICKLE_PATH = os.path.join(BASE_DIR, "data", "greek_shp.pkl")
EXCEL_FOLDER = os.path.join(BASE_DIR, "data", "output_nuts3_excels")

if not os.path.exists(EXCEL_FOLDER):
    raise Exception(f"Excel folder not found at '{EXCEL_FOLDER}'. Please update the EXCEL_FOLDER variable.")

###############################################################################
# CACHED LOADERS
###############################################################################
@functools.lru_cache(None)
def load_shapefile():
    """
    Loads the preprocessed Greek shapefile from the pickle file.
    """
    if not os.path.exists(GREEK_PICKLE_PATH):
        raise Exception("Greek pickle file not found. Please generate it first and place it in the data folder.")
    with open(GREEK_PICKLE_PATH, "rb") as f:
        gdf_greece = pickle.load(f)
    return gdf_greece

@functools.lru_cache(None)
def load_default_data():
    """
    Loads and concatenates all Excel files from EXCEL_FOLDER.
    Expects columns: [NUTS_ID, YEAR, SEX, age, VALUE].
    """
    all_files = [f for f in os.listdir(EXCEL_FOLDER) if f.lower().endswith(".xlsx")]
    if not all_files:
        raise Exception(f"No Excel files found in {EXCEL_FOLDER}")
    df_list = []
    for fname in all_files:
        path = os.path.join(EXCEL_FOLDER, fname)
        try:
            df_temp = pd.read_excel(path, engine="openpyxl")
            df_list.append(df_temp)
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue
    if not df_list:
        raise Exception("No valid Excel files could be read.")
    df_all = pd.concat(df_list, ignore_index=True)
    for col in ["NUTS_ID", "YEAR", "SEX", "age", "VALUE"]:
        if col not in df_all.columns:
            raise Exception(f"Missing column '{col}' in Excel data.")
    df_all["VALUE"] = pd.to_numeric(df_all["VALUE"], errors="coerce")
    return df_all

###############################################################################
# FUNCTIONS TO EXTRACT NUTS INFO
###############################################################################
def get_nuts_info(row):
    """
    Returns a tuple (NUTS_Name, NUTS_Code).
    NUTS_Name is built from available NUTS_Level_1, NUTS_Level_2, and NUTS_Level_3 fields.
    NUTS_Code is simply the NUTS_ID.
    """
    parts = []
    for col in ["NUTS_Level_1", "NUTS_Level_2", "NUTS_Level_3"]:
        val = row.get(col)
        if val and str(val).strip() != "":
            parts.append(str(val).strip())
    nuts_name = " - ".join(parts) if parts else ""
    nuts_code = row.get("NUTS_ID", "")
    return nuts_name, nuts_code

###############################################################################
# DYNAMIC MAP CENTER & ZOOM (Based on Full Bounds)
###############################################################################
def default_map_center_and_zoom():
    """
    Computes the bounding box from the preprocessed Greek shapefile and returns
    the center and a zoom level. The zoom is set to 4.5 to slightly zoom out.
    """
    gdf = load_shapefile()
    minx, miny, maxx, maxy = gdf.total_bounds
    center_lon = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2
    zoom = 4.5  # Slightly zoomed out compared to 5
    return center_lat, center_lon, zoom

###############################################################################
# CACHED MERGED RESULTS FOR DEFAULT DATA
###############################################################################
@functools.lru_cache(None)
def get_merged_gdf(year, sex, age):
    """
    Merges the preprocessed Greek shapefile with default Excel data filtered by year, sex, and age.
    Adds two new columns: NUTS_Name and NUTS_Code.
    """
    gdf_greece = load_shapefile()
    df_all = load_default_data()
    df_filtered = df_all[
        (df_all["YEAR"] == year) & (df_all["SEX"] == sex) & (df_all["age"] == age)
    ].copy()
    merged = gdf_greece.merge(df_filtered, how="left", on="NUTS_ID")
    merged["NUTS_Name"], merged["NUTS_Code"] = zip(*merged.apply(get_nuts_info, axis=1))
    return merged

###############################################################################
# CACHED FIGURE GENERATION (for default data)
###############################################################################
@functools.lru_cache(maxsize=32)
def get_choropleth_html(year, sex, age, color_scale, language):
    """
    Returns cached HTML for the Plotly choropleth map.
    """
    trans = translations_dict.get(language, translations_dict["en"])
    merged_gdf = get_merged_gdf(year, sex, age)
    geojson_data = merged_gdf.__geo_interface__
    center_lat, center_lon, zoom = default_map_center_and_zoom()
    vals = merged_gdf["VALUE"].dropna()
    if len(vals) == 0:
        val_min, val_max = 0, 1
    else:
        val_min, val_max = vals.min(), vals.max()
        if val_min == val_max:
            val_min -= 1e-6
            val_max += 1e-6
    fig = px.choropleth_mapbox(
        merged_gdf,
        geojson=geojson_data,
        locations=merged_gdf["NUTS_ID"],
        color="VALUE",
        color_continuous_scale=color_scale,
        range_color=(val_min, val_max),
        mapbox_style="carto-positron",
        center={"lat": center_lat, "lon": center_lon},
        zoom=zoom,
        featureidkey="properties.NUTS_ID",
        custom_data=["NUTS_Name", "NUTS_Code", "VALUE"],
        labels={"VALUE": trans["value"], "NUTS_ID": trans["region"]}
    )
    fig.update_traces(
        hovertemplate=f"{trans['name']}: " + "%{customdata[0]}<br>" +
                      f"{trans['code']}: " + "%{customdata[1]}<br>" +
                      f"{trans['value']}: " + "%{customdata[2]}<extra></extra>"
    )
    fig.update_layout(
        margin=dict(r=0, t=0, l=0, b=0),
        autosize=True,
        coloraxis_colorbar=dict(
            orientation="h",
            xanchor="center",
            x=0.5,
            y=0,
            thickness=20,
            len=0.8
        )
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn",
                       config={"responsive": True, "modeBarButtonsToAdd": ["toggleFullscreen"]})

@functools.lru_cache(maxsize=32)
def get_bar_chart_html(year, sex, age, color_scale, language):
    """
    Returns cached HTML for the Plotly bar chart.
    """
    trans = translations_dict.get(language, translations_dict["en"])
    merged = get_merged_gdf(year, sex, age)
    merged = merged[merged["VALUE"] > 0]
    merged_sorted = merged.sort_values("NUTS_Name")
    fig = px.bar(
        merged_sorted,
        x="NUTS_Code",
        y="VALUE",
        color="VALUE",
        color_continuous_scale=color_scale,
        labels={"NUTS_Code": trans["region"], "VALUE": trans["value"]},
        custom_data=["NUTS_Name", "NUTS_Code", "VALUE"]
    )
    fig.update_traces(
        hovertemplate=f"{trans['name']}: " + "%{customdata[0]}<br>" +
                      f"{trans['code']}: " + "%{customdata[1]}<br>" +
                      f"{trans['value']}: " + "%{customdata[2]}<extra></extra>"
    )
    fig.update_layout(
        margin=dict(r=20, t=80, l=0, b=0),
        autosize=True,
        coloraxis_colorbar=dict(
            orientation="v",
            xanchor="left",
            x=1.02,
            yanchor="middle",
            y=0.3,
            thickness=20,
            len=0.7
        )
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn",
                       config={"responsive": True, "modeBarButtonsToAdd": ["toggleFullscreen"]})

###############################################################################
# TRANSLATIONS (UI TEXT)
###############################################################################
translations_dict = {
    "en": {
        "title": "Greek Data Maps (NUTS3)",
        "select_language": "Select Language",
        "select_dataset": "Select Data Set",
        "select_year": "Select Year",
        "select_sex": "Select Sex",
        "select_age": "Select Age Group",
        "color_scale": "Color Scale",
        "default_data": "Population-NUTS DATA",
        "generate_map": "Generate Plots",
        "choropleth_map": "Choropleth Map",
        "bar_chart": "Bar Chart",
        "region": "Region",
        "name": "Name",
        "code": "Code",
        "value": "Value",
        "apply": "Apply"
    },
    "el": {
        "title": "Χάρτες Δεδομένων Ελλάδας (NUTS3)",
        "select_language": "Επιλέξτε Γλώσσα",
        "select_dataset": "Επιλέξτε Σύνολο Δεδομένων",
        "select_year": "Επιλέξτε Έτος",
        "select_sex": "Επιλέξτε Φύλο",
        "select_age": "Επιλέξτε Ομάδα Ηλικίας",
        "color_scale": "Κλίμακα Χρωμάτων",
        "default_data": "Δεδομένα Πληθυσμού NUTS",
        "generate_map": "Δημιουργία Διαγραμμάτων",
        "choropleth_map": "Χάρτης Χρωματικής Κλίμακας",
        "bar_chart": "Ραβδογράφημα",
        "region": "Περιοχή",
        "name": "Όνομα",
        "code": "Κωδικός",
        "value": "Τιμή",
        "apply": "Εφαρμογή"
    }
}

###############################################################################
# ROUTE
###############################################################################
@app.route("/", methods=["GET", "POST"])
def index():
    # Retrieve form values (or use defaults)
    language = request.form.get("language", "en")
    lang_trans = translations_dict.get(language, translations_dict["en"])
    selected_color_scale = request.form.get("color_scale", "Viridis")
    selected_dataset = request.form.get("dataset", "Population-NUTS DATA")
    form_type = request.form.get("form_type", "top")
    auto_scroll = (form_type == "top")
    
    # Determine the proper IMOP link and text based on language.
    if language == "el":
        imop_link = "https://www.dept.aueb.gr/el/imop"
        imop_text = "ΕΜΟΠ"
    else:
        imop_link = "https://www.dept.aueb.gr/en/imop"
        imop_text = "EMOP"
    
    # Build dropdown options from Excel data
    df_all = load_default_data()
    years_available = sorted(df_all["YEAR"].dropna().unique())
    sexes_available = sorted(df_all["SEX"].dropna().unique())
    ages_available = sorted(df_all["age"].dropna().unique())
    years_str = [str(y) for y in years_available]
    sexes_str = [str(s) for s in sexes_available]
    ages_str = [str(a) for a in ages_available]
    
    selected_year_str = request.form.get("selected_year", years_str[0] if years_str else "")
    selected_sex_str  = request.form.get("selected_sex", sexes_str[0] if sexes_str else "")
    selected_age_str  = request.form.get("selected_age", ages_str[0] if ages_str else "")
    
    selected_year = int(selected_year_str) if selected_year_str.isdigit() else (years_available[0] if years_available else 2023)
    selected_sex = selected_sex_str
    selected_age = selected_age_str
    
    chart_map_html = ""
    chart_bar_html = ""
    
    if request.method == "POST":
        chart_map_html = get_choropleth_html(selected_year, selected_sex, selected_age, selected_color_scale, language)
        chart_bar_html = get_bar_chart_html(selected_year, selected_sex, selected_age, selected_color_scale, language)
    
    # Modern template with responsive header and floating settings panel.
    template = """
    <!DOCTYPE html>
    <html lang="{{ language }}">
    <head>
      <meta charset="UTF-8">
      <title>{{ lang_trans['title'] }}</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <!-- Tailwind CSS -->
      <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
      <style>
        html { scroll-behavior: smooth; }
        .chart-container { width: 100%; min-height: 80vh; }
        .map-frame {
          border: 2px solid #6C2726;
          padding: 10px;
          box-sizing: border-box;
        }
        /* Floating Settings Panel */
        #settingsPanel {
          display: none;
          position: fixed;
          bottom: 70px;
          right: 20px;
          z-index: 50;
          background-color: #FFFFFF;
          border: 2px solid #6C2726;
          padding: 10px;
          border-radius: 0.5rem;
          width: 260px;
          box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
      </style>
    </head>
    <body class="bg-white text-gray-800">
      <!-- Responsive Header with AUEB Logo and Title -->
      <header class="flex items-center justify-center p-4 bg-[#6C2726]">
        <img src="https://www.aueb.gr/newopa/icons/menu/logo_opa.png" alt="AUEB Logo" class="h-8 md:h-12 mr-4">
        <h1 class="text-xl md:text-3xl font-bold text-white">
          {{ lang_trans['title'] }}
        </h1>
      </header>
      
      <!-- Original Top Menu -->
      <div class="max-w-xl mx-auto my-6 bg-white p-6 rounded shadow border" style="border-color: #6C2726;">
        <form method="post">
          <input type="hidden" name="form_type" value="top">
          <div class="grid grid-cols-1 gap-4">
            <!-- Language selection -->
            <div>
              <label class="block font-semibold" style="color: #6C2726;">{{ lang_trans['select_language'] }}</label>
              <select name="language" class="w-full p-2 rounded border" style="border-color: #6C2726; background-color: #FFFFFF; color: #000000;">
                <option value="en" {% if language == 'en' %}selected{% endif %}>English</option>
                <option value="el" {% if language == 'el' %}selected{% endif %}>Ελληνικά</option>
              </select>
            </div>
            <!-- Dataset selection -->
            <div>
              <label class="block font-semibold" style="color: #6C2726;">{{ lang_trans['select_dataset'] }}</label>
              <select name="dataset" class="w-full p-2 rounded border" style="border-color: #6C2726; background-color: #FFFFFF; color: #000000;">
                {% if language == 'el' %}
                <option value="Population-NUTS DATA" selected>Δεδομένα Πληθυσμού NUTS</option>
                {% else %}
                <option value="Population-NUTS DATA" selected>Population-NUTS DATA</option>
                {% endif %}
              </select>
            </div>
            <hr class="my-4 border" style="border-color: #6C2726;">
            <!-- File-specific menus -->
            <div>
              <label class="block font-semibold" style="color: #6C2726;">{{ lang_trans['select_year'] }}</label>
              <select name="selected_year" class="w-full p-2 rounded border" style="border-color: #6C2726; background-color: #FFFFFF; color: #000000;">
                {% for yr in years_str %}
                <option value="{{ yr }}" {% if yr == selected_year_str %}selected{% endif %}>{{ yr }}</option>
                {% endfor %}
              </select>
            </div>
            <div>
              <label class="block font-semibold" style="color: #6C2726;">{{ lang_trans['select_sex'] }}</label>
              <select name="selected_sex" class="w-full p-2 rounded border" style="border-color: #6C2726; background-color: #FFFFFF; color: #000000;">
                {% for sx in sexes_str %}
                <option value="{{ sx }}" {% if sx == selected_sex_str %}selected{% endif %}>{{ sx }}</option>
                {% endfor %}
              </select>
            </div>
            <div>
              <label class="block font-semibold" style="color: #6C2726;">{{ lang_trans['select_age'] }}</label>
              <select name="selected_age" class="w-full p-2 rounded border" style="border-color: #6C2726; background-color: #FFFFFF; color: #000000;">
                {% for ag in ages_str %}
                <option value="{{ ag }}" {% if ag == selected_age_str %}selected{% endif %}>{{ ag }}</option>
                {% endfor %}
              </select>
            </div>
            <div>
              <label class="block font-semibold" style="color: #6C2726;">{{ lang_trans['color_scale'] }}</label>
              <select name="color_scale" class="w-full p-2 rounded border" style="border-color: #6C2726; background-color: #FFFFFF; color: #000000;">
                {% for cs in color_scales %}
                <option value="{{ cs }}" {% if cs == selected_color_scale %}selected{% endif %}>{{ cs }}</option>
                {% endfor %}
              </select>
            </div>
            <div>
              <button type="submit" class="w-full py-2 rounded" style="background-color: #6C2726; color: #FFFFFF; font-weight: bold;">
                {{ lang_trans['generate_map'] }}
              </button>
            </div>
          </div>
        </form>
      </div>
      
      <!-- Main Content (Charts) -->
      <main class="p-8">
        {% if chart_map_html %}
        <div class="mt-8 chart-container">
          <h2 class="text-2xl font-bold mb-4 text-center" style="color: #6C2726;">{{ lang_trans['choropleth_map'] }}</h2>
          <div class="map-frame">{{ chart_map_html|safe }}</div>
        </div>
        {% endif %}
        {% if chart_bar_html %}
        <div class="mt-8 chart-container">
          <h2 class="text-2xl font-bold mb-4 text-center" style="color: #6C2726;">{{ lang_trans['bar_chart'] }}</h2>
          <div>{{ chart_bar_html|safe }}</div>
        </div>
        {% endif %}
      </main>
      
      <!-- Footer -->
      <footer class="p-4 text-center" style="color: #6C2726;">
        &copy; 2025 <a href="{{ imop_link }}" class="hover:underline" style="color: #6C2726;">{{ imop_text }}</a>. All rights reserved.
      </footer>
      
      <!-- Floating Settings Button -->
      <button id="settingsButton" class="fixed bottom-4 right-4 z-50 bg-red-600 text-white p-3 rounded-full shadow">
        <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0a1.72 1.72 0 001.29 1.16c.905.262 1.596 1.043 1.858 1.95a1.72 1.72 0 001.116 1.116c.907.262 1.688.953 1.95 1.858a1.72 1.72 0 001.16 1.29c.921.3.921 1.603 0 1.902a1.72 1.72 0 00-1.16 1.29c-.262.905-.953 1.688-1.95 1.95a1.72 1.72 0 00-1.116 1.116c-.262.907-.953 1.688-1.858 1.95a1.72 1.72 0 00-1.29 1.16z"></path>
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
        </svg>
      </button>
      
      <!-- Floating Settings Panel -->
      <div id="settingsPanel">
        <form id="settingsFormFloating" method="post">
          <input type="hidden" name="form_type" value="floating">
          <div class="mb-2">
            <label class="block text-red-600 font-semibold">{{ lang_trans['select_language'] }}</label>
            <select name="language" class="w-full p-2 border border-red-600 rounded bg-white text-black">
              <option value="en" {% if language == 'en' %}selected{% endif %}>English</option>
              <option value="el" {% if language == 'el' %}selected{% endif %}>Ελληνικά</option>
            </select>
          </div>
          <div class="mb-2">
            <label class="block text-red-600 font-semibold">{{ lang_trans['select_dataset'] }}</label>
            <select name="dataset" class="w-full p-2 border border-red-600 rounded bg-white text-black">
              {% if language == 'el' %}
              <option value="Population-NUTS DATA" selected>Δεδομένα Πληθυσμού NUTS</option>
              {% else %}
              <option value="Population-NUTS DATA" selected>Population-NUTS DATA</option>
              {% endif %}
            </select>
          </div>
          <div class="mb-2">
            <label class="block text-red-600 font-semibold">{{ lang_trans['select_year'] }}</label>
            <select name="selected_year" class="w-full p-2 border border-red-600 rounded bg-white text-black">
              {% for yr in years_str %}
              <option value="{{ yr }}" {% if yr == selected_year_str %}selected{% endif %}>{{ yr }}</option>
              {% endfor %}
            </select>
          </div>
          <div class="mb-2">
            <label class="block text-red-600 font-semibold">{{ lang_trans['select_sex'] }}</label>
            <select name="selected_sex" class="w-full p-2 border border-red-600 rounded bg-white text-black">
              {% for sx in sexes_str %}
              <option value="{{ sx }}" {% if sx == selected_sex_str %}selected{% endif %}>{{ sx }}</option>
              {% endfor %}
            </select>
          </div>
          <div class="mb-2">
            <label class="block text-red-600 font-semibold">{{ lang_trans['select_age'] }}</label>
            <select name="selected_age" class="w-full p-2 border border-red-600 rounded bg-white text-black">
              {% for ag in ages_str %}
              <option value="{{ ag }}" {% if ag == selected_age_str %}selected{% endif %}>{{ ag }}</option>
              {% endfor %}
            </select>
          </div>
          <div class="mb-2">
            <label class="block text-red-600 font-semibold">{{ lang_trans['color_scale'] }}</label>
            <select name="color_scale" class="w-full p-2 border border-red-600 rounded bg-white text-black">
              {% for cs in color_scales %}
              <option value="{{ cs }}" {% if cs == selected_color_scale %}selected{% endif %}>{{ cs }}</option>
              {% endfor %}
            </select>
          </div>
          <div>
            <button type="submit" class="w-full py-2 bg-red-600 text-white rounded">{{ lang_trans['apply'] }}</button>
          </div>
        </form>
      </div>
      
      <!-- JavaScript to handle scroll position and toggle settings panel -->
      <script>
        document.addEventListener("DOMContentLoaded", function() {
          var scrollPosition = sessionStorage.getItem("scrollPosition");
          if (scrollPosition) {
            window.scrollTo(0, parseInt(scrollPosition));
            sessionStorage.removeItem("scrollPosition");
          }
          var forms = document.getElementsByTagName("form");
          for (var i = 0; i < forms.length; i++) {
              forms[i].addEventListener("submit", function() {
                  sessionStorage.setItem("scrollPosition", window.scrollY);
              });
          }
          document.getElementById("settingsButton").addEventListener("click", function() {
              var panel = document.getElementById("settingsPanel");
              panel.style.display = (panel.style.display === "none" || panel.style.display === "") ? "block" : "none";
          });
          {% if auto_scroll and (chart_map_html or chart_bar_html) %}
          window.addEventListener("load", function(){
            var chartElem = document.querySelector(".chart-container");
            if(chartElem){
              chartElem.scrollIntoView({ behavior: 'smooth' });
            }
          });
          {% endif %}
        });
      </script>
    </body>
    </html>
    """

    return render_template_string(
        template,
        language=language,
        lang_trans=lang_trans,
        color_scales=color_scales,
        selected_color_scale=selected_color_scale,
        years_str=years_str,
        sexes_str=sexes_str,
        ages_str=ages_str,
        selected_year_str=selected_year_str,
        selected_sex_str=selected_sex_str,
        selected_age_str=selected_age_str,
        chart_map_html=chart_map_html,
        chart_bar_html=chart_bar_html,
        imop_link=imop_link,
        imop_text=imop_text,
        auto_scroll=auto_scroll
    )

if __name__ == "__main__":
    # For deployment, consider setting debug=False and host="0.0.0.0"
    app.run(debug=True, port=5000)
