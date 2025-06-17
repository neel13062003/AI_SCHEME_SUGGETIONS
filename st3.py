import streamlit as st
import pandas as pd
import re
from itertools import product
from openai import OpenAI
import json
import pymysql
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
# -------------------------------
# Configuration
# -------------------------------
st.set_page_config(page_title="Scheme Matcher", layout="wide")

# Use environment variable or Streamlit secrets for API key
# try:
#     openai.api_key = API
# except:
#     openai.api_key = API
#     if not openai.api_key:
#         st.error("⚠️ OpenAI API key not found. Please set it in Streamlit secrets or environment variables.")
#         st.stop()

try:
    client = OpenAI(
        api_key=st.secrets["OPEN_AI_KEY"],
        proxies=None
    )
except (KeyError, TypeError, ValueError) as e:
    st.error("⚠️ 1OpenAI API key not found or invalid. Please set it in Streamlit secrets or environment variables.")
    st.stop()


# DB_HOST_OLD = st.secrets["database"]["host"]
# DB_USER_OLD = st.secrets["database"]["user"]
# DB_PASSWORD_OLD = st.secrets["database"]["password"]
# DB_DATABASE_OLD = st.secrets["database"]["database"]

DB_HOST_OLD = st.secrets["HOST"]
DB_USER_OLD = st.secrets["USER"]
DB_PASSWORD_OLD = st.secrets["PASSWORD"]
DB_DATABASE_OLD = st.secrets["DATABASE"]

@st.cache_data(show_spinner=True, ttl=3600)  # Cache for 1 hour
def fetch_data():
    """Fetch data from database with error handling"""
    try:
        with st.spinner("Connecting to database..."):
            # Use SQLAlchemy engine to avoid pandas warning
            connection_string = f"mysql+pymysql://{DB_USER_OLD}:{DB_PASSWORD_OLD}@{DB_HOST_OLD}/{DB_DATABASE_OLD}"
            engine = create_engine(connection_string, connect_args={'connect_timeout': 10})
            
        with st.spinner("Fetching data from database..."):
            query = "SELECT * FROM grants"
            df = pd.read_sql(query, engine)
            
        st.success(f"✅ Successfully fetched {len(df)} records from database")
        return df
        
    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        return None

def safe_json_load(x):
    """Safely parse JSON strings with better error handling"""
    if pd.isna(x) or x is None:
        return []
    
    if not isinstance(x, str):
        return []
    
    x = x.strip()
    if not x or x in ['', 'null', 'NULL']:
        return []
    
    try:
        if x.startswith('[') and x.endswith(']'):
            return json.loads(x)
        elif x.startswith('{') and x.endswith('}'):
            return [json.loads(x)]  # Convert single object to list
        else:
            # Handle comma-separated values
            return [item.strip() for item in x.split(',') if item.strip()]
    except json.JSONDecodeError:
        # Fallback: treat as comma-separated string
        return [item.strip() for item in x.split(',') if item.strip()]
    except Exception:
        return []

# -------------------------------
# Standardization Dictionary
# -------------------------------
standardization_dict = {
    "Pvt. Ltd, LLP, Reg Partnership and Sole Proprietorship": 
        ["Private Limited", "LLP", "Registered Partnership", "Sole Proprietorship"],
    "Private Limited, Limited Company, LLP, and Registered Partnership": 
        ["Private Limited", "Limited Company", "LLP", "Registered Partnership"],
    "Private Limited, Limited Company, LLP, and Registered Partnership (STARTUP INDIA REGISTER)": 
        ["Private Limited", "Limited Company", "LLP", "Registered Partnership"],
    "NGO": ["NGO"],
    "Pvt. Ltd, LLP, Reg Partnership, Sole Proprietorship and NGO": 
        ["Private Limited", "LLP", "Registered Partnership", "Sole Proprietorship", "NGO"]
}

# -------------------------------
# Sector Mapping
# -------------------------------
category_keywords = {
    'Sector-Agnostic / All Sectors': ['agnostic', 'all sector', 'any of the focus sectors', 'various domains'],
    'General Manufacturing & Services': ['manufacturing', 'service', 'retail trade', 'trading'],
    'Social Impact': ['social impact', 'social challenges', 'economic empowerment', 'rural innovation', 'secure livelihoods', 'uplift farmer livelihoods'],
    'AI & ML': ['ai', 'ml', 'machine learning', 'deep learning', 'data science'],
    'IoT': ['iot', 'internet of things'],
    'DeepTech': ['deep-tech', 'deep tech', 'disruptive technologies'],
    'IT & Software': ['ict', 'it infrastructure'],
    'SaaS': ['saas'],
    'Cybersecurity': ['cyber security', 'cybersecurity'],
    'FinTech': ['fintech'],
    'EdTech': ['ed-tech', 'edtech'],
    'HR Tech': ['hr tech'],
    'InsureTech': ['insure tech'],
    'Enterprise Tech': ['enterprise tech'],
    'Hardware & Electronics': ['hardware', 'electronics', 'nanotechnology', 'semiconductors'],
    'AR/VR': ['ar/vr', 'augmented reality', 'virtual reality'],
    'Media, Animation & Gaming': ['media', 'animation', 'gaming', 'entertainment', 'comics'],
    'Digital Commerce': ['digital commerce'],
    'Product Design': ['product design'],
    'AgriTech': ['agritech', 'agri-tech', 'agriculture', 'agri science', 'farming', 'horticulture', 'agroecological'],
    'FoodTech': ['food-tech', 'foodtech', 'food processing', 'post harvest'],
    'Animal Husbandry': ['animal husbandry', 'veterinary', 'dairy', 'animal feed'],
    'Fisheries & Aquaculture': ['fisheries', 'aquaculture'],
    'CleanTech / ClimateTech': ['cleantech', 'clean-teach', 'climate tech', 'climatetech', 'greenhouse gas', 'decarbonization', 'climate action', 'sustainability', 'sustainable'],
    'Clean & Sustainable Energy': ['energy', 'green energy', 'sustainable energy', 'alternative fuels'],
    'Circular Economy & Waste Management': ['waste management', 'circular economy', 'waste to wealth', 'waste to energy', 'circulularity'],
    'Environmental Protection': ['environment', 'pollution', 'water security', 'biodiversity', 'soil health', 'natural resource'],
    'HealthTech / MedTech': ['health-tech', 'healthtech', 'medtech', 'medical devices'],
    'Healthcare & Life Sciences': ['healthcare', 'health', 'life sciences', 'pharma'],
    'Biotechnology': ['biotechnology', 'biotech'],
    'Automotive': ['automotive', 'automobiles'],
    'Smart & Future Mobility': ['mobility', 'transportation', 'ev', 'electric vehicle'],
    'Defense & Aerospace': ['defense', 'aerospace'],
    'Education & Skilling': ['education', 'skill development', 'skilling'],
    'Women-Focused': ['women-led', 'women focused', 'women-focused'],
    'Infrastructure': ['infrastructure', 'smart infrastructure'],
    'Hospitality & Travel': ['hospitality & travel'],
    'Toys': ['toys'],
    'Crafts': ['crafts'],
}

def standardize_company_types(company_types_list):
    """Standardize company types based on the standardization dictionary"""
    if not isinstance(company_types_list, list):
        return []
    
    result = []
    flat_text = " ".join(str(item) for item in company_types_list).lower()
    
    for key, values in standardization_dict.items():
        if any(keyword.lower() in flat_text for keyword in key.split(', ')):
            result.extend(values)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(result)) if result else company_types_list

def map_sectors(sector_list):
    """Map sectors to standardized categories"""
    if not isinstance(sector_list, list):
        return ["Sector-Agnostic / All Sectors"]
    
    found = set()
    flat_text = " ".join(str(item) for item in sector_list).lower()
    
    for category, keywords in category_keywords.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', flat_text):
                found.add(category)
                break
    
    return ['Sector-Agnostic / All Sectors'] if 'Sector-Agnostic / All Sectors' in found else (sorted(found) if found else ["Sector-Agnostic / All Sectors"])

def process_scheme_types(scheme_type):
    """Process scheme types"""
    if pd.isna(scheme_type) or scheme_type is None:
        return []
    return scheme_type.split('-') if isinstance(scheme_type, str) else []

@st.cache_data(show_spinner=True)
def process_data(data):
    """Process the fetched data"""
    if data is None or data.empty:
        return None
    
    with st.spinner("Processing data..."):
        # Process JSON columns
        json_columns = ['benefits', 'focusSectors', 'eligibilityCriteria', 'sectors', 'companyTypes']
        
        for col in json_columns:
            if col in data.columns:
                data[col] = data[col].apply(safe_json_load)
        
        # Standardize company types
        data['TYPE_STANDARDIZED'] = data['companyTypes'].apply(standardize_company_types)
        
        # Map sectors
        data['focus_sector_standardized'] = data['focusSectors'].apply(map_sectors)
        
        # Process scheme types
        data['type_of_scheme_final'] = data['schemeType'].apply(process_scheme_types)
        
        # Debug: Let's see what we have before filtering
        st.write("Debug Info:")
        st.write(f"Company types sample: {data['TYPE_STANDARDIZED'].head().tolist()}")
        st.write(f"Sector types sample: {data['focus_sector_standardized'].head().tolist()}")
        st.write(f"Scheme types sample: {data['type_of_scheme_final'].head().tolist()}")
        
        # Check for empty fields and provide defaults
        data['TYPE_STANDARDIZED'] = data['TYPE_STANDARDIZED'].apply(
            lambda x: x if isinstance(x, list) and len(x) > 0 else ['Unknown']
        )
        data['focus_sector_standardized'] = data['focus_sector_standardized'].apply(
            lambda x: x if isinstance(x, list) and len(x) > 0 else ['Sector-Agnostic / All Sectors']
        )
        data['type_of_scheme_final'] = data['type_of_scheme_final'].apply(
            lambda x: x if isinstance(x, list) and len(x) > 0 else ['General']
        )
        
        return data

def explode_combinations(row):
    """Create combinations of company types, sectors, and scheme types"""
    try:
        # Ensure all fields are lists and not empty
        company_types = row['TYPE_STANDARDIZED'] if isinstance(row['TYPE_STANDARDIZED'], list) and len(row['TYPE_STANDARDIZED']) > 0 else ['Unknown']
        sector_types = row['focus_sector_standardized'] if isinstance(row['focus_sector_standardized'], list) and len(row['focus_sector_standardized']) > 0 else ['Unknown']
        scheme_types = row['type_of_scheme_final'] if isinstance(row['type_of_scheme_final'], list) and len(row['type_of_scheme_final']) > 0 else ['Unknown']
        
        # Create combinations
        combos = list(product(company_types, sector_types, scheme_types))
        
        if not combos:
            return pd.DataFrame()
        
        # Create dataframe with combinations
        result_data = []
        for combo in combos:
            row_dict = {}
            row_dict['TYPE_STANDARDIZED'] = combo[0]
            row_dict['focus_sector_standardized'] = combo[1]
            row_dict['type_of_scheme_final'] = combo[2]
            
            # Add other columns from original row
            for col in row.index:
                if col not in ['TYPE_STANDARDIZED', 'focus_sector_standardized', 'type_of_scheme_final']:
                    row_dict[col] = row[col]
            
            result_data.append(row_dict)
        
        return pd.DataFrame(result_data)
        
    except Exception as e:
        print(f"Error processing row with scheme name '{row.get('schemeName', 'Unknown')}': {e}")
        # Return a single row with the original data but default values for problematic fields
        fallback_row = row.to_dict()
        fallback_row['TYPE_STANDARDIZED'] = 'Unknown'
        fallback_row['focus_sector_standardized'] = 'Unknown'
        fallback_row['type_of_scheme_final'] = 'Unknown'
        return pd.DataFrame([fallback_row])

# -------------------------------
# Main App
# -------------------------------
st.title("🧠 Scheme Query Analyzer (Database Version)")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.processed_data = None
    st.session_state.final_exploded_df = None
    st.session_state.show_all_columns = False
    st.session_state.filtered_results = None

# Data loading section
if not st.session_state.data_loaded:
    if st.button("🔄 Load Data from Database"):
        # Fetch data
        raw_data = fetch_data()
        
        if raw_data is not None:
            # Process data
            processed_data = process_data(raw_data)
            
            if processed_data is not None and not processed_data.empty:
                with st.spinner("Creating final dataset..."):
                    # Create exploded combinations
                    exploded_dfs = []
                    failed_rows = 0
                    
                    progress_bar = st.progress(0)
                    total_rows = len(processed_data)
                    
                    for idx, row in processed_data.iterrows():
                        progress_bar.progress((idx + 1) / total_rows)
                        
                        combo_df = explode_combinations(row)
                        if not combo_df.empty:
                            exploded_dfs.append(combo_df)
                        else:
                            failed_rows += 1
                    
                    progress_bar.empty()
                    
                    if exploded_dfs:
                        final_exploded_df = pd.concat(exploded_dfs, ignore_index=True)
                        
                        # Store in session state
                        st.session_state.processed_data = processed_data
                        st.session_state.final_exploded_df = final_exploded_df
                        st.session_state.data_loaded = True
                        
                        success_msg = f"✅ Data processed successfully! Final dataset has {len(final_exploded_df)} combinations from {len(processed_data)} original schemes."
                        if failed_rows > 0:
                            success_msg += f" ({failed_rows} rows had processing issues but were handled)"
                        st.success(success_msg)
                    else:
                        st.error("❌ No valid combinations could be created from the processed data.")
            else:
                st.error("❌ No valid data after processing.")
else:
    st.success("✅ Data already loaded in session")

# Display data if loaded
if st.session_state.data_loaded and st.session_state.final_exploded_df is not None:
    
    # Show summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Schemes", len(st.session_state.processed_data))
    with col2:
        st.metric("Total Combinations", len(st.session_state.final_exploded_df))
    with col3:
        unique_schemes = st.session_state.final_exploded_df['schemeName'].nunique()
        st.metric("Unique Scheme Names", unique_schemes)
    
    # Show data preview
    st.subheader("📊 Data Preview")
    display_columns = ['schemeName', 'TYPE_STANDARDIZED', 'focus_sector_standardized', 'type_of_scheme_final']
    available_columns = [col for col in display_columns if col in st.session_state.final_exploded_df.columns]
    
    if available_columns:
        st.dataframe(
            st.session_state.final_exploded_df[available_columns].head(100),
            use_container_width=True
        )
    else:
        st.dataframe(st.session_state.final_exploded_df.head(100), use_container_width=True)
    
    # -------------------------------
    # Query Section
    # -------------------------------
    st.subheader("🔍 Extract Scheme Filters from Query")
    user_query = st.text_area("Enter your scheme-related query:", height=100)
    
#     if st.button("🔍 Extract Filters") and user_query.strip():
#         try:
#             # Get unique values for prompting
#             company_types = sorted(st.session_state.final_exploded_df['TYPE_STANDARDIZED'].unique().tolist())
#             sector_types = sorted(st.session_state.final_exploded_df['focus_sector_standardized'].unique().tolist())
#             scheme_types = sorted(st.session_state.final_exploded_df['type_of_scheme_final'].unique().tolist())
            
#             prompt = f"""
# You are an intelligent assistant that extracts scheme filters from user queries.

# Extract the following three fields from the user's query:
# 1. company_type
# 2. sector_type  
# 3. scheme_type

# You MUST map values from the user's query to ONLY the predefined categories below. If nothing matches, return an empty list for that field.

# Available company_type options: {company_types}
# Available sector_type options: {sector_types}
# Available scheme_type options: {scheme_types}

# Return ONLY a valid JSON object in this exact format:
# {{
#   "company_type": [],
#   "sector_type": [],
#   "scheme_type": []
# }}

# User Query: "{user_query}"
#             """.strip()
            
#             with st.spinner("Processing query with AI..."):
#                 response = openai.chat.completions.create(
#                     model="gpt-4",
#                     messages=[
#                         {"role": "system", "content": "You are a helpful assistant that extracts structured data from text. Always return valid JSON."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     temperature=0.1,
#                     max_tokens=500
#                 )
                
#                 # Parse response
#                 response_text = response.choices[0].message.content.strip()
                
#                 # Clean up response if it has markdown formatting
#                 if response_text.startswith("```json"):
#                     response_text = response_text.replace("```json", "").replace("```", "").strip()
                
#                 extracted_filters = json.loads(response_text)
                
#                 st.subheader("📌 Extracted Filters")
#                 st.json(extracted_filters)
                
#                 # -------------------------------
#                 # Filter DataFrame
#                 # -------------------------------
#                 filtered_df = st.session_state.final_exploded_df.copy()
                
#                 # Apply filters
#                 if extracted_filters.get("company_type"):
#                     filtered_df = filtered_df[filtered_df["TYPE_STANDARDIZED"].isin(extracted_filters["company_type"])]
                
#                 if extracted_filters.get("sector_type"):
#                     filtered_df = filtered_df[filtered_df["focus_sector_standardized"].isin(extracted_filters["sector_type"])]
                
#                 if extracted_filters.get("scheme_type"):
#                     filtered_df = filtered_df[filtered_df["type_of_scheme_final"].isin(extracted_filters["scheme_type"])]
                
#                 # Store filtered results in session state
#                 st.session_state.filtered_results = filtered_df
                
#                 st.subheader("📊 Filtered Results")
                
#                 if not filtered_df.empty:
#                     st.success(f"Found {len(filtered_df)} matching schemes")
                    
#                     # Show key columns first
#                     key_columns = ['schemeName', 'TYPE_STANDARDIZED', 'focus_sector_standardized', 'type_of_scheme_final']
#                     available_key_columns = [col for col in key_columns if col in filtered_df.columns]
                    
#                     if available_key_columns:
#                         st.dataframe(filtered_df[available_key_columns], use_container_width=True)
                    
#                     # Option to show all columns - FIXED VERSION
#                     show_all = st.checkbox("Show all columns", key="filtered_show_all_columns")
#                     if show_all:
#                         st.subheader("📋 Complete Filtered Dataset")
#                         st.dataframe(filtered_df, use_container_width=True)
                        
#                     # Download option
#                     csv = filtered_df.to_csv(index=False)
#                     st.download_button(
#                         label="📥 Download filtered results as CSV",
#                         data=csv,
#                         file_name="filtered_schemes.csv",
#                         mime="text/csv"
#                     )
#                 else:
#                     st.warning("No schemes match the extracted filters.")
#                     st.session_state.filtered_results = None
                    
#         except json.JSONDecodeError as e:
#             st.error(f"❌ Error parsing AI response: {e}")
#             st.text("Raw AI Response:")
#             st.code(response.choices[0].message.content)
#         except Exception as e:
#             st.error(f"❌ Error processing query: {e}")
    
    if st.button("🔍 Extract Filters") and user_query.strip():
        try:
            # Get unique values for prompting
            company_types = sorted(st.session_state.final_exploded_df['TYPE_STANDARDIZED'].unique().tolist())
            sector_types = sorted(st.session_state.final_exploded_df['focus_sector_standardized'].unique().tolist())
            scheme_types = sorted(st.session_state.final_exploded_df['type_of_scheme_final'].unique().tolist())
            
            prompt = f"""
You are an intelligent assistant that extracts scheme filters from user queries.

Extract the following three fields from the user's query:
1. company_type
2. sector_type  
3. scheme_type

You MUST map values from the user's query to ONLY the predefined categories below. If nothing matches, return an empty list for that field.

Available company_type options: {company_types}
Available sector_type options: {sector_types}
Available scheme_type options: {scheme_types}

Return ONLY a valid JSON object in this exact format:
{{
  "company_type": [],
  "sector_type": [],
  "scheme_type": []
}}

User Query: "{user_query}"
        """.strip()
        
            with st.spinner("Processing query with AI..."):
                # Initialize OpenAI client explicitly with proxies=None for version 1.50.2
                
                client = OpenAI(
                    api_key=st.secrets["api_keys"]["openai"],
                    proxies=None  # Explicitly disable proxies to avoid the error
                )

                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that extracts structured data from text. Always  return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )

                # Parse response
                response_text = response.choices[0].message.content.strip()

                # Clean up response if it has markdown formatting
                if response_text.startswith("```json"):
                    response_text = response_text.replace("```json", "").replace("```", "").strip()

                extracted_filters = json.loads(response_text)

                st.subheader("📌 Extracted Filters")
                st.json(extracted_filters)

                # -------------------------------
                # Filter DataFrame
                # -------------------------------
                filtered_df = st.session_state.final_exploded_df.copy()

                # Apply filters
                if extracted_filters.get("company_type"):
                    filtered_df = filtered_df[filtered_df["TYPE_STANDARDIZED"].isin(extracted_filters["company_type"])]

                if extracted_filters.get("sector_type"):
                    filtered_df = filtered_df[filtered_df["focus_sector_standardized"].isin(extracted_filters["sector_type"])]

                if extracted_filters.get("scheme_type"):
                    filtered_df = filtered_df[filtered_df["type_of_scheme_final"].isin(extracted_filters["scheme_type"])]

                # Store filtered results in session state
                st.session_state.filtered_results = filtered_df

                st.subheader("📊 Filtered Results")

                if not filtered_df.empty:
                    st.success(f"Found {len(filtered_df)} matching schemes")

                    # Show key columns first
                    key_columns = ['schemeName', 'TYPE_STANDARDIZED', 'focus_sector_standardized', 'type_of_scheme_final']
                    available_key_columns = [col for col in key_columns if col in filtered_df.columns]

                    if available_key_columns:
                        st.dataframe(filtered_df[available_key_columns], use_container_width=True)

                    # Option to show all columns - FIXED VERSION
                    show_all = st.checkbox("Show all columns", key="filtered_show_all_columns")
                    if show_all:
                        st.subheader("📋 Complete Filtered Dataset")
                        st.dataframe(filtered_df, use_container_width=True)

                    # Download option
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download filtered results as CSV",
                        data=csv,
                        file_name="filtered_schemes.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No schemes match the extracted filters.")
                    st.session_state.filtered_results = None
                
        except json.JSONDecodeError as e:
            st.error(f"❌ Error parsing AI response: {e}")
            st.text("Raw AI Response:")
            st.code(response_text)  # Use response_text for consistency
        except Exception as e:
            st.error(f"❌ Error processing query: {e}")
    # Show previously filtered results if they exist
    elif st.session_state.get('filtered_results') is not None and not st.session_state.filtered_results.empty:
        st.subheader("📊 Previous Filtered Results")
        filtered_df = st.session_state.filtered_results
        
        st.success(f"Found {len(filtered_df)} matching schemes")
        
        # Show key columns first
        key_columns = ['schemeName', 'TYPE_STANDARDIZED', 'focus_sector_standardized', 'type_of_scheme_final']
        available_key_columns = [col for col in key_columns if col in filtered_df.columns]
        
        if available_key_columns:
            st.dataframe(filtered_df[available_key_columns], use_container_width=True)
        
        # Option to show all columns - FIXED VERSION
        show_all = st.checkbox("Show all columns", key="persistent_show_all_columns")
        if show_all:
            st.subheader("📋 Complete Filtered Dataset")
            st.dataframe(filtered_df, use_container_width=True)
            
        # Download option
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered results as CSV",
            data=csv,
            file_name="filtered_schemes.csv",
            mime="text/csv"
        )

# Reset button
if st.session_state.data_loaded:
    if st.button("🔄 Reset and Reload Data"):
        # Clear all session state variables
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()