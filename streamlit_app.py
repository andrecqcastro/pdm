import streamlit as st
import pandas as pd

###############################################################################################
############################### Temp Streamlit for testing ####################################
###############################################################################################

# Load data
# Replace 'data.csv' with the actual file name of your dataset
data = pd.read_csv('./artificial_data.csv')

# Set up the Streamlit layout
st.title("Amazon Product Reviews Dashboard")

# Display raw data
st.subheader("Raw Data")
st.dataframe(data)

# Filter by marketplace
marketplace_filter = st.selectbox(
    "Select Marketplace",
    options=data['marketplace'].unique()
)
filtered_data = data[data['marketplace'] == marketplace_filter]

# Display filtered data
st.subheader(f"Filtered Data for {marketplace_filter}")
st.dataframe(filtered_data)

# Star rating analysis
st.subheader("Star Rating Analysis")
rating_counts = filtered_data['star_rating'].value_counts().sort_index()
st.bar_chart(rating_counts)

# Most helpful reviews
st.subheader("Most Helpful Reviews")
helpful_reviews = filtered_data.sort_values(by='helpful_votes', ascending=False).head(10)
st.dataframe(helpful_reviews)

# Filter by product
product_filter = st.selectbox(
    "Select Product",
    options=filtered_data['product_title'].unique()
)
product_data = filtered_data[filtered_data['product_title'] == product_filter]

# Display product reviews
st.subheader(f"Reviews for {product_filter}")
st.dataframe(product_data)

# Votes analysis
st.subheader("Votes Analysis")
st.bar_chart(product_data[['helpful_votes', 'total_votes']].sum())

