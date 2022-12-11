import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from shapely.geometry import Point, Polygon
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

def distance(lat1: float, lon1: float, lat2: list, lon2: list):   
    R = 6371000  # Radius of Earth in meters
    phi_1 = np.radians(lat1) # CHANGE THIS
    phi_2 = np.radians(lat2) # CHANGE THIS

    delta_phi = np.radians(lat2 - lat1) # CHANGE THIS
    delta_lambda = np.radians(lon2 - lon1) # CHANGE THIS

    a = (
        np.sin(delta_phi / 2.0) ** 2 # CHANGE THIS
        + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2.0) ** 2 # CHANGE THIS (3x)
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) # CHANGE THIS (3x)

    meters = R * c  # Output distance in meters

    return np.round(meters, 0)


# Read dataframe
dataframe = pd.read_csv(
     "WK1_Airbnb_Amsterdam_listings_proj_solution.csv",
      names=[
          "Airbnb Listing ID",
          "Price",
          "Latitude",
          "Longitude",
          "Meters from chosen location",
          "Location",
      ],
  )

dataframe.drop(index=dataframe.index[0], axis=0, inplace=True)

with st.sidebar:
  st.title('Airbnb Amsterdam')
  st.header('Enter Address and Select Price Range')
  street = st.text_input("Street", "Van Gogh Museum")
  city = st.text_input("City", "Amsterdam")
  province = st.text_input("Province", "North Holland")
  country = st.text_input("Country", "Netherlands")

  geolocator = Nominatim(user_agent="GTA Lookup")
  geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
  location = geolocator.geocode(street+", "+city+", "+province+", "+country)

  lat = location.latitude
  lon = location.longitude
  dataframe['Meters from chosen location'] = distance(lat, lon, dataframe['Latitude'], dataframe['Longitude'])
  min = float(dataframe["Price"].round(2).min())
  max = float(dataframe["Price"].round(2).max())
  price_max = st.slider('Price Range:', min, max)
  #slider_range = st.slider('Price Range:', value=[min, max])
  search = st.button('Search')

if search:
  # Display title and text
  st.title("Week 1 - Data and visualization")
  st.markdown("Here we can see the dataframe created during this weeks project.")

  #st.write('old df len:', dataframe.shape) 
  data = []

  # always inserting new rows at the first position - last row will be always on top    
  data.insert(0, {'Airbnb Listing ID': 1, 'Price': 0, 'Latitude': lat, 'Longitude': lon, 'Meters from chosen location': 0, 'Location':1})

  dataframe = pd.concat([pd.DataFrame(data), dataframe], ignore_index=True)

  #st.write('new df len:', dataframe.shape)

  # We have a limited budget, therefore we would like to exclude
  # listings with a price per night in pounds as per user selection
  #dataframe = dataframe[(dataframe["Price"] >= slider_range[0]) & (dataframe["Price"] <= slider_range[1])]
  dataframe = dataframe[(dataframe["Price"] <= price_max)] 

  # Display as integer
  dataframe["Airbnb Listing ID"] = dataframe["Airbnb Listing ID"].astype(int)
  # Round of values
  dataframe["Price"] = "£" + dataframe["Price"].round(2).astype(str) 
  # Rename the number to a string
  dataframe["Location"] = dataframe["Location"].replace(
      {1.0: "To visit", 0.0: "Airbnb listing"}
  )

  # Display dataframe and text
  st.dataframe(dataframe)
  #st.write('location Lattitudw:', lat, 'longitude:', lon)
  #st.write('min_price:', slider_range[0], slider_range[1])
  st.markdown("Below is a map showing all the Airbnb listings with a red dot and the location we've chosen with a blue dot.")

  # Create the plotly express figure
  fig = px.scatter_mapbox(
      dataframe,
      lat="Latitude",
      lon="Longitude",
      color="Location",
      zoom=11,
      height=500,
      width=800,
      hover_name="Price",
      hover_data=["Meters from chosen location", "Location"],
      labels={"color": "Locations"},
  )
  fig.update_geos(center=dict(lat=dataframe.iloc[0][2], lon=dataframe.iloc[0][3]))
  fig.update_layout(mapbox_style="stamen-terrain")

  # Show the figure
  st.plotly_chart(fig, use_container_width=True)