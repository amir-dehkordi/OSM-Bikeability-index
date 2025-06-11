# Bikeability Analysis Project

This project provides a comprehensive analysis and interactive visualization of bikeability scores specifically for **15 Citi Bike stations located in proximity to New York City ferry terminals**. The primary goal is to assess environmental factors that contribute to a bike-friendly environment, thereby informing the attractiveness and feasibility of the "first/last mile" cycling segments for integrated multimodal journeys involving ferries.

## Features

-   **Comprehensive Bikeability Analysis**: Calculates a multi-faceted bikeability score based on various environmental attributes, specifically designed to evaluate areas around key multimodal interchange points:
    * **Bike lane density and quality**: Evaluates the presence and quality of dedicated cycling infrastructure (e.g., protected lanes, designated paths).
    * **Points of Interest (POIs) accessibility**: Measures proximity to diverse destinations, including commercial establishments, educational institutions, public transport hubs (subway, bus stations), and ferry terminals themselves.
    * **Network density and connectivity**: Assesses the overall density and interconnectedness of the transportation network within the station's vicinity.
    * **Green space availability**: Quantifies the presence of parks and recreational areas.
    * **Residential area density**: Measures the concentration of residential land use, indicating potential trip origins or destinations.
    * **Station capacity**: Considers the availability of docks at the bike stations.
    * **Intersection density**: Evaluates the navigability and granularity of the street network.

-   **Interactive Visualization**: Generates an interactive HTML map for intuitive exploration of the analysis results:
    * Color-coded station markers (red to green) indicating bikeability scores.
    * Detailed popups providing comprehensive metrics for each station upon click.
    * Multiple base map layer options (e.g., OpenStreetMap, CartoDB).
    * Fullscreen mode for enhanced viewing.
    * Mini map for easy navigation.


## Project Structure

-   `bikeability_analysis.py`: The core Python script responsible for querying OpenStreetMap (OSM) data, calculating all individual bikeability metrics, and computing the final weighted bikeability scores.
-   `visualize_bikeability.py`: A Python script that takes the calculated scores and generates the interactive HTML map.
-   `bikeability_map.html`: The output interactive map file, viewable in any web browser.
-   `bikeability_index.csv`: An output CSV file containing the calculated bikeability scores and all contributing metrics for each analyzed station.
-   `citibike_stations.csv`: The input CSV file containing geographical coordinates and capacity information for the selected Citi Bike stations.
## Requirements

The project requires the following Python packages:
- pandas
- osmnx
- numpy
- tqdm
- requests
- shapely
- geopandas
- folium
- branca


## Analysis Metrics
The final bikeability score is a weighted aggregation of the normalized factors listed above. Each component is individually normalized to a 0-100 scale, using empirically determined constants to define optimal thresholds. The combined weighted sum provides a comprehensive index, with higher weights typically assigned to factors such as dedicated bike infrastructure and accessibility to key destinations, reflecting their strong influence on cycling behavior.

## Contributing

Feel free to submit issues and enhancement requests!
 