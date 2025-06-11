import pandas as pd
import osmnx as ox
import numpy as np
from tqdm import tqdm
import requests # Used implicitly by osmnx
from shapely.geometry import Point # Used implicitly by geopandas/osmnx
import geopandas as gpd # Used by osmnx for GeoDataFrames
import traceback
import logging
import os
import math

# --- 1. Set up Logging ---
# Configure logging to output to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# --- 2. Define Constants for Normalization (ADJUST THESE VALUES!) ---
# These values determine what score corresponds to 100 for each metric.
# Tune them based on observations from your data and what you consider 'optimal'.

# POI Score: Max observed weighted score for a 100% POI score.
# Adjust this based on your observed 'weighted_score' from POI analysis.
MAX_EXPECTED_POI_WEIGHTED_SCORE = 400

# Capacity Score: The number of bike docks considered ideal for a 100% capacity score.
# Example: If 50 docks is considered 'perfect'.
OPTIMAL_CAPACITY_THRESHOLD = 50

# Network Density Score: Network length (miles/sq mile) at which score is 100.
# Adjust this based on your observed 'network_density' values.
# If you still get many 100s, increase this value significantly (e.g., 120, 150, or more).
OPTIMAL_NETWORK_DENSITY = 140

# --- 3. Define Core Functions ---

def get_nearby_pois(lat, lon, radius=500):
    """
    Get count of nearby points of interest with weighted scoring within a given radius.
    """
    poi_categories = {
        'amenity': ['restaurant', 'cafe', 'school', 'university', 'hospital',
                    'supermarket', 'pharmacy', 'bank', 'post_office',
                    'library', 'community_centre', 'arts_centre', 'theatre',
                    'cinema', 'place_of_worship', 'fountain', 'parking'],
        'shop': ['convenience', 'supermarket', 'bakery', 'butcher', 'clothes',
                 'department_store', 'bicycle_shop'],
        'leisure': ['park', 'fitness_centre', 'sports_centre', 'playground',
                    'garden', 'recreation_ground', 'stadium', 'track'],
        'public_transport': ['bus_stop', 'subway_entrance', 'train_station'],
        'tourism': ['attraction', 'museum', 'hotel', 'guest_house'],
        'highway': ['cycleway', 'service'],
        'parking': ['bicycle_parking']
    }

    poi_weights = {
        'restaurant': 1.0, 'cafe': 1.0, 'school': 1.5, 'university': 1.5,
        'hospital': 1.2, 'supermarket': 1.3, 'pharmacy': 1.1, 'bank': 0.8,
        'post_office': 0.8, 'convenience': 0.9, 'bakery': 0.9, 'butcher': 0.9,
        'clothes': 0.8, 'department_store': 1.2, 'park': 1.4, 'fitness_centre': 1.2,
        'sports_centre': 1.2, 'playground': 1.3, 'library': 1.0,
        'community_centre': 1.0, 'arts_centre': 0.9, 'theatre': 0.7, 'cinema': 0.7,
        'place_of_worship': 0.5, 'fountain': 0.3, 'bicycle_shop': 2.0,
        'bus_stop': 1.0, 'subway_entrance': 1.5, 'train_station': 1.5,
        'attraction': 0.8, 'museum': 0.8, 'hotel': 0.6, 'guest_house': 0.6,
        'bicycle_parking': 1.8, 'cycleway': 0.5, 'service': 0.5,
    }

    try:
        query_tags = {
            'amenity': True, 'shop': True, 'leisure': True,
            'public_transport': True, 'tourism': True,
            'highway': ['cycleway', 'service'],
            'parking': ['bicycle_parking']
        }

        pois_gdf = ox.features_from_point((lat, lon), tags=query_tags, dist=radius)

        if pois_gdf.empty:
            logging.info(f"   No POIs found for {lat}, {lon} within {radius}m.")
            return {'counts': {}, 'weighted_score': 0, 'normalized_score': 0}

        poi_counts = {}
        weighted_score = 0

        for _, row in pois_gdf.iterrows():
            found_tag_for_scoring = False
            for category, subcategories in poi_categories.items():
                if category in row and pd.notna(row[category]):
                    tag_values = row[category] if isinstance(row[category], list) else [str(row[category])]
                    for value in tag_values:
                        value = value.strip()
                        if value in subcategories:
                            poi_counts[f'{category}_{value}'] = poi_counts.get(f'{category}_{value}', 0) + 1
                            weighted_score += poi_weights.get(value, 1.0)
                            found_tag_for_scoring = True
                            break
                if found_tag_for_scoring:
                    break

            if 'bicycle_parking' in row and pd.notna(row['bicycle_parking']) and row['bicycle_parking'] == 'yes':
                poi_counts['parking_bicycle_parking'] = poi_counts.get('parking_bicycle_parking', 0) + 1
                weighted_score += poi_weights.get('bicycle_parking', 1.0)
            elif 'amenity' in row and pd.notna(row['amenity']) and ('bicycle_parking' in (row['amenity'] if isinstance(row['amenity'], list) else [str(row['amenity'])])):
                poi_counts['amenity_bicycle_parking'] = poi_counts.get('amenity_bicycle_parking', 0) + 1
                weighted_score += poi_weights.get('bicycle_parking', 1.0)

        logging.info(f"   Weighted POI score (raw) for {lat}, {lon}: {weighted_score:.2f}")
        normalized_score = min(100, (weighted_score / MAX_EXPECTED_POI_WEIGHTED_SCORE) * 100)

        return {
            'counts': poi_counts,
            'weighted_score': weighted_score,
            'normalized_score': normalized_score
        }
    except Exception as e:
        logging.error(f"Error getting POIs for {lat}, {lon}: {e}")
        return {'counts': {}, 'weighted_score': 0, 'normalized_score': 0}

def get_intersection_density(lat, lon, radius=500):
    """
    Calculate intersection density within a given radius.
    """
    try:
        # network_type='all' includes walking, cycling, driving networks
        G = ox.graph_from_point((lat, lon), dist=radius, network_type='all')
        if G is None:
            logging.info(f"   No graph found for intersection density at {lat}, {lon} within {radius}m.")
            return 0

        # Intersections are typically nodes with degree > 2 (where more than two ways meet)
        intersections = [
            node for node, degree in G.degree()
            if degree > 2
        ]

        area = np.pi * (radius)**2 # Area in square meters
        density = len(intersections) / area if area > 0 else 0
        logging.info(f"   Intersection density for {lat}, {lon}: {density:.4f} intersections/m^2")
        return density
    except Exception as e:
        logging.error(f"Error calculating intersection density for {lat}, {lon}: {e}")
        return 0

def get_multimodal_network_density(lat, lon, radius=500):
    """
    Calculate multimodal network density (drive, bike, walk) within a given radius.
    """
    try:
        # Retrieve graphs for different network types
        G_drive = ox.graph_from_point((lat, lon), dist=radius, network_type='drive')
        G_bike = ox.graph_from_point((lat, lon), dist=radius, network_type='bike')
        G_walk = ox.graph_from_point((lat, lon), dist=radius, network_type='walk')

        total_length = 0 # Length in meters
        for G in [G_drive, G_bike, G_walk]:
            if G is not None:
                # Sum the length of all edges in the graph
                total_length += sum(d.get('length', 0) for u, v, d in G.edges(data=True))

        total_length_miles = total_length / 1609.34 # Convert meters to miles
        area_sq_miles = (np.pi * (radius/1609.34)**2) # Area in square miles

        density = total_length_miles / area_sq_miles if area_sq_miles > 0 else 0
        logging.info(f"   Multimodal network density for {lat}, {lon}: {density:.4f} miles/sq_mile")
        return density
    except Exception as e:
        logging.error(f"Error calculating multimodal network density for {lat}, {lon}: {e}")
        return 0

def get_crime_score(lat, lon, radius=500):
    """
    Placeholder function for crime score.
    In a real application, this would query a crime data API or local database.
    """
    return 0

def get_bike_lane_density(lat, lon, radius=500):
    """
    Calculate enhanced bike lane density within a given radius, considering infrastructure quality.
    """
    try:
        G = ox.graph_from_point((lat, lon), dist=radius, network_type='bike')

        if G is None:
             logging.info(f"   No bike graph found for {lat}, {lon} within {radius}m.")
             return {
                'density': 0, 'weighted_density': 0, 'total_length': 0,
                'weighted_length': 0, 'infrastructure_metrics': {}
            }

        # Weights for different bike infrastructure types and conditions
        bike_infrastructure_scores = {
            'cycleway': 1.0, 'highway=cycleway': 1.0, 'cycleway=track': 1.0, 'cycleway=lane': 0.8,
            'cycleway=shared_busway': 0.7, 'bicycle=designated': 0.9, 'bicycle=official': 0.9,
            'highway=path': 0.8, 'highway=track': 0.7, 'bicycle=permissive': 0.6,
            'bicycle=private': 0.3, 'bicycle=dismount': 0.1, 'bicycle=use_sidepath': 0.8,
            'bicycle=no': 0.0, # Negative impact
            'surface=asphalt': 0.1, 'surface=paved': 0.1, 'surface=unpaved': -0.1, 'surface=dirt': -0.2,
            'smoothness=excellent': 0.1, 'smoothness=good': 0.05, 'smoothness=intermediate': 0.0,
            'smoothness=bad': -0.1, 'smoothness=horrible': -0.2,
            'highway=primary': -0.2, 'highway=secondary': -0.1, 'highway=tertiary': 0.05,
            'highway=residential': 0.1, 'highway=service': 0.05,
        }

        infrastructure_metrics = {}
        total_length = 0 # Total length of all bikeable ways in meters
        weighted_length = 0 # Weighted length based on quality in meters

        for u, v, k, data in G.edges(keys=True, data=True):
            length = data.get('length', 0)
            total_length += length
            segment_quality_score = 0
            found_bike_tag_for_score = False

            # Check for primary bike-specific tags
            if 'cycleway' in data:
                cw_tag_value = data['cycleway']
                if cw_tag_value in bike_infrastructure_scores:
                    segment_quality_score += bike_infrastructure_scores[cw_tag_value]
                    found_bike_tag_for_score = True
                elif f'cycleway={cw_tag_value}' in bike_infrastructure_scores: # Handles 'cycleway=track' format
                    segment_quality_score += bike_infrastructure_scores[f'cycleway={cw_tag_value}']
                    found_bike_tag_for_score = True
            if 'bicycle' in data: # E.g., bicycle=designated
                b_tag_value = data['bicycle']
                if b_tag_value in bike_infrastructure_scores:
                    segment_quality_score += bike_infrastructure_scores[b_tag_value]
                    found_bike_tag_for_score = True

            # If no specific bike tag, consider highway type (e.g., residential streets are generally safer)
            if not found_bike_tag_for_score and 'highway' in data:
                hw_tag_value = data['highway']
                if f'highway={hw_tag_value}' in bike_infrastructure_scores:
                    segment_quality_score += bike_infrastructure_scores[f'highway={hw_tag_value}']
                # Also consider paths/tracks that aren't explicitly tagged bicycle=no
                elif hw_tag_value in ['path', 'track'] and data.get('bicycle') != 'no':
                    segment_quality_score += bike_infrastructure_scores.get(f'highway={hw_tag_value}', 0.5) # Default score for generic paths/tracks

            # Add scores for surface and smoothness (can apply to any road)
            if 'surface' in data and data['surface'] in bike_infrastructure_scores:
                segment_quality_score += bike_infrastructure_scores[data['surface']]
            if 'smoothness' in data and data['smoothness'] in bike_infrastructure_scores:
                segment_quality_score += bike_infrastructure_scores[data['smoothness']]

            # Cap segment quality score to a reasonable range
            segment_quality_score = max(0, min(segment_quality_score, 1.5)) # Ensure non-negative and not excessively high

            weighted_length += length * segment_quality_score

            # Store metrics for detailed infrastructure types (e.g., total length of cycleways)
            for tag_key, _ in bike_infrastructure_scores.items():
                if isinstance(tag_key, str) and '=' in tag_key:
                    main_tag, sub_tag = tag_key.split('=')
                    if main_tag in data and data[main_tag] == sub_tag:
                        infrastructure_metrics[tag_key] = infrastructure_metrics.get(tag_key, 0) + length
                elif tag_key in data and data.get(tag_key) == 'yes': # For tags like 'bicycle_parking' or 'cycleway' if it's just 'yes'
                     infrastructure_metrics[tag_key] = infrastructure_metrics.get(tag_key, 0) + length


        total_length_miles = total_length / 1609.34 # Convert meters to miles
        weighted_length_miles = weighted_length / 1609.34
        area_sq_miles = (np.pi * (radius/1609.34)**2) # Area of buffer in square miles

        density = total_length_miles / area_sq_miles if area_sq_miles > 0 else 0
        weighted_density = weighted_length_miles / area_sq_miles if area_sq_miles > 0 else 0

        logging.info(f"   Bike lane weighted density for {lat}, {lon}: {weighted_density:.4f} miles/sq_mile")
        return {
            'density': density,
            'weighted_density': weighted_density,
            'total_length': total_length_miles,
            'weighted_length': weighted_length_miles,
            'infrastructure_metrics': infrastructure_metrics
        }
    except Exception as e:
        logging.error(f"Error calculating bike lane density for {lat}, {lon}: {e}")
        return {
            'density': 0, 'weighted_density': 0, 'total_length': 0,
            'weighted_length': 0, 'infrastructure_metrics': {}
        }

def get_capacity_score(capacity, optimal_capacity_threshold=OPTIMAL_CAPACITY_THRESHOLD):
    """
    Calculates a normalized capacity score.
    A capacity equal to or greater than 'optimal_capacity_threshold' receives a 100 score.
    Capacities below this threshold are scaled linearly.
    """
    if optimal_capacity_threshold <= 0:
        logging.warning("optimal_capacity_threshold must be greater than 0. Returning 0.")
        return 0

    raw_score = (capacity / optimal_capacity_threshold) * 100
    normalized_score = min(100, max(0, raw_score)) # Ensure non-negative and capped at 100
    return normalized_score

def get_green_space_score(lat, lon, radius=500):
    """
    Calculates a score based on the percentage of the buffer area covered by green spaces.
    A score of 100 means 100% of the buffer area is covered by green space.
    """
    green_space_tags = {
        'leisure': ['park', 'garden', 'playground', 'golf_course', 'nature_reserve'],
        'landuse': ['forest', 'meadow', 'grass', 'recreation_ground', 'farmland'],
        'natural': ['wood', 'scrub', 'wetland', 'grassland', 'heath']
    }

    try:
        green_features = ox.features_from_point((lat, lon), tags=green_space_tags, dist=radius)

        if green_features.empty:
            logging.info(f"   No green spaces found for {lat}, {lon} within {radius}m.")
            return 0

        # Filter for polygon geometries (most relevant for area calculation)
        polygons_gdf = green_features[green_features.geometry.apply(lambda geom: geom and geom.geom_type in ['Polygon', 'MultiPolygon'])]

        if polygons_gdf.empty:
            logging.info(f"   No polygon green spaces found for {lat}, {lon} within {radius}m.")
            return 0

        # Project to a local UTM zone for accurate area calculation (in meters)
        projected_gdf = ox.projection.project_gdf(polygons_gdf)

        total_green_area_sq_m = projected_gdf.geometry.area.sum()

        # Calculate the total area of the circular buffer given the radius
        buffer_area_sq_m = math.pi * (radius**2)

        logging.info(f"   Total green area for {lat}, {lon}: {total_green_area_sq_m:.2f} sq meters (out of {buffer_area_sq_m:.2f} total buffer area)")

        # Normalize directly as a percentage of the buffer area.
        # A 100 score now literally means 100% of the buffer is green space.
        normalized_score = min(100, (total_green_area_sq_m / buffer_area_sq_m) * 100)

        return normalized_score
    except Exception as e:
        logging.error(f"Error calculating green space score for {lat}, {lon}: {e}")
        return 0


def calculate_enhanced_bikeability(lat, lon, radius=500, capacity=0):
    """
    Calculates an enhanced bikeability index for a given location.
    Integrates multiple metrics within a consistent buffer distance.
    """
    try:
        # Pass the consistent radius to all sub-functions
        bike_lane_metrics = get_bike_lane_density(lat, lon, radius)
        bike_lane_score = min(100, bike_lane_metrics['weighted_density'] * 10) # Multiplier 10 can be adjusted

        poi_data = get_nearby_pois(lat, lon, radius)
        poi_score = poi_data['normalized_score']

        intersection_density = get_intersection_density(lat, lon, radius)
        intersection_score = min(100, intersection_density * 5) # Multiplier 5 can be adjusted

        network_density = get_multimodal_network_density(lat, lon, radius)
        # Adjusted network_score normalization
        network_score = min(100, (network_density / OPTIMAL_NETWORK_DENSITY) * 100)
        logging.info(f"   Network Score (normalized) for {lat}, {lon}: {network_score:.2f}")

        crime_score = get_crime_score(lat, lon, radius) # Placeholder

        capacity_score = get_capacity_score(capacity)
        logging.info(f"   Capacity Score (normalized) for {lat}, {lon}: {capacity_score:.2f}")

        green_space_score = get_green_space_score(lat, lon, radius)
        logging.info(f"   Green Space Score (normalized) for {lat}, {lon}: {green_space_score:.2f}")

        # --- FINAL SCORE WEIGHTS (Adjust these based on importance) ---
        # Ensure these weights sum up to 1.0 (100%)
        final_score = (
            0.50 * bike_lane_score +       # Importance of dedicated bike infrastructure
            0.20 * poi_score +             # Importance of nearby amenities
            0.10 * network_score +         # Importance of overall network connectivity
            0.05 * capacity_score +        # Importance of station capacity
            0.15 * green_space_score       # Importance of access to green spaces
            # 0.00 * crime_score           # If you integrate crime, adjust weights here
        )

        return {
            'total_score': final_score,
            'bike_lane_score': bike_lane_score,
            'bike_lane_metrics': bike_lane_metrics, # Detailed metrics
            'poi_score': poi_score,
            'poi_details': poi_data['counts'], # Detailed counts of POIs
            'intersection_score': intersection_score,
            'intersection_density': intersection_density,
            'network_score': network_score,
            'network_density': network_density,
            'crime_score': crime_score,
            'capacity_score': capacity_score,
            'green_space_score': green_space_score # New score added
        }
    except Exception as e:
        logging.error(f"Critical error calculating enhanced bikeability for {lat}, {lon}: {e}\n{traceback.format_exc()}")
        # Return default zero scores in case of an error for a specific station
        return {
            'total_score': 0, 'bike_lane_score': 0, 'bike_lane_metrics': {},
            'poi_score': 0, 'poi_details': {}, 'intersection_score': 0,
            'intersection_density': 0, 'network_score': 0, 'network_density': 0,
            'crime_score': 0, 'capacity_score': 0, 'green_space_score': 0
        }

def analyze_stations():
    """
    Analyzes a sample of Citi Bike stations and adds enhanced bikeability scores.
    """
    input_filename = 'citibike_stations.csv'
    output_filename = 'bikeability_index.csv'

    try:
        print(f"Reading stations data from {input_filename}...")
        # Read CSV from the same directory as the script
        stations = pd.read_csv(input_filename)

        required_cols = ['lat', 'lon', 'capacity', 'station_name', 'name']
        for col in required_cols:
            if col not in stations.columns:
                logging.error(f"Missing required column: {col}. Please check your CSV file.")
                print(f"Error: Missing required column '{col}'. Ensure your CSV has 'lat', 'lon', 'capacity', 'station_name', and 'name'.")
                return

        # Select a sample of 15 stations for analysis (can be changed)
        # Using .copy() to prevent SettingWithCopyWarning
        stations_to_analyze = stations.sample(n=15, random_state=42).copy()
        print(f"Analyzing {len(stations_to_analyze)} selected stations.")

        print("Available columns in dataframe:", stations_to_analyze.columns.tolist())
        print("\nFirst few rows of selected data:")
        print(stations_to_analyze.head()) # Use print() instead of display() for .py script

        print("Calculating enhanced bikeability scores...")
        scores_data = []
        # Determine which column to use as a station identifier for logging/errors
        station_id_col = 'station_name' if 'station_name' in stations_to_analyze.columns else ('name' if 'name' in stations_to_analyze.columns else stations_to_analyze.index.name or 'index')


        for index, row in tqdm(stations_to_analyze.iterrows(), total=len(stations_to_analyze), desc="Processing Stations"):
            current_station_id = row.get(station_id_col, f"Index_{index}")
            logging.info(f"--- Processing station {index+1}/{len(stations_to_analyze)}: {current_station_id} ---")

            capacity_val = 0
            if 'capacity' in row and pd.notna(row['capacity']):
                try:
                    capacity_val = int(row['capacity'])
                except ValueError:
                    logging.warning(f"Invalid capacity value for station {current_station_id}: '{row['capacity']}'. Using 0.")

            # Call the main bikeability calculation function
            score_result = calculate_enhanced_bikeability(row['lat'], row['lon'], capacity=capacity_val)
            scores_data.append(score_result)
            logging.info(f"--- Finished station {index+1}/{len(stations_to_analyze)}: {current_station_id} ---")


        print("\nAdding scores to dataframe...")
        # Add new columns to the stations_to_analyze DataFrame
        stations_to_analyze['bikeability_total'] = [s['total_score'] for s in scores_data]
        stations_to_analyze['bike_lane_score'] = [s['bike_lane_score'] for s in scores_data]
        stations_to_analyze['bike_lane_density'] = [s['bike_lane_metrics']['density'] for s in scores_data]
        stations_to_analyze['bike_lane_weighted_density'] = [s['bike_lane_metrics']['weighted_density'] for s in scores_data]
        stations_to_analyze['poi_score'] = [s['poi_score'] for s in scores_data]
        stations_to_analyze['intersection_score'] = [s['intersection_score'] for s in scores_data]
        stations_to_analyze['network_score'] = [s['network_score'] for s in scores_data]
        stations_to_analyze['crime_score'] = [s['crime_score'] for s in scores_data]
        stations_to_analyze['capacity_score'] = [s['capacity_score'] for s in scores_data]
        stations_to_analyze['green_space_score'] = [s['green_space_score'] for s in scores_data] 
        stations_to_analyze['intersection_density'] = [s['intersection_density'] for s in scores_data]
        stations_to_analyze['network_density'] = [s['network_density'] for s in scores_data]

        # Dynamically add columns for detailed POI counts
        all_poi_keys = set()
        for s in scores_data:
            if 'poi_details' in s and s['poi_details']:
                all_poi_keys.update(s['poi_details'].keys())
        for poi_type_key in sorted(list(all_poi_keys)): # Sort for consistent column order
            stations_to_analyze[f'poi_{poi_type_key}'] = [s.get('poi_details', {}).get(poi_type_key, 0) for s in scores_data]

        # Dynamically add columns for detailed bike infrastructure metrics
        all_infra_keys = set()
        for s in scores_data:
            if 'bike_lane_metrics' in s and 'infrastructure_metrics' in s['bike_lane_metrics'] and s['bike_lane_metrics']['infrastructure_metrics']:
                all_infra_keys.update(s['bike_lane_metrics']['infrastructure_metrics'].keys())
        for infra_key in sorted(list(all_infra_keys)): # Sort for consistent column order
            stations_to_analyze[f'infra_{infra_key}'] = [s.get('bike_lane_metrics', {}).get('infrastructure_metrics', {}).get(infra_key, 0) for s in scores_data]


        print("\nSaving results...")
        stations_to_analyze.to_csv(output_filename, index=False)
        print(f"Analysis complete! Results saved to {output_filename}")

        print("\nSummary Statistics of Key Scores:")
        print(stations_to_analyze[['bikeability_total', 'bike_lane_score', 'poi_score',
                                     'intersection_score', 'network_score', 'crime_score',
                                     'capacity_score', 'green_space_score']].describe())

        print(f"\nResults saved to '{output_filename}' in the current directory.")

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found in the same directory as the script.")
        print("Please ensure 'citibike_stations.csv' is present.")
    except Exception as e:
        logging.critical(f"A critical error occurred during analysis: {e}\n{traceback.format_exc()}")
        print(f"A critical error occurred: {e}. Check the log output for details.")

# --- Main execution block ---
if __name__ == "__main__":
    print("Starting bikeability analysis for NYC Citi Bike stations...")
    analyze_stations()