import pandas as pd
import folium
from folium.plugins import Fullscreen, MiniMap
import branca.colormap as cm
import numpy as np
from folium import plugins
import json

def create_bikeability_map(input_file='bikeability_index.csv'):
    """
    Creates an enhanced interactive map visualization of bikeability scores.
    """
    try:
        # Read the bikeability data
        print("Reading bikeability data...")
        df = pd.read_csv(input_file)
        
        # Create a base map centered on NYC with different tile layers
        print("Creating base map...")
        m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
        
        # Add different tile layers
        folium.TileLayer('openstreetmap').add_to(m)
        folium.TileLayer('cartodbpositron').add_to(m)
        folium.TileLayer('cartodbdark_matter').add_to(m)
        
        # Create color scales for different metrics
        colormap_total = cm.LinearColormap(
            colors=['red', 'yellow', 'green'],
            vmin=df['bikeability_total'].min(),
            vmax=df['bikeability_total'].max(),
            caption='Total Bikeability Score'
        )
        colormap_total.add_to(m)
        
        colormap_bike_lane = cm.LinearColormap(
            colors=['red', 'yellow', 'green'],
            vmin=df['bike_lane_score'].min(),
            vmax=df['bike_lane_score'].max(),
            caption='Bike Lane Score'
        )
        
        colormap_poi = cm.LinearColormap(
            colors=['red', 'yellow', 'green'],
            vmin=df['poi_score'].min(),
            vmax=df['poi_score'].max(),
            caption='POI Score'
        )
        
        # Create feature group for markers
        fg_markers = folium.FeatureGroup(name='Station Markers')
        
        # Add markers for each station with enhanced popups
        print("Adding station markers...")
        for idx, row in df.iterrows():
            # Create enhanced popup content with detailed information and styling
            popup_content = f"""
            <div style='font-family: Arial, sans-serif; padding: 10px;'>
                <h3 style='color: #2c3e50; margin-bottom: 10px;'>{row['station_name']}</h3>
                <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px;'>
                    <h4 style='color: #34495e; margin-bottom: 5px;'>Scores:</h4>
                    <table style='width: 100%; border-collapse: collapse;'>
                        <tr>
                            <td style='padding: 5px;'><b>Total Score:</b></td>
                            <td style='padding: 5px;'>{row['bikeability_total']:.1f}</td>
                        </tr>
                        <tr>
                            <td style='padding: 5px;'><b>Bike Lane Score:</b></td>
                            <td style='padding: 5px;'>{row['bike_lane_score']:.1f}</td>
                        </tr>
                        <tr>
                            <td style='padding: 5px;'><b>POI Score:</b></td>
                            <td style='padding: 5px;'>{row['poi_score']:.1f}</td>
                        </tr>
                        <tr>
                            <td style='padding: 5px;'><b>Network Score:</b></td>
                            <td style='padding: 5px;'>{row['network_score']:.1f}</td>
                        </tr>
                        <tr>
                            <td style='padding: 5px;'><b>Green Space Score:</b></td>
                            <td style='padding: 5px;'>{row['green_space_score']:.1f}</td>
                        </tr>
                        <tr>
                            <td style='padding: 5px;'><b>Capacity Score:</b></td>
                            <td style='padding: 5px;'>{row['capacity_score']:.1f}</td>
                        </tr>
                    </table>
                </div>
                <div style='margin-top: 10px;'>
                    <h4 style='color: #34495e; margin-bottom: 5px;'>Infrastructure Metrics:</h4>
                    <p>Bike Lane Density: {row['bike_lane_density']:.2f} miles/sq_mile</p>
                    <p>Weighted Bike Lane Density: {row['bike_lane_weighted_density']:.2f} miles/sq_mile</p>
                    <p>Intersection Density: {row['intersection_density']:.2f} intersections/m²</p>
                    <p>Network Density: {row['network_density']:.2f} miles/sq_mile</p>
                </div>
            </div>
            """
            
            # Create a circle marker with color based on bikeability score
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=8,
                popup=folium.Popup(popup_content, max_width=400),
                color=colormap_total(row['bikeability_total']),
                fill=True,
                fill_color=colormap_total(row['bikeability_total']),
                fill_opacity=0.7,
                weight=2,
                tooltip=f"{row['station_name']} - Score: {row['bikeability_total']:.1f}"
            ).add_to(fg_markers)
        
        # Add the feature group to the map
        fg_markers.add_to(m)
        
        # Add additional plugins
        Fullscreen().add_to(m)
        MiniMap().add_to(m)
        
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Save the map
        print("Saving map...")
        output_file = 'bikeability_map.html'
        m.save(output_file)
        print(f"Map saved as {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error creating bikeability map: {str(e)}")
        return False

def main():
    """
    Main function to run the visualization
    """
    print("Starting bikeability visualization...")
    success = create_bikeability_map()
    
    if success:
        print("\nVisualization complete!")
        print("Open 'bikeability_map.html' in your web browser to view the interactive map.")
        print("\nFeatures available in the visualization:")
        print("- Multiple map layers (OpenStreetMap, CartoDB)")
        print("- Station markers with detailed popups")
        print("- Fullscreen mode")
        print("- Mini map")
    else:
        print("\nVisualization failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 