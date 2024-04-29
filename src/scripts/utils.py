#### Utils file for helper functions
#### in the Fathomnet EDA
################################
import os
import logging
from fathomnet.api import images
import plotly.express as px
from ipyleaflet import Map, Heatmap, basemaps

logging.basicConfig(level=logging.INFO)

def get_concepts(species):
    """
    Get the concepts for a given species.
    Inputs:
    - species: str, the species to get the concepts for
    """
    
    species.replace('_', ' ')
    # Get the concepts for the species
    concepts = images.get_concepts(species)
    return concepts

def plot_depth_histogram(selected_concept):
    """
    Plot a histogram of the depths of images for a given concept.
    Inputs:
    - selected_concept: str, the concept to plot the histogram for

    Returns:
    - None
    """
    concept_images = get_concepts(selected_concept)

    # Print the total number of images found
    logging.info(f"Found {len(concept_images)} images of {selected_concept}")

    # Extract the depth (in meters) from each image where it is available
    depths = [image.depthMeters for image in concept_images if image.depthMeters is not None]

    # Check if depths list is not empty to avoid errors in plotting
    if depths:
        # Make a horizontal histogram
        fig = px.histogram(y=depths, title=f'{selected_concept} Images by Depth', labels={'y': 'Depth (m)'})
        fig.update_layout(yaxis={'autorange': 'reversed'})  # Ensure the depth axis shows deeper as lower down
        fig.show()
    else:
        logging.warning(f"No depth data available for {selected_concept}.")


def plot_location_heatmap(species, center=(36.807, -121.988), zoom=10):
    """Plot a heatmap of image locations based on latitude and longitude.
    Inputs:
    - concept_images: list of Image objects
    - center: tuple of floats, the center of the map (latitude, longitude)
    - zoom: int, the zoom level of the map
    
    Returns:
    - map_object: ipyleaflet Map object
    """
    concept_images = get_concepts(species)
    locations = [
        (image.latitude, image.longitude)
        for image in concept_images
        if image.latitude is not None and image.longitude is not None
    ]

    # Create a map using the Esri Ocean basemap
    map_object = Map(basemap=basemaps.Esri.OceanBasemap, center=center, zoom=zoom)
    map_object.layout.height = "800px"

    # Check if there are any locations to plot
    if locations:
        # Overlay the image locations as a heatmap
        heatmap_layer = Heatmap(locations=locations, radius=20, min_opacity=0.5)
        map_object.add_layer(heatmap_layer)
        return map_object
    else:
        logging.warning(f"No location data available for {species}.")
        return None
