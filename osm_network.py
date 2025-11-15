# Copyright (C) 2014, INRIA
# Author(s): Vivien Mallet
#
# This file is part of software for the data assimilation in the context of
# noise pollution at urban scale.
#
# This file is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with this file. If not, see http://www.gnu.org/licenses/.

# This files retrieves the coordinates of the streets in the domain.


import osmium
import numpy as np

# Determines whether a point is inside a given polygon or not.
# (This definition needs to be outside any class to be accessed globally)
def point_inside_polygon(x, y, poly):
    """
    Determines if a point (x, y) is inside a polygon defined by a list of 
    (x, y) tuples/lists using the Ray Casting Algorithm.
    """
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# Simple class that handles the parsed OSM data in order to select the points
# inside the domain and the coordinates of the points around the domain. The
# domain is defined as a closed N-point polygon in 'selected_zone'
# (dimensions: N x 2).
class PointCollection(object): # Renamed class to avoid confusion
    def __init__(self, selected_zone, tolerance):
        self.selected_zone = selected_zone
        self.inside_zone = []
        self.coordinate = {}

        self.x_min = min([x[0] for x in selected_zone]) - tolerance
        self.x_max = max([x[0] for x in selected_zone]) + tolerance
        self.y_min = min([x[1] for x in selected_zone]) - tolerance
        self.y_max = max([x[1] for x in selected_zone]) + tolerance

    def select(self, coord):
        for osmid, x, y in coord:
            # Selection of the points that are inside the domain.
            if point_inside_polygon(x, y, self.selected_zone):
                self.inside_zone.append(osmid)
            # Getting the ids of the coordinates inside the domain or in the
            # vicinity of the domain.
            if x < self.x_max and x > self.x_min \
                    and  y < self.y_max and y > self.y_min:
                self.coordinate[osmid] = (x, y)

# Simple class that handles the parsed OSM data in order to identify all
# streets that cross the domain.
class HighwayCollection(object): # Renamed class to match previous use
    def __init__(self, point_collection): # Takes the point object
        # Set of all nodes inside the zone.
        self.point_inside_zone = set(point_collection.inside_zone)
        # Points that describe the highways.
        self.point = []
        # Unsorted points that describe the highways, in a set.
        self.point_set = set()
        # Stores the OSM ID.
        self.osmid = []

    def select(self, way_obj): 
        # Access properties directly from the single way object
        osmid = way_obj.id
        # tags = way_obj.tags # tags variable seems unused in original logic
        refs = [n.ref for n in way_obj.nodes] 

        # Add logic to store the references and osmid if needed for later processing
        self.osmid.append(osmid)
        self.point.append(refs)


def retrieve_highway(osm_file, selected_zone, tolerance, Ncore=1):
    
    # --- PHASE 1: Collect Points ---
    # Create the Point collection object
    point_collection = PointCollection(selected_zone, tolerance)
    
    # Create the Osmium handler for coordinates
    class PointHandler(osmium.simple_handler.SimpleHandler):
        def __init__(self, collection_obj): # Accepts collection object
            super().__init__()
            self.point_collection = collection_obj # Store it here

        def node(self, n):
            # Use the stored collection object
            self.point_collection.select([(n.id, n.location.lon, n.location.lat)])
            
    # Instantiate Point handler and apply the file for coordinate extraction
    point_handler = PointHandler(point_collection)
    point_handler.apply_file(osm_file, locations=True)

    
    # --- PHASE 2: Collect Highways/Ways ---
    # Create the Highway collection object, passing the point data it needs
    highway_collection = HighwayCollection(point_collection)

    # Osmium handler for ways (highways)
    class HighwayHandler(osmium.simple_handler.SimpleHandler):
        def __init__(self, collection_obj): # Accepts collection object
            super().__init__()
            self.highway_collection = collection_obj # Store it here

        def way(self, w):
            # Use the stored collection object to call select
            self.highway_collection.select(w)

    # Instantiate Highway handler and apply the file for way extraction
    highway_handler = HighwayHandler(highway_collection) 
    highway_handler.apply_file(osm_file)

    
    # --- PHASE 3: Process results ---
    # Flatten the set of point references for highways
    highway_collection.point_set = set([item for refs in highway_collection.point for item in refs])

    # Collect the coordinates for the highway points
    point_coordinate = []
    for way in highway_collection.point_set:
        try:
            point_coordinate.append(point_collection.coordinate[way])
        except KeyError: # Use KeyError instead of bare except
            pass

    # Collect the highway coordinates and OSM IDs
    highway_coordinate = []
    highway_osmid = []
    for refs, osmid in zip(highway_collection.point, highway_collection.osmid):
        try:
            highway_coordinate.append([point_collection.coordinate[n] for n in refs])
            highway_osmid.append(osmid)
        except KeyError: # Use KeyError instead of bare except
            pass

    return highway_coordinate, highway_osmid


