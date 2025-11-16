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
from multiprocessing import Manager, Process
from itertools import chain

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


class PointCollection():
    """Class to store coordinate of selected points/nodes"""

    def __init__(self, zone, tolerance):
        self.coordinate = Manager().dict()
        self.zone = zone
        self.tolerance = tolerance

    def select(self, n):
        """Select node if it is within the given zone."""
        lon = n.location.lon
        lat = n.location.lat
        if point_inside_polygon(lon, lat, self.zone):
            self.coordinate[n.id] = (lon, lat)


class HighwayCollection():
    """Class to store the point id of selected ways/highways"""

    def __init__(self, zone):
        self.osmid = []
        self.point = []
        # NEW: Lists to store road name and highway type
        self.name = []
        self.highway_type = []
        # END NEW
        self.zone = zone

    def select(self, w):
        """Select highway (way) if its end points are within the zone and it's a known highway type."""
        
        # Check if the way is a highway (we ignore cycleways, footways, etc. here for traffic analysis)
        if w.tags.get('highway') is not None:
            # Check if at least one endpoint is in the zone
            lon0 = w.node_refs[0].location.lon
            lat0 = w.node_refs[0].location.lat
            lon1 = w.node_refs[-1].location.lon
            lat1 = w.node_refs[-1].location.lat

            in_zone = point_inside_polygon(lon0, lat0, self.zone) or \
                      point_inside_polygon(lon1, lat1, self.zone)
            
            # Additional check if it's not a path or a cycleway (optional, depends on definition)
            highway_tag = w.tags.get('highway')
            if highway_tag in ['footway', 'path', 'cycleway']:
                 return
            
            if in_zone:
                # Capture the name and type from the tags
                highway_name = w.tags.get('name', '')
                highway_type = w.tags.get('highway', '')
                
                self.point.append([n.ref for n in w.node_refs])
                self.osmid.append(w.id)
                # NEW: Append the collected tags
                self.name.append(highway_name)
                self.highway_type.append(highway_type)
                # END NEW


def retrieve_highway(osm_file, zone, tolerance, ncore):
    """
    Main function to retrieve street network data from OSM file, including 
    coordinates, OSM IDs, road names, and highway types.
    
    Returns:
        highway_coordinate: List of lists of (lon, lat) coordinates for each road segment.
        highway_osmid: List of OSM IDs for each road segment.
        highway_names: List of road names for each road segment.
        highway_types: List of highway types (e.g., 'primary', 'residential') for each road segment.
    """
    
    # --- PHASE 1: Collect coordinates of nodes in the zone ---
    point_collection = PointCollection(zone, tolerance)
    
    # Simple handler for nodes/points
    class PointHandler(osmium.simple_handler.SimpleHandler):
        def __init__(self, collection_obj): # Accepts collection object
            super().__init__()
            self.point_collection = collection_obj # Store it here

        def node(self, n):
            # Use the stored collection object to call select
            self.point_collection.select(n)

    # Instantiate Point handler and apply the file for node extraction
    point_handler = PointHandler(point_collection)
    point_handler.apply_file(osm_file)


    # --- PHASE 2: Collect point references, OSM IDs, names, and types for ways (highways) ---
    highway_collection = HighwayCollection(zone)
    
    # Simple handler for ways (highways)
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

    
    # --- PHASE 3: Process results and align data ---
    
    # Flatten the set of point references for highways
    highway_collection.point_set = set([item for refs in highway_collection.point for item in refs])

    # Collect the coordinates for the highway points (no longer strictly needed for return, but useful for debug)
    # point_coordinate = []
    # for way in highway_collection.point_set:
    #     try:
    #         point_coordinate.append(point_collection.coordinate[way])
    #     except KeyError: 
    #         pass

    # Collect the final, aligned highway data (coordinates, OSM IDs, names, and types)
    highway_coordinate = []
    highway_osmid = []
    highway_names = []      # NEW
    highway_types = []      # NEW
    
    # Iterate over all collected data lists simultaneously
    for refs, osmid, name, htype in zip(highway_collection.point, 
                                        highway_collection.osmid,
                                        highway_collection.name,
                                        highway_collection.highway_type):
        try:
            # Check if all node references have coordinates (i.e., they were in the domain)
            coords = [point_collection.coordinate[n] for n in refs]
            
            # If successful, append all four pieces of data
            highway_coordinate.append(coords)
            highway_osmid.append(osmid)
            highway_names.append(name)
            highway_types.append(htype)
            
        except KeyError: 
            # Skip this way entirely if any node coordinate is missing (i.e., outside the defined domain)
            pass

    # MODIFIED RETURN: Returns 4 lists now
    return highway_coordinate, highway_osmid, highway_names, highway_types
