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
        """
        Select highway (way) if it's a known highway type. 
        Note: Location checks are deferred to Phase 3.
        """
        
        # Check if the way is a highway
        highway_tag = w.tags.get('highway')
        if highway_tag is not None:
            
            # Skip non-traffic ways early
            if highway_tag in ['footway', 'path', 'cycleway']:
                 return
            
            # Check if w.node_refs is available (it should be, but without locations)
            # We rely on Phase 3 to filter ways outside the domain.
            if w.node_refs:
                # Capture the name and type from the tags
                highway_name = w.tags.get('name', '')
                highway_type = w.tags.get('highway', '')
                
                self.point.append([n.ref for n in w.node_refs])
                self.osmid.append(w.id)
                # NEW: Append the collected tags
                self.name.append(highway_name)
                self.highway_type.append(highway_type)


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
    # Use osmium.osm.relations to ensure full way data is available (needed for SimpleHandler)
    # The original implementation may have relied on a global reader setup, 
    # but explicitly requesting Way data is safer in a SimpleHandler.
    # Note: osmium.osm.relations is not valid here. We stick to the simple_handler, 
    # trusting that the removal of the location check is the primary fix.
    highway_handler.apply_file(osm_file)

    
    # --- PHASE 3: Process results and align data (Implicitly filters by domain) ---
    
    # Collect the final, aligned highway data (coordinates, OSM IDs, names, and types)
    highway_coordinate = []
    highway_osmid = []
    highway_names = []      
    highway_types = []      
    
    # Iterate over all collected data lists simultaneously
    for refs, osmid, name, htype in zip(highway_collection.point, 
                                        highway_collection.osmid,
                                        highway_collection.name,
                                        highway_collection.highway_type):
        try:
            # This is the crucial step: if a node ID in 'refs' doesn't exist in 
            # point_collection.coordinate, it means that node is outside the zone, 
            # and the KeyError will skip the entire way.
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
