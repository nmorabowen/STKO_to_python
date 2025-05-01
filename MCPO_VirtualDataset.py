import h5py
import numpy as np
import glob  # Import the glob module
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import concurrent.futures
import re

from .NODES import NODES
from .ON_ELEMENTS import ON_ELEMENTS
from .ERRORS import errorChecks
from .GET_MODEL_INFO import GetModelInfo
from .PLOTTER import plotter
from .CDATA import CDATA
from .MAPPING import MAPPING

class MCPO_VirtualDataset(NODES, 
                          ON_ELEMENTS, 
                          errorChecks, 
                          GetModelInfo, 
                          plotter,
                          CDATA,
                          MAPPING):
    
    # Common path templates
    MODEL_NODES_PATH = "/{model_stage}/MODEL/NODES"
    MODEL_ELEMENTS_PATH = "/{model_stage}/MODEL/ELEMENTS"
    RESULTS_ON_ELEMENTS_PATH = "/{model_stage}/RESULTS/ON_ELEMENTS"
    RESULTS_ON_NODES_PATH = "/{model_stage}/RESULTS/ON_NODES"

    def __init__(self, 
                 results_directory:str, 
                 recorder_name:str,
                 results_directory_name='results_h5', #OLD must be removed
                 results_filename='results.h5', #OLD must be removed
                 file_extension='*.mpco',
                 print_summary=False):
        """
        Initialize the MCPO_VirtualDataset instance.
        
        Args:
            results_directory (str): Path to the directory containing the source partition files.
            results_directory_name (str): Name of the directory to store the virtual dataset.
            results_filename (str): Name of the virtual dataset file.
            file_extension (str): File extension to identify partition files.
        """
        self.results_directory = results_directory
        self.file_extension = file_extension
        self.recorder_name = recorder_name
        
        # Get file list for mpco files and cdata files
        self.file_info_mpco=self._get_file_list(extension='mpco')
        self.file_info_cdata=self._get_file_list(extension='cdata')
        
        # Get the results partitions
        self.results_partitions = self._get_results_partitions()
        self.cdata_partitions = self._get_cdata_partitions()
        
        # Get model global information
        self.model_stages= self.get_model_stages()
        self.element_results_names=self.get_elements_results_names()
        self.element_types=self.get_element_types()
        self.unique_element_types=self._get_all_types()
        
        self.node_results_names=self.get_node_results_names()
        
        # Get model time series and steps info
        self.time=self.get_time_series()
        
        # Define the database path and directory
        self._define_virtual_paths(results_directory_name=results_directory_name, results_filename=results_filename)
        
        # Create the virtual dataset directory
        self._create_results_directory()
        
        # Create the virtual dataset
        self.create_virtual_dataset()
        
        # Get node and element information
        # In order to query the data efficiently, we will store the mappings in a structured numpy array and df
        # Usually the size of this arrays is not too big, so we can store them in memory, the method contain a print_memory=True statement to check the size of the arrays
        self.nodes_info=self._get_all_nodes_ids(print_memory=True)
        self.elements_info=self._get_all_element_index(print_memory=True)
        
        # Get number of steps
        self.number_of_steps=self.get_number_of_steps()
        
        # Get the selection sets mapping
        self.selection_set=self.extract_selection_set_ids()
        
        self.build_and_store_mappings()
        
        # Print summary
        if print_summary:
            self._print_summary()
            

    

    def _define_virtual_paths(self, results_directory_name, results_filename):
        """
        Define paths for the virtual dataset and its directory.
        """
        self.virtual_data_set_directory = os.path.join(self.results_directory, results_directory_name)
        self.virtual_data_set = os.path.join(self.virtual_data_set_directory, results_filename)
    
    def _create_results_directory(self):
        """
        Ensure that the directory for the virtual dataset exists.
        """
        os.makedirs(self.virtual_data_set_directory, exist_ok=True)
    
    def _get_results_partitions(self):
        file_info=self.file_info_mpco
        # Filter by recorder name
        results_partitions=file_info[self.recorder_name]
        
        return results_partitions

    def _get_cdata_partitions(self):
        file_info=self.file_info_cdata
        # Filter by recorder name
        results_partitions=file_info[self.recorder_name]
        
        return results_partitions
    
    def create_virtual_dataset(self):
        """
        Create or update the virtual dataset file.
        """
        if os.path.exists(self.virtual_data_set):
            print(f"Virtual dataset already exists at {self.virtual_data_set}. File will be overwritten.")
            os.remove(self.virtual_data_set)
        else:
            print(f"Creating virtual dataset at {self.virtual_data_set}...")

        # Create an empty HDF5 file for the virtual dataset
        with h5py.File(self.virtual_data_set, 'w') as virtual_h5:
            print(f"Virtual dataset file created at {self.virtual_data_set}, ready for linking datasets.")
    
    def _print_summary(self):
        """
        Print a summary of the virtual dataset.
        ---------------------------------------
        """
        print(f'File name: {self.virtual_data_set}')
        print(f"Virtual dataset created at {self.virtual_data_set}")
        print(f'Number of partitions: {len(self.results_partitions)}')
        
        print('------------------------------------------------------')
        
        print(f"Number of model stages: {len(self.model_stages)}")
        print(f'Model stages: {self.model_stages}')
        for stage in self.model_stages:
            print(f"  - {stage}")
            
        print('------------------------------------------------------')
        print(f'Number of nodal results: {len(self.node_results_names)}')
        for name in self.node_results_names:
            print(f"  - {name}")
            
        print('------------------------------------------------------')
        print(f'Number of element results: {len(self.element_results_names)}')
        for name in self.element_results_names:
            print(f"  - {name}")
        print(f'Number of unique element types: {len(self.unique_element_types)}')
        for name in self.unique_element_types:
            print(f"  - {name}")
        
        print('------------------------------------------------------')
        print('General model information:')
        
        print(f"Number of nodes: {len(self.nodes_info)}")
        print(f"Number of element types: {len(self.unique_element_types)}")
        print(f"Number of elements: {len(self.elements_info)}")
        print(f"Number of steps: {self.number_of_steps}")
        print(f"Number of selection sets: {len(self.selection_set)}")

    def print_selection_set_info(self):
        """Method to print the selection set information.
        """
        
        for key in self.selection_set.keys():
            print(f"Selection set: {key}")
            print(f"Selection Set name: {self.selection_set[key]['SET_NAME']}")
            print('------------------------------------------------------')
    
    def create_reduced_hdf5(self, node_ids, element_ids, output_file):
        """
        Create a reduced HDF5 file containing only the specified nodes and elements with their attributes.

        Args:
            node_ids (list): List of node IDs to include in the reduced file.
            element_ids (list): List of element IDs to include in the reduced file.
            output_file (str): Path to save the reduced HDF5 file.
        """
        with h5py.File(self.virtual_data_set, 'r') as original_file, h5py.File(output_file, 'w') as reduced_file:
            # Copy relevant nodes
            for model_stage in self.get_model_stages():
                nodes_path = self.MODEL_NODES_PATH.format(model_stage=model_stage)
                if nodes_path in original_file:
                    node_info = self.get_node_coordinates(model_stage, node_ids)
                    node_group = reduced_file.require_group(nodes_path)
                    node_list = node_info['node list']
                    coords = node_info['coordinates']

                    node_group.create_dataset('node_ids', data=node_list)
                    node_group.create_dataset('coordinates', data=coords)

                # Copy relevant elements
                elements_path = self.RESULTS_ON_ELEMENTS_PATH.format(model_stage=model_stage)
                if elements_path in original_file:
                    element_group = original_file[elements_path]
                    reduced_element_group = reduced_file.require_group(elements_path)
                    for element_id in element_ids:
                        if f"ID_{element_id}" in element_group:
                            original_data = element_group[f"ID_{element_id}"]
                            reduced_element_group.create_dataset(f"ID_{element_id}", data=original_data)

            print(f"Reduced HDF5 file created at {output_file}")
    

