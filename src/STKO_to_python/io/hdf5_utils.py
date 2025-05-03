# mcpo_virtual_dataset/io/hdf5_utils.py
import h5py

class HDF5Utils:
    """
    HDF5Utils
    =========

    A utility class that provides static methods for safely working with HDF5 files.

    This class encapsulates common operations for HDF5 files such as opening files,
    accessing groups and datasets, reading data, and working with attributes. It includes
    error handling and validation to ensure robust interaction with HDF5 data structures.

    Methods
    -------
    open_file(path, mode='r')
        Opens an HDF5 file with proper error handling.

    get_group(file, group_path, required=False)
        Retrieves a group from an HDF5 file with optional validation.

    get_dataset(group, dataset_name, required=False)
        Retrieves a dataset from an HDF5 group with optional validation.

    list_keys(group)
        Lists all keys in an HDF5 group.

    get_attrs(group, keys)
        Retrieves specified attributes from an HDF5 object.

    has_path(file, path)
        Checks if a path exists in an HDF5 file.

    read_dataset_as_numpy(group, dataset_name, required=False)
        Reads a dataset into a NumPy array.

    get_all_attributes(obj)
        Retrieves all attributes from an HDF5 object.

    Examples
    --------
    >>> with HDF5Utils.open_file('data.h5', 'r') as f:
    ...     # Get a group
    ...     group = HDF5Utils.get_group(f, 'MODEL_STAGE[1]')
    ...     # List available datasets
    ...     datasets = HDF5Utils.list_keys(group)
    ...     # Read a dataset as numpy array
    ...     data = HDF5Utils.read_dataset_as_numpy(group, 'temperatures')
    ...     # Get specific attributes
    ...     attrs = HDF5Utils.get_attrs(group, ['units', 'timestamp'])

    Notes
    -----
    This utility class is designed to make HDF5 operations more robust by providing
    consistent error handling and validation options. It helps prevent common issues
    when working with HDF5 files such as missing paths or failed file operations.
    """
    
    @staticmethod
    def open_file(path, mode='r'):
        """
        Opens an HDF5 file and returns a file object.
        
        Serves as a context manager for safe HDF5 file access with proper error handling.
        
        Parameters
        ----------
        path : str
            Path to the HDF5 file.
        mode : str, optional
            File access mode, default is 'r' (read-only).
            Other common modes: 'r+' (read/write), 'w' (create/overwrite),
            'a' (read/write/create).
            
        Returns
        -------
        h5py.File
            An open HDF5 file object.
            
        Raises
        ------
        IOError
            If the file cannot be opened with a descriptive error message.
            
        Examples
        --------
        >>> with HDF5Utils.open_file('data.h5', 'r') as f:
        ...     # Work with the file
        ...     pass
        """
        try:
            return h5py.File(path, mode)
        except Exception as e:
            raise IOError(f"Error opening HDF5 file '{path}': {e}")

    @staticmethod
    def get_group(file, group_path, required=False):
        """
        Retrieves a group from an HDF5 file.
        
        Parameters
        ----------
        file : h5py.File
            An open HDF5 file object.
        group_path : str
            Path to the group within the HDF5 file.
        required : bool, optional
            If True, raises KeyError when the group doesn't exist.
            Default is False, which returns None for missing groups.
            
        Returns
        -------
        h5py.Group or None
            The requested HDF5 group if it exists, None otherwise.
            
        Raises
        ------
        KeyError
            If required=True and the group does not exist.
            
        Examples
        --------
        >>> with HDF5Utils.open_file('data.h5', 'r') as f:
        ...     # Get an optional group
        ...     group1 = HDF5Utils.get_group(f, 'measurements')
        ...     # Get a required group (will raise KeyError if missing)
        ...     group2 = HDF5Utils.get_group(f, 'calibration', required=True)
        """
        
        group = file.get(group_path)
        if required and group is None:
            raise KeyError(f"Required group '{group_path}' not found in file.")
        return group

    @staticmethod
    def get_dataset(group, dataset_name, required=False):
        """
        Retrieves a dataset from an HDF5 group.
        
        Parameters
        ----------
        group : h5py.Group
            An HDF5 group object.
        dataset_name : str
            Name of the dataset within the group.
        required : bool, optional
            If True, raises KeyError when the dataset doesn't exist.
            Default is False, which returns None for missing datasets.
            
        Returns
        -------
        h5py.Dataset or None
            The requested HDF5 dataset if it exists, None otherwise.
            
        Raises
        ------
        KeyError
            If required=True and the dataset does not exist.
            
        Examples
        --------
        >>> with HDF5Utils.open_file('data.h5', 'r') as f:
        ...     group = HDF5Utils.get_group(f, 'measurements')
        ...     # Get an optional dataset
        ...     dataset1 = HDF5Utils.get_dataset(group, 'temperatures')
        ...     # Get a required dataset (will raise KeyError if missing)
        ...     dataset2 = HDF5Utils.get_dataset(group, 'timestamps', required=True)
        """
        
        dataset = group.get(dataset_name)
        if required and dataset is None:
            raise KeyError(f"Required dataset '{dataset_name}' not found in group.")
        return dataset

    @staticmethod
    def list_keys(group):
        """
        Lists all keys (datasets and subgroups) in an HDF5 group.
        
        Parameters
        ----------
        group : h5py.Group or None
            An HDF5 group object. If None, returns an empty list.
            
        Returns
        -------
        list
            List of string keys in the group, or empty list if group is None.
            
        Examples
        --------
        >>> with HDF5Utils.open_file('data.h5', 'r') as f:
        ...     group = HDF5Utils.get_group(f, 'measurements')
        ...     keys = HDF5Utils.list_keys(group)
        ...     print(f"Available datasets: {keys}")
        """
        
        return list(group.keys()) if group else []

    @staticmethod
    def get_attrs(group, keys):
        """
        Fetches specified attributes from an HDF5 object if they exist.
        
        Parameters
        ----------
        group : h5py.Group or h5py.Dataset
            An HDF5 object with attributes.
        keys : list
            List of attribute names to retrieve.
            
        Returns
        -------
        dict
            Dictionary mapping attribute names to their values.
            Only includes attributes that exist in the object.
            
        Examples
        --------
        >>> with HDF5Utils.open_file('data.h5', 'r') as f:
        ...     dataset = HDF5Utils.get_dataset(f, 'measurements/temperatures')
        ...     attrs = HDF5Utils.get_attrs(dataset, ['units', 'timestamp', 'calibrated'])
        ...     if 'units' in attrs:
        ...         print(f"Temperature units: {attrs['units']}")
        """
        
        """Fetch a dict of attributes if they exist."""
        return {k: group.attrs.get(k) for k in keys if k in group.attrs}
    
    @staticmethod
    def has_path(file, path):
        """
        Checks if a given path exists in the HDF5 file.
        
        Parameters
        ----------
        file : h5py.File
            An open HDF5 file object.
        path : str
            Path to check within the HDF5 file.
            
        Returns
        -------
        bool
            True if the path exists, False otherwise.
            
        Examples
        --------
        >>> with HDF5Utils.open_file('data.h5', 'r') as f:
        ...     if HDF5Utils.has_path(f, 'measurements/temperatures'):
        ...         print("Temperature data exists")
        ...     else:
        ...         print("Temperature data not found")
        """
        return path in file
    
    @staticmethod
    def read_dataset_as_numpy(group, dataset_name, required=False):
        """Read dataset into numpy array."""
        ds = HDF5Utils.get_dataset(group, dataset_name, required=required)
        return ds[...] if ds is not None else None
    
    @staticmethod
    def get_all_attributes(obj):
        """
        Returns all attributes of an HDF5 object as a dictionary.
        
        Parameters
        ----------
        obj : h5py.Group or h5py.Dataset
            An HDF5 object with attributes.
            
        Returns
        -------
        dict
            Dictionary containing all attributes of the object.
            
        Examples
        --------
        >>> with HDF5Utils.open_file('data.h5', 'r') as f:
        ...     dataset = HDF5Utils.get_dataset(f, 'measurements/temperatures')
        ...     # Get all attributes
        ...     all_attrs = HDF5Utils.get_all_attributes(dataset)
        ...     print(f"Dataset has {len(all_attrs)} attributes")
        ...     for name, value in all_attrs.items():
        ...         print(f"{name}: {value}")
        """
        return dict(obj.attrs)
    
    @staticmethod
    def list_all_groups(file):
        """
        Lists all groups in an HDF5 file.
        
        Parameters
        ----------
        file : h5py.File
            An open HDF5 file object.
            
        Returns
        -------
        list
            List of group names in the file.
            
        Examples
        --------
        >>> with HDF5Utils.open_file('data.h5', 'r') as f:
        ...     groups = HDF5Utils.list_all_groups(f)
        ...     print(f"Groups in the file: {groups}")
        """
        print('================================================================')
        print("Groups in the root of the file:")
        for key in file.keys():
            print(key)




