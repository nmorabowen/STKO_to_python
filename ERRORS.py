

class errorChecks:
    """
    A mixin class for MPCO_VirtualDataSet containing error handling methods.
    This class provides error checking functionality for model stages and result names
    to ensure valid inputs when working with the MPCO_VirtualDataSet class.
    Methods
    -------
    _model_stages_error(model_stage)
        Validates that the provided model stage is valid.
        Parameters
        ----------
        model_stage : str
            The model stage name to validate
        Raises
        ------
        TypeError
            If model_stage is not a string or not in the list of valid model stages
    _results_name_error(result_name, model_stage)  
        Validates both the model stage and result name.
        Parameters
        ----------
        result_name : str
            The name of the result to validate
        model_stage : str
            The model stage name to validate
        Raises
        ------
        TypeError
            If result_name is not a string or not in list of valid result names
            If model_stage is not valid (via _model_stages_error)
    """
    
    def _model_stages_error(self, model_stage):
        """
        Validates if the given model stage is valid and exists in the model stages.
        Parameters
        ----------
        model_stage : str
            The name of the model stage to validate.
        Raises
        ------
        TypeError
            If the model_stage is not a string or if it's not found in the available model stages.
        Notes
        -----
        Uses get_model_stages() to fetch the list of valid model stages for comparison.
        """
        
        model_stages=self.get_model_stages()
        
        if not isinstance(model_stage, str) or model_stage not in model_stages:
            raise TypeError(f'f"The model stage must be a string, the model stages names are: {model_stages}"')
        
    def _element_results_name_error(self, result_name, model_stage):
        """
        Check if the result name is valid for the given model stage.
        Parameters
        ----------
        result_name : str
            Name of the result to be checked.
        model_stage : str
            Stage of the model where the result is located.
        Raises
        ------
        TypeError
            If result_name is not a string or if it's not in the available results names for the given model stage.
        Notes
        -----
        This method first validates the model stage, then checks if the result name exists in the available results
        for that stage. The available results are obtained using get_elements_results_names method.
        """
        
        # Model stage check
        self._model_stages_error(model_stage)
        
        results_names=self.get_elements_results_names(model_stage=model_stage)
        
        if not isinstance(result_name, str) or result_name not in results_names:
            raise TypeError(f'f"The result name must be a string, the results names are: {results_names}"')
        
    def _nodal_results_name_error(self, result_name, model_stage):
        """
        Check if the result name is valid for the given model stage.
        Parameters
        ----------
        result_name : str
            Name of the result to be checked.
        model_stage : str
            Stage of the model where the result is located.
        Raises
        ------
        TypeError
            If result_name is not a string or if it's not in the available results names for the given model stage.
        Notes
        -----
        This method first validates the model stage, then checks if the result name exists in the available results
        for that stage. The available results are obtained using get_node_results_names method.
        """
        
        # Model stage check
        self._model_stages_error(model_stage)
        
        results_names=self.get_node_results_names(model_stage=model_stage)
        
        if not isinstance(result_name, str) or result_name not in results_names:
            raise TypeError(f'f"The result name must be a string, the results names are: {results_names}"')
    
    
    def _element_type_name_error(self, element_type):
            
            element_types=self.element_types['unique_element_types']
            
            if not isinstance(element_type, str) or element_type not in element_types:
                raise TypeError(f'The element type must be a string, the element types are: {sorted(list(element_types))}')