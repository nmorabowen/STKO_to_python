import numpy as np
import matplotlib.pyplot as plt

class plotter:
    """This is a mixin class for plotting results for the MCPO_VirtualDataset class"""
    
    def plot_nodal_results(
        self,
        model_stage,
        results_name_verticalAxis=None,
        node_ids_verticalAxis=None,
        selection_set_id_verticalAxis=None,
        direction_verticalAxis=None,
        values_operation_verticalAxis='Sum',
        scaling_factor_verticalAxis=1.0,
        results_name_horizontalAxis=None,
        node_ids_horizontalAxis=None,
        selection_set_id_horizontallAxis=None,
        direction_horizontalAxis=None,
        values_operation_horizontalAxis='Sum',
        scaling_factor_horizontalAxis=1.0,
        ax=None,
        figsize=(10, 6),
        color='k',
        linetype='-',
        linewidth=0.75,
        label=None,
    ):
        """
        Plots nodal results with specified parameters.
        
        Args:
            model_stage (str): Name of the model stage to retrieve data.
            results_name_verticalAxis (str): Name of the vertical axis results.
            node_ids_verticalAxis (list): List of node IDs for vertical axis results.
            selection_set_id_verticalAxis (int): Selection set ID for vertical axis.
            direction_verticalAxis (str): Direction of vertical axis results ('x', 'y', or 'z').
            values_operation_verticalAxis (str): Aggregation operation for vertical axis ('Sum', 'Mean', etc.).
            results_name_horizontalAxis (str): Name of the horizontal axis results.
            node_ids_horizontalAxis (list): List of node IDs for horizontal axis results.
            selection_set_id_horizontallAxis (int): Selection set ID for horizontal axis.
            direction_horizontalAxis (str): Direction of horizontal axis results ('x', 'y', or 'z').
            values_operation_horizontalAxis (str): Aggregation operation for horizontal axis.
            ax (matplotlib.axes.Axes): Pre-existing axes to plot on. If None, a new figure is created.
            figsize (tuple): Size of the figure if no `ax` is provided.
            color (str): Line color for the plot.
            linetype (str): Line style for the plot.
            linewidth (float): Line width for the plot.
            label (str): Label for the legend.
            
        Returns:
            matplotlib.axes.Axes: Axes object containing the plot.
        """
        
        
        if results_name_verticalAxis not in ['STEP', 'TIME']:
            self._nodal_results_name_error(results_name_verticalAxis, model_stage)
            
        if results_name_horizontalAxis not in ['STEP', 'TIME']:
            self._nodal_results_name_error(results_name_horizontalAxis, model_stage)
        
        
        if results_name_verticalAxis not in ['STEP', 'TIME']:
            # Retrieve results for vertical
            vertical_results_df = self.get_nodal_results(
                model_stage=model_stage,
                results_name=results_name_verticalAxis,
                node_ids=node_ids_verticalAxis,
                selection_set_id=selection_set_id_verticalAxis,
            )
            # Aggregate results
            y_array = plotter._aggregate_results(vertical_results_df, direction_verticalAxis, values_operation_verticalAxis) * scaling_factor_verticalAxis
        elif results_name_verticalAxis == 'STEP':
            y_array = self.time.loc[model_stage].index.tolist()
        else:
            y_array = self.time.loc[model_stage]['TIME'].values
        
        if results_name_horizontalAxis not in ['STEP', 'TIME']:
            # Retrieve results for horizontal axes
            horizontal_results_df = self.get_nodal_results(
                model_stage=model_stage,
                results_name=results_name_horizontalAxis,
                node_ids=node_ids_horizontalAxis,
                selection_set_id=selection_set_id_horizontallAxis,
            )
            # Aggregate results
            x_array = plotter._aggregate_results(horizontal_results_df, direction_horizontalAxis, values_operation_horizontalAxis) * scaling_factor_horizontalAxis
        elif results_name_horizontalAxis == 'STEP':
            x_array = self.time.loc[model_stage].index.tolist()
        else:
            x_array = self.time.loc[model_stage]['TIME'].values

        if len(x_array) != len(y_array):
            raise ValueError("Mismatch in lengths of horizontal and vertical data arrays.")

        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Plot the data
        ax.plot(x_array, y_array, color=color, linestyle=linetype, linewidth=linewidth, label=label)
        if label:
            ax.legend()
        ax.set_xlabel(results_name_horizontalAxis or "Horizontal Axis")
        ax.set_ylabel(results_name_verticalAxis or "Vertical Axis")
        ax.grid(True)
        
        return ax
    
        
    @staticmethod
    def _aggregate_results(results_df, direction, operation):
        """
        Aggregates the results DataFrame based on the specified operation for the given direction.

        Args:
            results_df (pd.DataFrame): DataFrame with results, including a 'step' index and direction columns.
            direction (str): Direction to aggregate ('x', 'y', 'z').
            operation (str): Operation to perform ('Sum', 'Mean', 'Max', 'Min').

        Returns:
            np.ndarray: Aggregated values for the specified direction.
        """
        # Validate direction
        if direction not in results_df.columns:
            raise KeyError(f"Direction '{direction}' not found in DataFrame columns: {results_df.columns}. The available directions are: {results_df.columns}")
        
        # Group and aggregate only the relevant column
        if operation == 'Sum':
            aggregated_values = results_df.groupby('step')[direction].sum()
        elif operation == 'Mean':
            aggregated_values = results_df.groupby('step')[direction].mean()
        elif operation == 'Max':
            aggregated_values = results_df.groupby('step')[direction].max()
        elif operation == 'Min':
            aggregated_values = results_df.groupby('step')[direction].min()
        else:
            raise ValueError(f"Invalid operation: {operation}")
        
        return aggregated_values.values
        
        
    
