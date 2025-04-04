import matplotlib.pyplot as plt
from typing import List, Optional

class ContactMapVisualizer:
    """
    A utility class for visualizing protein/RNA contact maps.
    """
    @staticmethod
    def plot_contact_map(contact_map,
                         sequence: Optional[str] = None,
                         title: str = "Protein Contact Map",
                         cmap: str = "viridis",
                         save_path: Optional[str] = None):
        """
        Plot a single contact map.

        :param contact_map: 2D array representing the contact map
        :param sequence: Optional sequence string for labeling axes
        :param title: Title of the plot
        :param cmap: Color map to use
        :param save_path: Optional path to save the figure
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(contact_map, cmap=cmap)
        plt.colorbar()
        plt.title(title)

        # Add sequence labels if provided
        if sequence is not None:
            ticks = range(0, len(sequence), 20) # Show every 20th residue
            plt.xticks(ticks, [sequence[i] for i in ticks])
            plt.yticks(ticks, [sequence[i] for i in ticks])

        # Save figure if path is provided
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_multiple_contact_map(contact_maps: List,
                                  titles: Optional[List[str]] = None,
                                  cols: int = 2):
        """
        Plot multiple contact maps in a grid layout.

        :param contact_maps: List of contact maps to plot
        :param titles: Optional list of titles for each subplot
        :param cols: Number of columns in the grid
        """

        # Calculate number of rows needed
        rows = (len(contact_maps) + cols - 1) // cols
        plt.figure(figsize=(cols * 6, rows * 5))

        # Plot each contact map in a subplot
        for i, cmap in enumerate(contact_maps):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(cmap, cmap="viridis")
            plt.colorbar()
            if titles and i < len(titles):
                plt.title(titles[i])

        plt.tight_layout() # Adjust layout to prevent overlap
        plt.show()
