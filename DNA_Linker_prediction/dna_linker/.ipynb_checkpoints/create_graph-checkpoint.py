import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import matplotlib
# Set the global fontsize for Matplotlib
matplotlib.rcParams.update({'font.size': 12})  

from cryocat import cryomotl
from cryocat import cryomap
import pandas as pd
import networkx as nx
import pickle


def draw_graph(connections):
    """
    Draw a graph using the connections information.

    Parameters:
    - connections (dict): A dictionary where keys are particle indices and values
      are lists of tuples. Each tuple contains the index of a connected particle,
      the corresponding probability, and the case.
    """
    G = nx.Graph()
    
    for particle, particle_connections in connections.items():
        for connection in particle_connections:
            neighbor, probability, case = connection
            G.add_edge(particle, neighbor, weight=probability, case=case)
    
    pos = nx.spring_layout(G)  # Position nodes using the spring layout algorithm
    
    # Extract edge weights and cases
    edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]
    edge_cases = [data['case'] for _, _, data in G.edges(data=True)]
    
    # Draw edges with thickness proportional to probability and color based on case
    nx.draw(G, pos, width=5*(edge_weights)/np.max(edge_weights), edge_color=(edge_weights)/np.max(edge_weights), 
            edge_cmap=plt.cm.Blues, with_labels=True)
    #nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)})
    
    plt.show()



def draw_graph2(connections, pmin=0.1, ):
    """
    Draw a graph using the connections information.

    Parameters:
    - connections (dict): A dictionary where keys are particle indices and values
      are lists of tuples. Each tuple contains the index of a connected particle,
      the corresponding probability, and the case.
    """
    G = nx.Graph()
    
    for particle, particle_connections in connections.items():
        for connection in particle_connections:
            neighbor, probability, case = connection
            if probability>=pmin:
                G.add_edge(particle, neighbor, weight=probability, case=case)
    
    # Generate random colors for nodes and edges
    node_colors = np.random.rand(len(G.nodes()), 3)
    edge_colors = np.random.rand(len(G.edges()), 3)

    pos = nx.spring_layout(G)  # Position nodes using the spring layout algorithm
    
    # Extract edge weights and cases
    edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]
    edge_cases = [data['case'] for _, _, data in G.edges(data=True)]

    
    
    # Get connected components
    connected_components = list(nx.connected_components(G))

    # Generate a unique color for each connected component
    colors = plt.cm.get_cmap('viridis', len(connected_components))

    # Create a color map for nodes
    node_color_map = {}
    for i, component in enumerate(connected_components):
        for node in component:
            node_color_map[node] = colors(i)
    
    # Extract node colors in the same order as G.nodes()
    node_colors = [node_color_map[node] for node in G.nodes()]
    
    # Extract edge weights
    edge_weights = nx.get_edge_attributes(G, 'weight')

    # Draw nodes with specified colors
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200, cmap=plt.cm.viridis)
    
    # Draw nodes with colors assigned by component
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200, cmap=plt.cm.hsv)
    
    # Draw edges with thickness proportional to weights
    nx.draw_networkx_edges(G, pos, 
                           width=[weight * 10 for weight in edge_weights.values()],
                           edge_color='black')
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=7, font_color='white')

    
    # Find all connected components
    connected_components = list(nx.connected_components(G))
    #connected_components = list(nx.strongly_connected_components(G))

    # Find the largest connected component
    if len(connected_components)!=0:
        largest_component = max(connected_components, key=len)
        print("Largest connected component:", largest_component)
        print("Size of largest connected component:", len(largest_component))
    else:
        largest_component =0
    # Print the largest connected component and its size
    
    print(largest_component)
    plt.show()

    return largest_component

