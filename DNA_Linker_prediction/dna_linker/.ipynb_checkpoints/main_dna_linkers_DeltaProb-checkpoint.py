import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import matplotlib
#print(matplotlib.__version__)

import warnings
warnings.filterwarnings('ignore')
from cryocat import cryomotl
from cryocat import cryomap
import pandas as pd
import networkx as nx
import pickle
import copy

from dna_linker import dna_linkers as dnal
from dna_linker import create_graph as cgraph

from dna_linker import config

#####################
# CONSTANTS
#####################
kB=config.kB
T= config.T
bin=config.bin 
lp=config.lp
lo=config.lo
pmin=config.pmin




def main_function(packet_list):
    tomo_id=packet_list[0]
    cluster=packet_list[1]
    output_path=packet_list[2]
    output_path2=packet_list[3] 
    output_path_dictionary=packet_list[4] 
    lo=packet_list[5]

    
    largest_components=[]
    
    
    print(f'Tomogram: {tomo_id}, and cluster {cluster}')

    probs_linkers=[]
    second_probs=[]

    motl=cryomotl.EmMotl(input_motl=output_path+f'motl_tomo{tomo_id}_cluster{cluster}.em')
    # Load the particles centered at the exit
    motl_exit=cryomotl.EmMotl(input_motl=output_path+f'All_motl_tomo{tomo_id}_cluster{cluster}_exit.em')
    motl_exit2=cryomotl.EmMotl(input_motl=output_path+f'All_motl_tomo{tomo_id}_cluster{cluster}_exit2.em')
                
    # Load the particles centered at the entry
    motl_entry=cryomotl.EmMotl(input_motl=output_path+f'All_motl_tomo{tomo_id}_cluster{cluster}_entry.em')
    motl_entry2=cryomotl.EmMotl(input_motl=output_path+f'All_motl_tomo{tomo_id}_cluster{cluster}_entry2.em')
                
    # Calculate the probabilities for all pairs and cases within the cluster           
    probs=dnal.calculate_probabilities_all_connexions(motl=motl, 
                                                             motl_exit=motl_exit, 
                                                             motl_exit2=motl_exit2, 
                                                             motl_entry=motl_entry,
                                                             motl_entry2=motl_entry2, 
                                                             lo=dnal.lo)
                
    # Plot the connectivity matrix with the max probabilities
    fig=plt.figure()
    max_probs = np.max(probs, axis=2)
    plt.imshow(max_probs)
    # -
    plt.title(f'Connectivity tomo {tomo_id}, cluster {cluster}')
    plt.ylabel('Particles')
    plt.xlabel('Particles')
    plt.colorbar(label=r'Max P($\theta$, L, $\Gamma$)')
                
    plt.show()
    #Save the connectivity plots
    fig.savefig(output_path2+f'MaxProb_tomo{tomo_id}_cluster{cluster}.pdf', 
                            bbox_inches='tight')
                
    ################################################################################
    #############Estima the connections based on a hierarchical assingment of ###################################################################
    ################################################################################
                
    # Matrix with all the probabilities
    matrix=probs
    num_particles = matrix.shape[0]
    # Initialize dictionary to store connections
    connections = {i: [] for i in range(num_particles)}
                
    # Remaining connection keeps the max number of connections to 2.
    remaining_connections = {i: 2 for i in range(num_particles)}  # Initialize remaining connections for each particle
    # Create connections iteratively based on the maximum values
                
    print(matrix.shape)
    while np.any(matrix):
        max_index = dnal.find_next_maximum(matrix)
    
        if matrix[max_index]==0:
            break
        # Take a snapshot of connections before updating
        connections_snapshot = copy.deepcopy(connections)
            
        connections, remaining_connections = dnal.create_connection(connections, matrix,  max_index, remaining_connections)
        if connections != connections_snapshot:
            
            # Trigger frame generation if connections changed
            assigned_prob = matrix[max_index]
            probs_linkers.append(assigned_prob)
            # Calculate second maximum for the row corresponding to max_index[0]
            max_probs = np.max(matrix, axis=2)
            row_values = max_probs[max_index[0], :]
            sorted_row = np.sort(row_values)
            second_max = sorted_row[-2] if len(sorted_row) > 1 else None  # Ensure there are at least 2 elements
            second_probs.append(second_max)
            
            
        
        #matrix[max_index] =0  # Set the maximum value to negative infinity to exclude it in the next iteration
        matrix[max_index[0],max_index[1],:] = 0  # Zero out the maximum value
        matrix[max_index[1],max_index[0],:] = 0  # Zero out the maximum value - The matrix is symmetric
        
    ################################################################################
    ################################################################################
    ############################nb####################################################
    
    largest_component=cgraph.draw_graph2(connections)
    largest_components.append(largest_component)
                
    ################################################################################
    ################################################################################
    ###############################################################################
                
                
    output_filename=output_path2+f'motl_tomo{tomo_id}_cluster{cluster}_linkers.em'
                
    dnal.write_out_EmMolt_linker(connections=connections, motl_exit=motl_exit, 
                                     motl_entry=motl_entry,
                                     motl_entry2=motl_entry2,
                                     motl_exit2=motl_exit2,
                                     output_filename=output_filename, min_prob=0.1 )

    
    dnal.save_connnections(connections, output_path_dictionary+f'Connectivity_motl_tomo{tomo_id}_cluster{cluster}.pickle')
    
    probs_linkers=np.array(probs_linkers)
    second_probs=np.array(second_probs)
    
    ## Save Delta P
    np.savetxt(fname=output_path_dictionary+f'Delta_probs/tomo{tomo_id}_cluster{cluster}_probs_linkers.txt', X=probs_linkers)
    np.savetxt(fname=output_path_dictionary+f'Delta_probs/tomo{tomo_id}_cluster{cluster}_second_probs.txt',X=second_probs)
    
    #paths_conn.append(output_path_dictionary+f'Connectivity_motl_tomo{tomo_id}_cluster{cluster}.pickle')

    return largest_component
    
    