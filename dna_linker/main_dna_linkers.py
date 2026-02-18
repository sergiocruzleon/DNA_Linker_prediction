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
gpu_accelerate=config.gpu_accelerate
single_cluster_file=config.single_cluster_file




def main_function(packet_list):
    tomo_id=packet_list[0]
    cluster=packet_list[1]
    output_path=packet_list[2]
    output_path2=packet_list[3] 
    output_path_dictionary=packet_list[4] 
    lo=packet_list[5]
    lp=packet_list[6]

    
    largest_components=[]
    
    
    print(f'Tomogram: {tomo_id}, and cluster {cluster}')

    motl=cryomotl.EmMotl(input_motl=output_path+f'motl_tomo{tomo_id}_cluster{cluster}.em')
    # Load the particles centered at the exit
    motl_exit=cryomotl.EmMotl(input_motl=output_path+f'All_motl_tomo{tomo_id}_cluster{cluster}_exit.em')
    motl_exit2=cryomotl.EmMotl(input_motl=output_path+f'All_motl_tomo{tomo_id}_cluster{cluster}_exit2.em')
                
    # Load the particles centered at the entry
    motl_entry=cryomotl.EmMotl(input_motl=output_path+f'All_motl_tomo{tomo_id}_cluster{cluster}_entry.em')
    motl_entry2=cryomotl.EmMotl(input_motl=output_path+f'All_motl_tomo{tomo_id}_cluster{cluster}_entry2.em')
                
    # Calculate the probabilities for all pairs and cases within the cluster
    if gpu_accelerate and dnal.TORCH_AVAILABLE:
        print(f"  Using GPU acceleration ({dnal.DEVICE}) for probability calculations...")
        probs=dnal.calculate_probabilities_gpu(motl=motl, 
                                                         motl_exit=motl_exit, 
                                                         motl_exit2=motl_exit2, 
                                                         motl_entry=motl_entry,
                                                         motl_entry2=motl_entry2, 
                                                         lo=lo, lp=lp)
    elif gpu_accelerate and not dnal.TORCH_AVAILABLE:
        print("  WARNING: GPU acceleration requested but PyTorch not available. Using CPU...")
        probs=dnal.calculate_probabilities_all_connexions(motl=motl, 
                                                         motl_exit=motl_exit, 
                                                         motl_exit2=motl_exit2, 
                                                         motl_entry=motl_entry,
                                                         motl_entry2=motl_entry2, 
                                                         lo=lo, lp=lp)
    else:
        probs=dnal.calculate_probabilities_all_connexions(motl=motl, 
                                                         motl_exit=motl_exit, 
                                                         motl_exit2=motl_exit2, 
                                                         motl_entry=motl_entry,
                                                         motl_entry2=motl_entry2, 
                                                         lo=lo, lp=lp)
                
    # Compute max probabilities (needed for connectivity analysis)
    max_probs = np.max(probs, axis=2)

    #print(f"----------------------------")
    #print(f"--The used lo is {lo}-pixels-")
    #print(f"--The used lp is {lp}-pixels-")
    #print(f"--The used lo is {lo}--")
    #print(f"----------------------------")
    
    # Plot the connectivity matrix with the max probabilities - stream directly to disk
    # Only save plots if explicitly enabled in config
    if config.save_plots:
        fig=plt.figure()
        plt.imshow(max_probs)
        # -
        plt.title(f'Connectivity tomo {tomo_id}, cluster {cluster}')
        plt.ylabel('Particles')
        plt.xlabel('Particles')
        plt.colorbar()
        # Save directly to disk without displaying
        fig.savefig(output_path2 + f'Connectivity_tomo{tomo_id}_cluster{cluster}.pdf', dpi=150)
        plt.close(fig)
    
    
    # Stream the max probabilities matrix directly to disk as numpy file
    # This avoids keeping large arrays in memory
    max_probs_path = output_path2+f'MaxProb_tomo{tomo_id}_cluster{cluster}.npy'
    np.save(max_probs_path, max_probs)
    # Free memory immediately after saving
    del max_probs
                
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
                
    #print(matrix.shape)
    while np.any(matrix):
        max_index = dnal.find_next_maximum(matrix)
    
        if matrix[max_index]==0:
            break
        connections, remaining_connections = dnal.create_connection(connections, matrix,  max_index, remaining_connections)
        matrix[max_index] =0  # Set the maximum value to negative infinity to exclude it in the next iteration
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
                                     output_filename=output_filename, min_prob=dnal.pmin )

    
    dnal.save_connnections(connections, output_path_dictionary+f'Connectivity_motl_tomo{tomo_id}_cluster{cluster}.pickle')
    #paths_conn.append(output_path_dictionary+f'Connectivity_motl_tomo{tomo_id}_cluster{cluster}.pickle')

    return largest_component
    
    