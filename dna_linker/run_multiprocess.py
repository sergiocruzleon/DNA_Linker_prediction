import numpy as np
import pandas as pd

from cryocat import cryomap
from cryocat import cryomotl

#from cryocat import motl_conversions as mc


#DNA Linker
from dna_linker import dna_linkers as dnal
from dna_linker import create_graph as cgraph
from dna_linker import main_dna_linkers as main
from dna_linker import utils_motlFiles as util_motl
from dna_linker import config

# Muliprocessing
from multiprocessing import Pool
import networkx as nx
import matplotlib.pyplot as plt
import pickle

import sys


def run_cluster_processing(
    motl_trace_input: str,
    output_path_cluster: str,
    output_path_linker: str,
    output_path_dictionary: str,
    dnal_object,
    lp_object,
    max_processes: int = 10,
    batch_gpu: bool = True
) -> None:
    """
    Runs parallel cluster-based processing on a traced motive list.

    Parameters:
        motl_trace_input (str): Path to the traced motive list (.em file).
        output_path_cluster (str): Directory to store per-cluster outputs.
        output_path_linker (str): Directory for shifted linker outputs.
        output_path_dictionary (str): Directory for intermediate dictionary data.
        dnal_object: Contour length parameter (lo) passed into each task.
        lp_object: Persistence length parameter (lp) passed into each task.
        max_processes (int): Max number of worker processes to use (default: 10).
        batch_gpu (bool): Whether to use batched GPU processing (default: True).

    Returns:
        None
    """
    # Load traced motive list
    motl_trace = cryomotl.EmMotl(input_motl=motl_trace_input)
    tomograms = motl_trace.df['tomo_id'].unique()

    # Build task list
    packed_paths = []
    for tomo_id in tomograms:
        print(f"Tomogram: {tomo_id}")
        df_motl_tomo = motl_trace.df[motl_trace.df['tomo_id'] == tomo_id]
        clusters = df_motl_tomo['geom1'].unique()

        for cluster in clusters:
            packed_paths.append([
                tomo_id,
                cluster,
                output_path_cluster,
                output_path_linker,
                output_path_dictionary,
                dnal_object,
                lp_object
            ])

    # Check if we should use batched GPU processing
    use_gpu_batching = batch_gpu and config.gpu_accelerate and dnal.TORCH_AVAILABLE
    
    if use_gpu_batching and len(packed_paths) > 1:
        print(f"\nUsing batched GPU processing for {len(packed_paths)} clusters...")
        run_batched_gpu_processing(packed_paths)
    else:
        # Limit worker processes to avoid oversubscription
        num_processes = min(max_processes, len(packed_paths))

        # Run tasks in parallel
        with Pool(num_processes) as p:
            p.map(main.main_function, packed_paths)
    
    # After all clusters are processed, combine them if single_cluster_file is enabled
    if config.single_cluster_file:
        print("\nCombining all cluster linker files into single file...")
        combined_path = dnal.combine_cluster_linker_files(
            output_path_linker=output_path_linker,
            output_filename='all_linkers.em'
        )
        if combined_path:
            print(f"Combined file saved to: {combined_path}")


def run_batched_gpu_processing(packed_paths):
    """
    Process multiple clusters in a single GPU batch for maximum throughput.
    
    This collects all cluster data, processes them in one GPU batch,
    then writes out the results for each cluster.
    
    Args:
        packed_paths: List of task packets, each containing:
            [tomo_id, cluster, output_path_cluster, output_path_linker, 
             output_path_dictionary, dnal_object]
    """
    from dna_linker import dna_linkers as dnal
    from dna_linker import create_graph as cgraph
    from dna_linker import config
    import matplotlib.pyplot as plt
    import pickle
    
    print(f"  Loading {len(packed_paths)} clusters...")
    
    # Collect all cluster data
    cluster_data = []
    for packet in packed_paths:
        tomo_id = packet[0]
        cluster = packet[1]
        output_path = packet[2]
        
        # Load cluster data
        motl = cryomotl.EmMotl(input_motl=output_path + f'motl_tomo{tomo_id}_cluster{cluster}.em')
        motl_exit = cryomotl.EmMotl(input_motl=output_path + f'All_motl_tomo{tomo_id}_cluster{cluster}_exit.em')
        motl_exit2 = cryomotl.EmMotl(input_motl=output_path + f'All_motl_tomo{tomo_id}_cluster{cluster}_exit2.em')
        motl_entry = cryomotl.EmMotl(input_motl=output_path + f'All_motl_tomo{tomo_id}_cluster{cluster}_entry.em')
        motl_entry2 = cryomotl.EmMotl(input_motl=output_path + f'All_motl_tomo{tomo_id}_cluster{cluster}_entry2.em')
        
        cluster_data.append({
            'tomo_id': tomo_id,
            'cluster': cluster,
            'output_path': output_path,
            'motl': motl,
            'motl_exit': motl_exit,
            'motl_exit2': motl_exit2,
            'motl_entry': motl_entry,
            'motl_entry2': motl_entry2
        })
    
    # Extract just the motl objects for batch processing
    motl_tuples = [
        (d['motl'], d['motl_exit'], d['motl_exit2'], d['motl_entry'], d['motl_entry2'])
        for d in cluster_data
    ]
    
    # Process all clusters in one GPU batch
    lo = packed_paths[0][5]  # Get lo from the first packet (all packets have same lo)
    lp = packed_paths[0][6]  # Get lp from the first packet (all packets have same lp)
    probs_list = dnal.calculate_probabilities_batch_gpu(motl_tuples, lo=lo, lp=lp)
    
    # Process each cluster result
    for idx, cluster_info in enumerate(cluster_data):
        tomo_id = cluster_info['tomo_id']
        cluster = cluster_info['cluster']
        output_path = cluster_info['output_path']
        output_path2 = packed_paths[idx][3]
        output_path_dictionary = packed_paths[idx][4]
        motl = cluster_info['motl']
        
        probs = probs_list[idx]
        
        print(f"  Processing cluster {cluster} (tomo {tomo_id})...")
        
       
        # Rest of the processing from main_function
        max_probs = np.max(probs, axis=2)
        threshold = config.pmin
        connectivity = max_probs > threshold
        
        # Build graph and find connected components
        G = nx.Graph()
        for i in range(len(motl.df)):
            G.add_node(i)
        rows, cols = np.where(connectivity)
        for i, j in zip(rows, cols):
            if i < j:
                G.add_edge(i, j, weight=max_probs[i, j])
        
        connected_components = list(nx.connected_components(G))
        largest_component = max(connected_components, key=len)
        
        # Save connections
        connections = {}
        for i, j in zip(rows, cols):
            if i < j and connectivity[i, j]:
                connections[(i, j)] = {
                    'probability': float(max_probs[i, j]),
                    'case': int(np.argmax(probs[i, j, :]))
                }
        
        # Save connectivity dictionary
        dnal.save_connnections(
            connections,
            output_path_dictionary + f'Connectivity_motl_tomo{tomo_id}_cluster{cluster}.pickle'
        )
        
        # Save linker MOTL - convert connections format and call write_out_EmMolt_linker
        if connections:
            # Convert connections from {(i,j): {'probability': p, 'case': c}} to {i: [(j, p, c), ...]}
            connections_converted = {i: [] for i in range(len(motl.df))}
            for (i, j), data in connections.items():
                prob = data['probability']
                case = data['case']
                connections_converted[i].append((j, prob, case))
                # Add reverse connection with appropriate case
                # case 0 (entry-entry) <-> case 0 (entry-entry)
                # case 1 (exit-entry) <-> case 2 (entry-exit)
                # case 2 (entry-exit) <-> case 1 (exit-entry)
                # case 3 (exit-exit) <-> case 3 (exit-exit)
                case_map = {0: 0, 1: 2, 2: 1, 3: 3}
                connections_converted[j].append((i, prob, case_map[case]))
            
            # Call the proper write_out_EmMolt_linker function
            dnal.write_out_EmMolt_linker(
                connections=connections_converted,
                motl_exit=motl_exit,
                motl_entry=motl_entry,
                motl_entry2=motl_entry2,
                motl_exit2=motl_exit2,
                output_filename=output_path2 + f'motl_tomo{tomo_id}_cluster{cluster}_linkers.em',
                min_prob=config.pmin
            )
        
        print(f"    Cluster {cluster}: {len(largest_component)} particles in largest component")



#################################################################################
#################################################################################
#################################################################################
