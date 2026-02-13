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

# Muliprocessing
from multiprocessing import Pool

import sys


def run_cluster_processing(
    motl_trace_input: str,
    output_path_cluster: str,
    output_path_linker: str,
    output_path_dictionary: str,
    dnal_object,
    max_processes: int = 10
) -> None:
    """
    Runs parallel cluster-based processing on a traced motive list.

    Parameters:
        motl_trace_input (str): Path to the traced motive list (.em file).
        output_path_cluster (str): Directory to store per-cluster outputs.
        output_path_linker (str): Directory for shifted linker outputs.
        output_path_dictionary (str): Directory for intermediate dictionary data.
        dnal_object: An object (likely DNA linker handler) passed into each task.
        max_processes (int): Max number of worker processes to use (default: 10).

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
                dnal_object
            ])

    # Limit worker processes to avoid oversubscription
    num_processes = min(max_processes, len(packed_paths))

    # Run tasks in parallel
    with Pool(num_processes) as p:
        p.map(main.main_function, packed_paths)



#################################################################################
#################################################################################
#################################################################################
