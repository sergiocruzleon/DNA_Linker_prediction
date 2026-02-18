from . import main_dna_linkers as main
from . import utils_motlFiles as util_motl
from . import run_multiprocess as run
from . import config


def run_full_pipeline(
    path_mask: str,
    motl_name: str,
    entry: str,
    exit: str,
    origin_entry: str,
    origin_exit: str,
    path_output: str,
    motl_trace_input: str,
    tracing_distance: float,
    max_distance: float,
    output_path_cluster: str,
    output_path_linker: str,
    output_path_dictionary: str,
    dnal_object,
    lp_object,
    max_processes: int = 8
) -> None:
    """
    Runs the full linker tracing and clustering pipeline.

    Parameters:
        path_mask (str): Path to the folder containing entry/exit masks.
        motl_name (str): Filename of the original motive list to be recentered.
        entry (str): Entry mask filename.
        exit (str): Exit mask filename.
        origin_entry (str): Entry origin mask filename.
        origin_exit (str): Exit origin mask filename.
        path_output (str): Output path for recentered motive lists and trace.
        motl_trace_input (str): Output filename for traced motive list.
        tracing_distance (float): Tracing distance in nm.
        max_distance (float): Maximum linkage distance in voxel units.
        output_path_cluster (str): Output path for cluster-based lists.
        output_path_linker (str): Output path for linker-shifted motif lists.
        output_path_dictionary (str): Output path for linker connection dictionaries.
        dnal_object: Preloaded DNA linker object (e.g., dnal.lo).
        max_processes (int): Number of processes for parallel execution.

    Returns:
        None
    """

    # Step 1: Recentering and writing
    output_entry, output_exit, motl_all_path = util_motl.recenter_and_write_motl(
        path_mask=path_mask,
        motl_name=motl_name,
        entry_mask_name=entry,
        exit_mask_name=exit,
        path_output=path_output
    )

    # Step 2: Tracing and annotation
    util_motl.trace_and_annotate_motl(
        output_entry=output_entry,
        output_exit=output_exit,
        path_mask=path_mask,
        motl_name=motl_name,
        path_output=path_output,
        motl_trace_input=motl_trace_input,
        tracing_distance=tracing_distance,
        max_distance=max_distance
    )

    # Step 3: Create lists per tomogram cluster
    util_motl.create_motllists_perTomo_perCluster(
        motl_trace_input=motl_trace_input,
        output_path=output_path_cluster
    )

    util_motl.generate_motif_lists_per_cluster(
        motl_trace_input=motl_trace_input,
        output_path_cluster=output_path_cluster,
        path_mask=path_mask,
        origin_entry=origin_entry,
        entry=entry,
        origin_exit=origin_exit,
        exit=exit
    )

    util_motl.create_shifted_motif_lists_along_linker(
        motl_trace_input=motl_trace_input,
        output_path_cluster=output_path_cluster,
        path_mask=path_mask,
        origin_entry=origin_entry,
        entry=entry,
        origin_exit=origin_exit,
        exit=exit
    )

    # Step 4: Parallel cluster linker generation
    run.run_cluster_processing(
        motl_trace_input=motl_trace_input,
        output_path_cluster=output_path_cluster,
        output_path_linker=output_path_linker,
        output_path_dictionary=output_path_dictionary,
        dnal_object=dnal_object,
        lp_object=lp_object,
        max_processes=max_processes,
        batch_gpu=config.batch_gpu
    )
