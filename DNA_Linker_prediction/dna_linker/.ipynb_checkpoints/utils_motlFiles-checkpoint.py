import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# CryoCAT
from cryocat import cryomotl
from cryocat import cryomap
from cryocat import ribana as ra

def determine_origin_and_tangent_vector(mask_list, epsilon = 0.00001 ):
    """
    Takes a list of two masks at the "Origin" and final of the DNA (starting) in the average. In this order.
    This method returns the position of the origin and the tangent vector normalized
    ---
    Inputs: List of paths to the masks.

    ----
    Returns 
    coordinates_origin, 
    norm_vector_exit
    
    """
    coordinates_origin=np.array([0,0,0])
    vector_exit=np.array([0,0,0])
    if len(mask_list)!=2:
        print('The parameter mask_list must contain two masks: the "Origin" and final end of the DNA, in this order.')
       
    else:
        for idx in range(len(mask_list)):
            mask = cryomap.read(mask_list[idx])
            mask_size = np.array(mask.shape)
            # new_size = np.repeat(size_list[el], 3)
            old_center = mask_size / 2
            # new_center = new_size / 2
            
            # find center of mask
            i, j, k = np.asarray(mask > epsilon).nonzero()
            s = np.array([min(i), min(j), min(k)])
            e = np.array([max(i), max(j), max(k)])
            mask_center = (s + e) / 2
            mask_center += 1  # account for python 0 based indices
        
            if idx==0:
                coordinates_origin=mask_center
            elif idx==1:
                vector_exit=mask_center-coordinates_origin
            
        norm_vector_exit=vector_exit/np.linalg.norm(vector_exit)
        return  coordinates_origin, norm_vector_exit, old_center



def create_motllists_perTomo_perCluster(motl_trace_input,output_path):
    """
    motl_trace_input,
    output_path
    """
    motl_trace=cryomotl.EmMotl(input_motl=motl_trace_input)
    tomograms=motl_trace.df['tomo_id'].unique()
    for tomo_id in tomograms:
        df_motl_tomo=motl_trace.df[motl_trace.df['tomo_id']==tomo_id]
        clusters=df_motl_tomo['geom1'].unique()
        for cluster in clusters:
            df_motl_tomo_cluster=df_motl_tomo[df_motl_tomo['geom1']==cluster]
            
            motl_tomo_cluster=cryomotl.Motl(motl_df=df_motl_tomo_cluster)
            motl_tomo_cluster.renumber_particles()
            motl_tomo_cluster.write_out(output_path+f'motl_tomo{tomo_id}_cluster{cluster}.em')
            print(f'The file: ', output_path+f'motl_tomo{tomo_id}_cluster{cluster}.em', 'has been saved!')



def create_new_motl_per_case(input_motl,
                         masks_path_origin_case, 
                         masks_path_case,
                         output_filename):
    """
    The cases can be: entry, entry2, exit and exit2, and the masks should be as follow:
    Entry:
    mask_exit=[masks_path+origin_exit, masks_path+exit]
    Entry2
    mask_entry=[masks_path+entry, masks_path+origin_entry]
    
    Exit
    mask_exit=[masks_path+origin_exit, masks_path+exit]
    
    Exit2
    mask_exit=[masks_path+exit, masks_path+origin_exit]  
    """
    # Load motl
    motl=cryomotl.EmMotl(input_motl=input_motl)
    
    ## Calculate the origin and vector at the entry of the nucleosome
    masks=[masks_path_origin_case, masks_path_case]
    coordinates_entry, norm_vector_entry, old_center=determine_origin_and_tangent_vector(mask_list=masks)
    
    # Apply shift to the coordinates - Now the particle is centered at the position of the entry point of the DNA
    shifts = coordinates_entry - old_center
    motl.shift_positions(shifts)
    motl.update_coordinates()
    
    new_motl=motl
    
    # Save shifted particles
    new_motl.renumber_particles()
    new_motl.write_out(output_filename)



def create_motiflists_for_entry_exit(output_path, tomo_id, cluster,
                           masks_path, mask_origin_entry, mask_entry,
                            mask_origin_exit, mask_exit, prefix=''):
    """
    Create 4 new motif lists starting from the motif list for the tomo_id and the cluster.
    The 4 new lists correspond to the entry, entry2, exit, exit2 (See Figure XX). 
    """
    output_paths=[]
    # Entry
    path_motl=output_path+f'motl_tomo{tomo_id}_cluster{cluster}.em'
    masks_path_origin_entry=masks_path+mask_origin_entry 
    masks_path_entry=masks_path+mask_entry
    output_filename=output_path+prefix+f'motl_tomo{tomo_id}_cluster{cluster}_entry.em'
    
    create_new_motl_per_case(input_motl=path_motl,
                             masks_path_origin_case=masks_path_origin_entry, 
                             masks_path_case=masks_path_entry,
                             output_filename=output_filename)
    output_paths.append(output_filename)
    # Entry2
    path_motl=output_path+f'motl_tomo{tomo_id}_cluster{cluster}.em'
    masks_path_origin_entry=masks_path+mask_entry
    masks_path_entry=masks_path+mask_origin_entry 
    output_filename=output_path+prefix+f'motl_tomo{tomo_id}_cluster{cluster}_entry2.em'
    
    create_new_motl_per_case(input_motl=path_motl,
                             masks_path_origin_case=masks_path_origin_entry, 
                             masks_path_case=masks_path_entry,
                             output_filename=output_filename)
    output_paths.append(output_filename)
    
    # Exit
    path_motl=output_path+f'motl_tomo{tomo_id}_cluster{cluster}.em'
    masks_path_origin_exit=masks_path+mask_origin_exit
    masks_path_exit=masks_path+mask_exit
    output_filename=output_path+prefix+f'motl_tomo{tomo_id}_cluster{cluster}_exit.em'
    
    create_new_motl_per_case(input_motl=path_motl,
                             masks_path_origin_case=masks_path_origin_exit, 
                             masks_path_case=masks_path_exit,
                             output_filename=output_filename)
    output_paths.append(output_filename)
    
    # Exit2
    path_motl=output_path+f'motl_tomo{tomo_id}_cluster{cluster}.em'
    masks_path_origin_exit=masks_path+mask_exit
    masks_path_exit=masks_path+mask_origin_exit
    output_filename=output_path+prefix+f'motl_tomo{tomo_id}_cluster{cluster}_exit2.em'
    
    create_new_motl_per_case(input_motl=path_motl,
                             masks_path_origin_case=masks_path_origin_exit, 
                             masks_path_case=masks_path_exit,
                             output_filename=output_filename)
    output_paths.append(output_filename)
    return output_paths


def create_shifted_motiflists_along_linker(output_path, path_motl, 
                           masks_path, mask_origin_entry, mask_entry,
                            mask_origin_exit, mask_exit, prefix=''):
    """
    Create 4 new motif lists starting from the motif list for the tomo_id and the cluster.
    The 4 new lists correspond to the entry, entry2, exit, exit2 (See Figure XX). 
    """
    output_paths=[]
    # Entry
    
    masks_path_origin_entry=masks_path+mask_origin_entry 
    masks_path_entry=masks_path+mask_entry
    output_filename=output_path+prefix+f'motl_entry.em'
    
    create_new_motl_per_case(input_motl=path_motl,
                             masks_path_origin_case=masks_path_origin_entry, 
                             masks_path_case=masks_path_entry,
                             output_filename=output_filename)
    output_paths.append(output_filename)
    # Entry2
    
    masks_path_origin_entry=masks_path+mask_entry
    masks_path_entry=masks_path+mask_origin_entry 
    output_filename=output_path+prefix+f'motl_entry2.em'
    
    create_new_motl_per_case(input_motl=path_motl,
                             masks_path_origin_case=masks_path_origin_entry, 
                             masks_path_case=masks_path_entry,
                             output_filename=output_filename)
    output_paths.append(output_filename)
    
    # Exit
    
    masks_path_origin_exit=masks_path+mask_origin_exit
    masks_path_exit=masks_path+mask_exit
    output_filename=output_path+prefix+f'motl_exit.em'
    
    create_new_motl_per_case(input_motl=path_motl,
                             masks_path_origin_case=masks_path_origin_exit, 
                             masks_path_case=masks_path_exit,
                             output_filename=output_filename)
    output_paths.append(output_filename)
    
    # Exit2
    
    masks_path_origin_exit=masks_path+mask_exit
    masks_path_exit=masks_path+mask_origin_exit
    output_filename=output_path+prefix+f'motl_exit2.em'
    
    create_new_motl_per_case(input_motl=path_motl,
                             masks_path_origin_case=masks_path_origin_exit, 
                             masks_path_case=masks_path_exit,
                             output_filename=output_filename)
    output_paths.append(output_filename)
    return output_paths




def recenter_and_write_motl(
    path_mask: str,
    motl_name: str,
    entry_mask_name: str,
    exit_mask_name: str,
    path_output: str,
    output_entry_name: str = 'entry_b1_Trest_run_data_righthanded.em',
    output_exit_name: str = 'exit_b1_Trest_run_data_righthanded.em'
) -> tuple[str, str, str]:
    """
    Recenters a cryo-EM motive list to subparticle centers using provided entry and exit masks,
    and writes the shifted motive lists to output files.

    Parameters:
        path_mask (str): Directory path where the motive list and masks are located.
        motl_name (str): Filename of the motive list to be loaded.
        entry_mask_name (str): Filename of the mask used to recenter the 'entry' subparticles.
        exit_mask_name (str): Filename of the mask used to recenter the 'exit' subparticles.
        path_output (str): Directory path where the output .em files will be saved.
        output_entry_name (str): Filename for the recentered entry output.
        output_exit_name (str): Filename for the recentered exit output.

    Returns:
        tuple[str, str, str]: Tuple of paths:
            (output_entry_path, output_exit_path, full_motl_path)
    """
    # Load the full motive list
    motl_all = cryomotl.EmMotl(path_mask + motl_name)

    # Create full paths to the entry and exit masks
    mask_entry = path_mask + entry_mask_name
    mask_exit = path_mask + exit_mask_name

    # Recenter the motive list to the subparticle centers defined by the masks
    sh_motl_entry = cryomotl.Motl.recenter_to_subparticle(motl_all.df, mask_entry)
    sh_motl_exit = cryomotl.Motl.recenter_to_subparticle(motl_all.df, mask_exit)

    # Construct full output file paths
    output_entry = path_output + output_entry_name
    output_exit = path_output + output_exit_name

    # Write out the recentered motive lists
    sh_motl_entry.write_out(output_entry)
    sh_motl_exit.write_out(output_exit)

    # Return paths for downstream use
    return output_entry, output_exit, path_mask + motl_name




def trace_and_annotate_motl(
    output_entry: str,
    output_exit: str,
    path_mask:str,
    motl_name: str,
    path_output: str,
    tracing_distance: int,
    max_distance: float,
    motl_trace_input: str
) -> None:
    """
    Traces chains between entry and exit motive lists, sorts and annotates the results,
    and saves the traced and annotated motive lists.

    Parameters:
        output_entry (str): Path to the previously recentered entry motive list.
        output_exit (str): Path to the previously recentered exit motive list.
        motl_all_path (str): Path to the original, full motive list (.em file or dataframe).
        path_output (str): Directory path to save the traced motive list files.
        tracing_distance (int): Distance threshold (in nm) used for naming the output files.
        max_distance (float): Maximum distance allowed between linked particles when tracing.
        motl_trace_input (str): Output filename for the traced and annotated full motive list.

    Returns:
        None
    """
    # Load the full motive list
    motl_all = cryomotl.EmMotl(path_mask + motl_name)
    
    # Trace chains between the entry and exit particles
    traced_motl = ra.trace_chains(output_entry, output_exit, max_distance=max_distance, min_distance=0)

    # Sort the resulting DataFrame for consistency
    traced_motl.df.sort_values(['tomo_id', 'object_id', 'geom2'], inplace=True)

    # Define output file names based on the tracing distance
    entry_motl_traced = path_output + f'entry_b1_Trest_run_data_righthanded_tr{tracing_distance}nm.em'
    exit_motl_traced = path_output + f'exit_b1_Trest_run_data_righthanded_tr{tracing_distance}nm.em'
    output_motl_traced = motl_trace_input

    # Annotate the traced motive list with occupancy and traced info
    ra.add_occupancy(traced_motl)
    ra.add_traced_info(traced_motl, output_entry, entry_motl_traced)
    ra.add_traced_info(traced_motl, output_exit, exit_motl_traced)
    ra.add_traced_info(traced_motl, motl_all, output_motl_traced)



def generate_motif_lists_per_cluster(
    motl_trace_input: str,
    output_path_cluster: str,
    path_mask: str,
    origin_entry: str,
    entry: str,
    origin_exit: str,
    exit: str,
    prefix: str = 'All_'
) -> None:
    """
    Generates motif lists for each tomogram and cluster from a traced motive list.
    
    Parameters:
        motl_trace_input (str): Path to the traced motive list (.em file).
        output_path_cluster (str): Directory to store generated motif lists.
        path_mask (str): Directory containing the masks.
        origin_entry (str): Path to the entry mask origin file.
        entry (str): Filename of the entry mask.
        origin_exit (str): Path to the exit mask origin file.
        exit (str): Filename of the exit mask.
        prefix (str): Prefix to use for naming output files (default: 'All_').

    Returns:
        None
    """
    # Load the traced motive list
    motl_trace = cryomotl.EmMotl(input_motl=motl_trace_input)

    # Get unique tomogram IDs
    tomograms = motl_trace.df['tomo_id'].unique()

    for tomo_id in tomograms:
        # Filter DataFrame for this tomogram
        df_motl_tomo = motl_trace.df[motl_trace.df['tomo_id'] == tomo_id]

        # Get unique clusters in this tomogram
        clusters = df_motl_tomo['geom1'].unique()

        for cluster in clusters:
            # Create motif lists for entry and exit for each cluster
            output_paths = create_motiflists_for_entry_exit(
                output_path=output_path_cluster,
                tomo_id=tomo_id,
                cluster=cluster,
                masks_path=path_mask,
                mask_origin_entry=origin_entry,
                mask_entry=entry,
                mask_origin_exit=origin_exit,
                mask_exit=exit,
                prefix=prefix
            )



def create_shifted_motif_lists_along_linker(
    motl_trace_input: str,
    output_path_cluster: str,
    path_mask: str,
    origin_entry: str,
    entry: str,
    origin_exit: str,
    exit: str,
    prefix: str = 'All_'
) -> list:
    """
    Creates motif lists by shifting along the linker between entry and exit masks
    for each traced object in the motive list.

    Parameters:
        motl_trace_input (str): Path to the traced motive list (.em file).
        output_path_cluster (str): Directory where output motif lists will be saved.
        path_mask (str): Directory containing the entry and exit masks.
        origin_entry (str): Path to the entry mask origin file.
        entry (str): Filename of the entry mask.
        origin_exit (str): Path to the exit mask origin file.
        exit (str): Filename of the exit mask.
        prefix (str): Prefix for output filenames (default: 'All_').

    Returns:
        list: List of paths to the generated motif list files.
    """
    # Load the traced motive list
    motl_trace = cryomotl.EmMotl(input_motl=motl_trace_input)

    # Generate motif lists along the linker and return output paths
    output_paths = create_shifted_motiflists_along_linker(
        output_path=output_path_cluster,
        path_motl=motl_trace_input,
        masks_path=path_mask,
        mask_origin_entry=origin_entry,
        mask_entry=entry,
        mask_origin_exit=origin_exit,
        mask_exit=exit,
        prefix=prefix
    )

    return output_paths
