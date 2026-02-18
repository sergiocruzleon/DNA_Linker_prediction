import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import matplotlib
import os

import warnings
warnings.filterwarnings('ignore')
from cryocat import cryomotl
from cryocat import cryomap
import pandas as pd
import networkx as nx
import pickle

# PyTorch for GPU acceleration (optional)
try:
    import torch
    TORCH_AVAILABLE = True
    # Check for GPU availability
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

# joblib for parallel processing
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    # Define no-op decorators if joblib is not available
    def Parallel(*args, **kwargs):
        raise ImportError("joblib is required for parallel processing. Install with: pip install joblib")
    
    def delayed(func):
        return func

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
#####################
# FUNCTIONS
#####################

def bending_energy(L, lp, theta, kB, T):
    """
    
    """
    return (2*lp)/(L)*(theta/2)**2*kB*T

def prob_bending_energy(L, theta, lp=lp):
    """
    
    """
    #return np.exp(-bending_energy(L=L, lp=lp, theta=theta, kB=kB, T=T)/(kB*T))
    return np.exp(-((2*lp)/(L))*(theta)**2)

def prob_linker_length(L, lo=10):
    """
    Exponential decay of the probability as a function of the length of the linker
    L (float): length of the linker
    lo (float): Expected average length
    """
    return np.exp(-L/lo)


def _vectorized_angle_between_vectors(v1, v2):
    """
    Vectorized angle calculation between arrays of vectors.
    
    Args:
        v1: (..., 3) array of vectors
        v2: (..., 3) array of vectors
        
    Returns:
        Angles in radians for each pair of vectors.
    """
    # Compute dot products
    dot = np.sum(v1 * v2, axis=-1)
    # Compute norms
    norm1 = np.linalg.norm(v1, axis=-1)
    norm2 = np.linalg.norm(v2, axis=-1)
    # Compute cosine with clipping to avoid numerical issues
    cos_theta = dot / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def _vectorized_probability_matrix(pos_a, vec_a, pos_b, vec_b, lo, lp):
    """
    Compute probability matrix for all pairs between two sets of particles.
    
    Args:
        pos_a: (N, 3) array of positions for particles A
        vec_a: (N, 3) array of direction vectors for particles A
        pos_b: (M, 3) array of positions for particles B
        vec_b: (M, 3) array of direction vectors for particles B
        lo: contour length parameter
        lp: persistence length parameter
        
    Returns:
        (N, M) probability matrix
    """
    # Broadcasting: pos_a[:, None, :] - pos_b[None, :, :] gives (N, M, 3)
    # This is the connecting vectors from each A to each B
    diff = pos_a[:, None, :] - pos_b[None, :, :]
    
    # Distances: norm along the last axis
    distances = np.linalg.norm(diff, axis=-1)
    
    # Normalized connecting vectors
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        connecting = diff / distances[:, :, None]
        connecting = np.nan_to_num(connecting, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Angles between connecting vectors and particle vectors
    # For particle A: angle with -vec_a (incoming direction)
    # For particle B: angle with vec_b (outgoing direction)
    angle_a = _vectorized_angle_between_vectors(connecting, -vec_a[:, None, :])
    angle_b = _vectorized_angle_between_vectors(connecting, vec_b[None, :, :])
    
    # Average bending angle
    theta_half = (angle_a + angle_b) / 2.0
    
    # Compute probabilities
    # P = P(length) * P(bending)
    prob_length = prob_linker_length(distances, lo=lo)
    prob_bend = prob_bending_energy(distances, theta_half, lp=lp)
    
    return prob_length * prob_bend


def calculate_probabilities(pos_selected, vector_selected, pos_current, vector_current, lo=500 / 1.971, lp=lp):
    """
    Calculate the probability of connection between two particles.

    Args:
    - pos_selected (numpy.ndarray): Position of the selected particle.
    - vector_selected (numpy.ndarray): Direction vector of the selected particle.
    - pos_current (numpy.ndarray): Position of the current particle.
    - vector_current (numpy.ndarray): Direction vector of the current particle.
    - lo: contour length parameter
    - lp: persistence length parameter

    Returns:
    - probability (float): Probability of connection between the particles based on the bending energy.
    """
    # Tangent vector between the current and selected particles
    vector_connecting = pos_current - pos_selected
    vector_connecting = vector_connecting / np.linalg.norm(vector_connecting)

    # Determines the angles between the vectors and the connecting vector 
    angle_current_connecting = calculate_angle(vector_connecting, -vector_current)
    angle_selected_connecting = calculate_angle(vector_connecting, vector_selected)
    
    #Asumming that the DNA bends uniformly, the bending angle theta_half_entry is given by:
    theta_half = (angle_current_connecting + angle_selected_connecting) / 2.
    
    #Distance between particles
    distance = calculate_distance(pos_selected, pos_current)
    
    probability = prob_linker_length(L=distance, lo=lo) * prob_bending_energy(L=distance, theta=theta_half, lp=lp)

    return probability   


def calculate_distance(v1, v2):
    """
    Calculates the Euclidean distance between two 3D points (vectors).
    
    Parameters:
    v1 (numpy.ndarray): First 3D vector.
    v2 (numpy.ndarray): Second 3D vector.
    
    Returns:
    float: The Euclidean distance between v1 and v2.
    """
    return np.linalg.norm(v2 - v1)

def calculate_angle(v1, v2):
    """
    Calculates the angle (in rad) between two 3D vectors.
    
    Parameters:
    v1 (numpy.ndarray): First 3D vector.
    v2 (numpy.ndarray): Second 3D vector.
    
    Returns:
    float: The angle (in rad) between v1 and v2.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    return np.arccos(cos_theta)


def define_case2(case):
    case2=0
    if case==0 or case ==3:
        case2 = case
        return case2
    elif case==1:
        case2=2
        return case2
    elif case==2:
        case2=1
        return case2
    else:
        print('Something is wrong. The cases should be an integer between 0-3')
    return case2

def find_next_maximum(matrix):
    """
    Find the maximum value in the matrix and return its index.

    Parameters:
    - matrix (numpy.ndarray): The input matrix.

    Returns:
    - max_index (tuple): The index of the maximum value in the matrix.
    """
    max_index = np.unravel_index(np.argmax(matrix), matrix.shape)
    return max_index


def create_connection(connections, matrix, max_index, remaining_connections):
    """
    Create a connection between particles based on the maximum index.

    Parameters:
    - connections (dict): Dictionary containing particle connections.
    - max_index (tuple): The index of the maximum value in the matrix.

    Returns:
    - connections (dict): Updated dictionary of particle connections.
    """
    i, j, case = max_index
    # Probability
    max_value=matrix[max_index]
    case2=define_case2(case)
    # Check if there is already a connection between the particles
    if any(connection[0] == j for connection in connections[i]):
        return connections, remaining_connections
    if any(connection[0] == i for connection in connections[j]):
        return connections, remaining_connections
        
    # Check if the combination is valid and if there is no existing connection between the particles
    if (case == 0 and 0 in [c[2] for c in connections[i]]) or \
       (case == 1 and 1 in [c[2] for c in connections[i]]) or \
       (case == 2 and 2 in [c[2] for c in connections[i]]) or \
       (case == 3 and 3 in [c[2] for c in connections[i]]) or \
       (case == 0 and 1 in [c[2] for c in connections[i]]) or \
       (case == 1 and 0 in [c[2] for c in connections[i]]) or \
       (case == 0 and 2 in [c[2] for c in connections[i]]) or \
       (case == 2 and 0 in [c[2] for c in connections[i]]) or \
       (case == 2 and 3 in [c[2] for c in connections[i]]) or \
       (case == 3 and 2 in [c[2] for c in connections[i]]) or \
       (case == 1 and 3 in [c[2] for c in connections[i]]) or \
       (case == 3 and 1 in [c[2] for c in connections[i]]) or \
       (case2 == 0 and 0 in [c[2] for c in connections[j]]) or \
       (case2 == 1 and 1 in [c[2] for c in connections[j]]) or \
       (case2 == 2 and 2 in [c[2] for c in connections[j]]) or \
       (case2 == 3 and 3 in [c[2] for c in connections[j]]) or \
       (case2 == 0 and 1 in [c[2] for c in connections[j]]) or \
       (case2 == 1 and 0 in [c[2] for c in connections[j]]) or \
       (case2 == 0 and 2 in [c[2] for c in connections[j]]) or \
       (case2 == 2 and 0 in [c[2] for c in connections[j]]) or \
       (case2 == 2 and 3 in [c[2] for c in connections[j]]) or \
       (case2 == 3 and 2 in [c[2] for c in connections[j]]) or \
       (case2 == 1 and 3 in [c[2] for c in connections[j]]) or \
       (case2 == 3 and 1 in [c[2] for c in connections[j]]) or \
       (remaining_connections[i] == 0) or \
       (remaining_connections[j] == 0):
        
        return connections, remaining_connections
    # Add connection to the dictionary
    if remaining_connections[i] > 0 and remaining_connections[j] > 0:
        connections[i].append((j, max_value, case))

        
        connections[j].append((i, max_value, (case2)))  # Assign opposite case for the second node
        remaining_connections[i] -= 1
        remaining_connections[j] -= 1
    
    return connections,remaining_connections



def calculate_linker_length_connected (connections, motl_exit, motl_entry, p_min=0.1):
    """
    Calculate the linker length of particles that were predicted to be connected
    """
    distances=[]
    counter=0
    for i in connections.keys():
        for conex in connections[i]:
            j, prob, case= conex[0],conex[1],conex[2]
            if prob>=p_min:
                # Check if the connection has already been processed in the opposite direction
                
                    counter=counter+1
                    if case==0: # entry-entry
                        particle1=motl_entry.df.iloc[i]
                        particle2=motl_entry.df.iloc[j]
                    elif case==1: # exit-entry     
                        particle1=motl_exit.df.iloc[i]
                        particle2=motl_entry.df.iloc[j]
                    elif case==2: # entry-exit
                        particle1=motl_entry.df.iloc[i]
                        particle2=motl_exit.df.iloc[j]
                    elif case==3: # exit-exit
                        particle1=motl_exit.df.iloc[i]
                        particle2=motl_exit.df.iloc[j]
                    else:
                        print(f'Something went WRONG! check the cases assingment')
                    
                ###############################################
                    pos_current=np.array([particle1['x'], particle1['y'], particle1['z']])
                    pos_selected=np.array([particle2['x'], particle2['y'], particle2['z']])
        
                    vector_connecting=pos_current-pos_selected
                    #print(f'The distance between nucleosonme {i} and {j} is {np.linalg.norm(vector_connecting)} voxels')
                    distances.append(np.linalg.norm(vector_connecting))
    return np.array(distances)
                    #vector_connecting=vector_connecting/np.linalg.norm(vector_connecting)



def save_connnections(connections, outputfilename):
    # Save dictionary to file
    with open(outputfilename, 'wb') as f:
        pickle.dump(connections, f)

def load_connnections(outputfilename):
    with open(outputfilename, 'rb') as f:
        connections = pickle.load(f)
    return connections



def write_out_EmMolt_linker(connections, motl_exit, motl_entry, 
                            motl_exit2, motl_entry2, 
                            output_filename, min_prob=pmin):
    df2=pd.DataFrame()
    counter=0
    # Create an empty set to keep track of processed connections
    processed_connections = set()
    for i in connections.keys():
        for conex in connections[i]:
            j, prob, case= conex[0],conex[1],conex[2]
            if prob>=min_prob:
                # Check if the connection has already been processed in the opposite direction
                if (j, i) not in processed_connections:        
                    counter=counter+1
                    if case==0: # entry-entry
                        particle1=motl_entry.df.iloc[i]
                        particle1_1=motl_entry2.df.iloc[i]
                        
                        particle2=motl_entry.df.iloc[j]
                        particle2_1=motl_entry2.df.iloc[j]
                    elif case==1: # exit-entry     
                        particle1=motl_exit.df.iloc[i]
                        particle1_1=motl_exit2.df.iloc[i]
                        
                        particle2=motl_entry.df.iloc[j]
                        particle2_1=motl_entry2.df.iloc[j]
                        
                    elif case==2: # entry-exit
                        particle1=motl_entry.df.iloc[i]
                        particle1_1=motl_entry2.df.iloc[i]
                        
                        particle2=motl_exit.df.iloc[j]
                        particle2_1=motl_exit2.df.iloc[j]
                        
                    elif case==3: # exit-exit
                        particle1=motl_exit.df.iloc[i]
                        particle1_1=motl_exit2.df.iloc[i]
                        
                        particle2=motl_exit.df.iloc[j]
                        particle2_1=motl_exit2.df.iloc[j]
                    else:
                        print(f'Something went WRONG! check the cases assingment')
                    
                ###############################################
                    pos_current=np.array([particle1['x'], particle1['y'], particle1['z']])
                    pos_current_1=np.array([particle1_1['x'], particle1_1['y'], particle1_1['z']])
                    
                    pos_selected=np.array([particle2['x'], particle2['y'], particle2['z']])
                    pos_selected_1=np.array([particle2_1['x'], particle2_1['y'], particle2_1['z']])

                    vector_current = (pos_current_1 - pos_current) / np.linalg.norm(pos_current_1 - pos_current)
                    vector_selected = (pos_selected_1 - pos_selected) / np.linalg.norm(pos_selected_1 - pos_selected)
                    
                    ####### Determine the connecting vector
                    vector_connecting=pos_current-pos_selected
                    linker_lenght=np.linalg.norm(vector_connecting) # lenght of the linker
                    
                    vector_connecting=vector_connecting/np.linalg.norm(vector_connecting)
                    
                    # Change to include the linker lenght
                    

                    ### Determine the bending angle
                    angle_current_connecting = calculate_angle(vector_connecting, -vector_current)
                    angle_selected_connecting = calculate_angle(vector_connecting, vector_selected)
    
                    #Asumming that the DNA bends uniformly, the bending angle theta_half_entry is given by:
                    theta_half = (angle_current_connecting + angle_selected_connecting)/2.
                    theta_deg=np.rad2deg(theta_half)
                    #print(f'Linker length {linker_lenght}, and angle {theta_deg}')
        ##############################################
                    
                    new_coord=(pos_current+pos_selected)/2
                    
                    new_linker = motl_entry.df.loc[1].copy()
        
                    new_linker['geom1']=0
                    new_linker['geom2']	=0
                    new_linker['subtomo_id']=counter	
                    new_linker['object_id']=counter
                    new_linker['subtomo_mean']=0	
                    new_linker['shift_x']=0
                    new_linker['shift_y']=0
                    new_linker['shift_z']=0	
                    new_linker['geom3']	=0
                    # Linker length in voxels: For now, only the end-to-end distance. Angle is not used.
                    # L=r*\theta= \theta* (D/2*sin(theta/2)) - Eq. (2) Kreysing, Cruz-Leon, Betz et al.
                    new_linker['geom4'] =theta_half*linker_lenght/(np.sin(theta_half))
                    # Linker length in degrees
                    new_linker['geom5']	=theta_deg	
                    
                    
                    new_linker['x']=new_coord[0]
                    new_linker['y']=new_coord[1]
                    new_linker['z']=new_coord[2]
                    
                    new_linker['score']=prob
            
                    target_vector = vector_connecting
                    
                    # Calculate the rotation quaternion to rotate the z-axis to the target vector
                    z_axis = np.array([0, 0, 1])  # The z-axis
                    
                    # Calculate the rotation axis and angle
                    rotation_axis = np.cross(z_axis, target_vector)
                    # Normalize the rotation axis
                    rotation_axis=rotation_axis/np.linalg.norm(rotation_axis)
                    rotation_angle = np.arccos(np.dot(z_axis, target_vector) / (np.linalg.norm(z_axis) * np.linalg.norm(target_vector)))
                    
                    # Create a Rotation object from axis-angle representation
                    rotation2= Rotation.align_vectors( [target_vector], [z_axis])[0]
                    euler_angles=rotation2.as_euler('zxz',  degrees=True)
                    new_linker.loc[['phi', 'theta',  'psi']]=euler_angles
            
            
                    df2 = pd.concat([df2, new_linker.to_frame().T], ignore_index=True)
                    # Add to processed_connections the i,j combination
                    processed_connections.add((i, j))
                
    # Create a emMotl from the dataframe       
    if len(df2)>0:
        linkers_motl=cryomotl.Motl(df2)
            #Save file
        linkers_motl.write_out(output_filename)
    #print(processed_connections)
    #print(f'The motiflist with {len(df2)} linkers was written to the file: {output_filename}')




###### MAIN FUNCTION
###### MAIN FUNCTION

def calculate_probabilities_all_connexions(motl, motl_exit, motl_exit2, motl_entry, motl_entry2, lo=lo, lp=lp):
    """
    Calculate probabilities for all particle connections using vectorized operations.
    
    This is an optimized version that pre-extracts all positions and vectors
    to NumPy arrays, then uses broadcasting for O(N²) computations.
    
    Args:
        motl: Motl object with particle data
        motl_exit: Motl with exit positions
        motl_exit2: Motl with exit2 positions (offset for vector calculation)
        motl_entry: Motl with entry positions
        motl_entry2: Motl with entry2 positions (offset for vector calculation)
        lo: contour length parameter
        lp: persistence length parameter
        
    Returns:
        probs: (N, N, 4) array of probabilities for each particle pair and case:
               [:,:,0] = entry-entry
               [:,:,1] = exit-entry
               [:,:,2] = entry-exit
               [:,:,3] = exit-exit
    """
    num_particles = len(motl.df)
    probs = np.zeros((num_particles, num_particles, 4))
    
    # Pre-extract all positions and vectors to NumPy arrays (avoids repeated pandas access)
    # Shape: (num_particles, 3)
    pos_exit = motl_exit.df[['x', 'y', 'z']].values
    pos_exit2 = motl_exit2.df[['x', 'y', 'z']].values
    pos_entry = motl_entry.df[['x', 'y', 'z']].values
    pos_entry2 = motl_entry2.df[['x', 'y', 'z']].values
    
    # Compute direction vectors for all particles at once
    # Shape: (num_particles, 3)
    vec_exit = pos_exit2 - pos_exit
    vec_exit = vec_exit / np.linalg.norm(vec_exit, axis=1, keepdims=True)
    
    vec_entry = pos_entry2 - pos_entry
    vec_entry = vec_entry / np.linalg.norm(vec_entry, axis=1, keepdims=True)
    
    # Compute all 4 probability matrices using broadcasting
    # Case 0: entry-entry (pos_entry, vec_entry) -> (pos_entry, vec_entry)
    probs[:, :, 0] = _vectorized_probability_matrix(
        pos_a=pos_entry, vec_a=vec_entry,
        pos_b=pos_entry, vec_b=vec_entry,
        lo=lo, lp=lp
    )
    
    # Case 1: exit-entry (pos_exit, vec_exit) -> (pos_entry, vec_entry)
    probs[:, :, 1] = _vectorized_probability_matrix(
        pos_a=pos_exit, vec_a=vec_exit,
        pos_b=pos_entry, vec_b=vec_entry,
        lo=lo, lp=lp
    )
    
    # Case 2: entry-exit (pos_entry, vec_entry) -> (pos_exit, vec_exit)
    probs[:, :, 2] = _vectorized_probability_matrix(
        pos_a=pos_entry, vec_a=vec_entry,
        pos_b=pos_exit, vec_b=vec_exit,
        lo=lo, lp=lp
    )
    
    # Case 3: exit-exit (pos_exit, vec_exit) -> (pos_exit, vec_exit)
    probs[:, :, 3] = _vectorized_probability_matrix(
        pos_a=pos_exit, vec_a=vec_exit,
        pos_b=pos_exit, vec_b=vec_exit,
        lo=lo, lp=lp
    )
    
    # Set diagonal to 0 (same particle connection is not valid)
    probs[np.eye(num_particles, dtype=bool)] = 0.0
    
    return probs


#####################
# PARALLEL PROCESSING (joblib)
#####################

def _compute_single_probability(args):
    """
    Compute probability for a single particle pair (used for parallel processing).
    
    Args:
        args: Tuple of (i, j, case, pos_a, vec_a, pos_b, vec_b, lo, lp)
            
    Returns:
        tuple: (i, j, case, probability)
    """
    i, j, case, pos_a, vec_a, pos_b, vec_b, lo, lp = args
    if i == j:
        return (i, j, case, 0.0)
    
    # Get particle positions and vectors
    pos_a_i = pos_a[i]
    vec_a_i = vec_a[i]
    pos_b_j = pos_b[j]
    vec_b_j = vec_b[j]
    
    # Compute connecting vector
    diff = pos_a_i - pos_b_j
    distance = np.linalg.norm(diff)
    
    if distance < 1e-10:
        return (i, j, case, 0.0)
    
    connecting = diff / distance
    
    # Compute angles - same for all cases in this implementation
    connecting_reshaped = connecting.reshape(1, 3)
    angle_a = _vectorized_angle_between_vectors(
        connecting_reshaped, -vec_a_i.reshape(1, 3)
    )[0]
    angle_b = _vectorized_angle_between_vectors(
        connecting_reshaped, vec_b_j.reshape(1, 3)
    )[0]
    
    theta_half = (angle_a + angle_b) / 2.0
    
    # Compute probability
    prob_length = prob_linker_length(distance, lo=lo)
    prob_bend = prob_bending_energy(distance, theta_half, lp=lp)
    
    return (i, j, case, prob_length * prob_bend)


def calculate_probabilities_all_connexions_parallel(
    motl, motl_exit, motl_exit2, motl_entry, motl_entry2, 
    lo=lo, lp=lp, n_jobs=-1
):
    """
    Calculate probabilities for all particle connections using joblib parallelism.
    
    This is a parallel version that processes particle pairs across multiple cores.
    For small particle counts, the sequential version may be faster due to overhead.
    For large particle counts, this provides significant speedup.
    
    Args:
        motl: Motl object with particle data
        motl_exit: Motl with exit positions
        motl_exit2: Motl with exit2 positions (offset for vector calculation)
        motl_entry: Motl with entry positions
        motl_entry2: Motl with entry2 positions (offset for vector calculation)
        lo: contour length parameter
        lp: persistence length parameter
        n_jobs: Number of jobs for parallel processing (-1 for all cores)
            
    Returns:
        probs: (N, N, 4) array of probabilities for each particle pair and case:
               [:,:,0] = entry-entry
               [:,:,1] = exit-entry
               [:,:,2] = entry-exit
               [:,:,3] = exit-exit
    
    Raises:
        ImportError: If joblib is not installed
    """
    if not JOBLIB_AVAILABLE:
        raise ImportError(
            "joblib is required for parallel processing. "
            "Install with: pip install joblib"
        )
    
    num_particles = len(motl.df)
    probs = np.zeros((num_particles, num_particles, 4))
    
    # Pre-extract all positions and vectors to NumPy arrays
    pos_exit = motl_exit.df[['x', 'y', 'z']].values
    pos_exit2 = motl_exit2.df[['x', 'y', 'z']].values
    pos_entry = motl_entry.df[['x', 'y', 'z']].values
    pos_entry2 = motl_entry2.df[['x', 'y', 'z']].values
    
    # Compute direction vectors
    vec_exit = pos_exit2 - pos_exit
    vec_exit = vec_exit / np.linalg.norm(vec_exit, axis=1, keepdims=True)
    
    vec_entry = pos_entry2 - pos_entry
    vec_entry = vec_entry / np.linalg.norm(vec_entry, axis=1, keepdims=True)
    
    # Define position/vector pairs for each case
    case_configs = [
        (pos_entry, vec_entry, pos_entry, vec_entry),   # case 0: entry-entry
        (pos_exit, vec_exit, pos_entry, vec_entry),       # case 1: exit-entry
        (pos_entry, vec_entry, pos_exit, vec_exit),       # case 2: entry-exit
        (pos_exit, vec_exit, pos_exit, vec_exit),        # case 3: exit-exit
    ]
    
    # Process each case in parallel
    for case_idx, (pos_a, vec_a, pos_b, vec_b) in enumerate(case_configs):
        # Create tasks for all particle pairs (excluding diagonal)
        tasks = []
        for i in range(num_particles):
            for j in range(num_particles):
                if i != j:
                    tasks.append((i, j, case_idx, pos_a, vec_a, pos_b, vec_b, lo, lp))
        
        # Run in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_single_probability)(task) for task in tasks
        )
        
        # Fill in results
        for i, j, case, prob in results:
            probs[i, j, case] = prob
    
    return probs


def _compute_single_linker_length(args):
    """
    Compute linker length for a single connection (used for parallel processing).
    
    Args:
        args: Tuple of (connection_data, motl_exit, motl_entry, p_min)
            
    Returns:
        float: Linker length or None if invalid
    """
    i, j, prob, case, motl_exit, motl_entry, p_min = args
    
    if prob < p_min:
        return None
    
    try:
        if case == 0:  # entry-entry
            particle1 = motl_entry.df.iloc[i]
            particle2 = motl_entry.df.iloc[j]
        elif case == 1:  # exit-entry
            particle1 = motl_exit.df.iloc[i]
            particle2 = motl_entry.df.iloc[j]
        elif case == 2:  # entry-exit
            particle1 = motl_entry.df.iloc[i]
            particle2 = motl_exit.df.iloc[j]
        elif case == 3:  # exit-exit
            particle1 = motl_exit.df.iloc[i]
            particle2 = motl_exit.df.iloc[j]
        else:
            return None
        
        pos_current = np.array([particle1['x'], particle1['y'], particle1['z']])
        pos_selected = np.array([particle2['x'], particle2['y'], particle2['z']])
        
        vector_connecting = pos_current - pos_selected
        return np.linalg.norm(vector_connecting)
    except (IndexError, KeyError):
        return None


def calculate_linker_length_connected_parallel(connections, motl_exit, motl_entry, p_min=0.1, n_jobs=-1):
    """
    Calculate linker lengths for connected particles using joblib parallelism.
    
    Args:
        connections: Dictionary containing particle connections
        motl_exit: Motl with exit positions
        motl_entry: Motl with entry positions
        p_min: Minimum probability threshold
        n_jobs: Number of jobs for parallel processing (-1 for all cores)
            
    Returns:
        numpy.ndarray: Array of linker lengths
    
    Raises:
        ImportError: If joblib is not installed
    """
    if not JOBLIB_AVAILABLE:
        raise ImportError(
            "joblib is required for parallel processing. "
            "Install with: pip install joblib"
        )
    
    # Create tasks for all connections
    tasks = []
    for i in connections.keys():
        for conex in connections[i]:
            j, prob, case = conex[0], conex[1], conex[2]
            tasks.append((i, j, prob, case, motl_exit, motl_entry, p_min))
    
    # Run in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_single_linker_length)(task) for task in tasks
    )
    
    # Filter out None results and convert to array
    distances = [r for r in results if r is not None]
    return np.array(distances)


#####################
# GPU ACCELERATED FUNCTIONS
#####################

def _torch_angle_between_vectors(v1, v2):
    """
    Vectorized angle calculation between arrays of vectors using PyTorch.
    
    Args:
        v1: (..., 3) tensor of vectors
        v2: (..., 3) tensor of vectors
        
    Returns:
        Angles in radians for each pair of vectors.
    """
    # Compute dot products
    dot = torch.sum(v1 * v2, dim=-1)
    # Compute norms
    norm1 = torch.norm(v1, dim=-1)
    norm2 = torch.norm(v2, dim=-1)
    # Compute cosine with clipping to avoid numerical issues
    cos_theta = dot / (norm1 * norm2 + 1e-10)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    return torch.acos(cos_theta)


def _torch_prob_linker_length(L, lo=10):
    """
    Exponential decay of the probability as a function of the length of the linker.
    PyTorch version for GPU acceleration.
    
    Args:
        L: tensor of lengths
        lo: Expected average length
    
    Returns:
        Tensor of probabilities
    """
    return torch.exp(-L / lo)


def _torch_prob_bending_energy(L, theta, lp=lp):
    """
    Bending energy probability using PyTorch.
    
    Args:
        L: tensor of lengths
        theta: tensor of bending angles (half angles)
        lp: persistence length parameter
    
    Returns:
        Tensor of probabilities
    """
    return torch.exp(-((2 * lp) / L) * (theta ** 2))


def _torch_probability_matrix(pos_a, vec_a, pos_b, vec_b, lo, lp):
    """
    Compute probability matrix for all pairs between two sets of particles using PyTorch.
    
    Args:
        pos_a: (N, 3) tensor of positions for particles A
        vec_a: (N, 3) tensor of direction vectors for particles A
        pos_b: (M, 3) tensor of positions for particles B
        vec_b: (M, 3) tensor of direction vectors for particles B
        lo: contour length parameter
        lp: persistence length parameter
        
    Returns:
        (N, M) probability tensor
    """
    # Broadcasting: pos_a[:, None, :] - pos_b[None, :, :] gives (N, M, 3)
    diff = pos_a[:, None, :] - pos_b[None, :, :]
    
    # Distances: norm along the last axis
    distances = torch.norm(diff, dim=-1)
    
    # Normalized connecting vectors
    connecting = diff / (distances[:, :, None] + 1e-10)
    connecting = torch.nan_to_num(connecting, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Angles between connecting vectors and particle vectors
    angle_a = _torch_angle_between_vectors(connecting, -vec_a[:, None, :])
    angle_b = _torch_angle_between_vectors(connecting, vec_b[None, :, :])
    
    # Average bending angle
    theta_half = (angle_a + angle_b) / 2.0
    
    # Compute probabilities
    prob_length = _torch_prob_linker_length(distances, lo=lo)
    prob_bend = _torch_prob_bending_energy(distances, theta_half, lp=lp)
    
    return prob_length * prob_bend


def calculate_probabilities_gpu(motl, motl_exit, motl_exit2, motl_entry, motl_entry2, lo=lo, lp=lp):
    """
    Calculate probabilities for all particle connections using PyTorch GPU acceleration.
    
    This function uses PyTorch for matrix operations, which can be accelerated on:
    - NVIDIA GPUs via CUDA
    - Apple Silicon GPUs via MPS (Metal Performance Shaders)
    - Falls back to CPU if no GPU is available
    
    Args:
        motl: Motl object with particle data
        motl_exit: Motl with exit positions
        motl_exit2: Motl with exit2 positions (offset for vector calculation)
        motl_entry: Motl with entry positions
        motl_entry2: Motl with entry2 positions (offset for vector calculation)
        lo: contour length parameter
        lp: persistence length parameter
        
    Returns:
        probs: (N, N, 4) numpy array of probabilities for each particle pair and case:
               [:,:,0] = entry-entry
               [:,:,1] = exit-entry
               [:,:,2] = entry-exit
               [:,:,3] = exit-exit
    
    Raises:
        ImportError: If PyTorch is not installed
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for GPU acceleration. "
            "Install with: pip install torch"
        )
    
    # Determine device (cuda, mps, or cpu)
    device = DEVICE
    
    # If device is not available, fall back to CPU
    if device == 'mps':
        try:
            # Test MPS with a small tensor
            test_tensor = torch.tensor([1.0], device='mps')
            del test_tensor
            print(f"  Using MPS (Apple Silicon GPU) for acceleration...")
        except Exception as e:
            print(f"  Warning: MPS test failed ({e}), falling back to CPU")
            device = 'cpu'
    elif device == 'cuda':
        try:
            # Test CUDA with a small tensor
            test_tensor = torch.tensor([1.0]).cuda()
            del test_tensor
        except Exception as e:
            print(f"  Warning: CUDA test failed ({e}), falling back to CPU")
            device = 'cpu'
    
    num_particles = len(motl.df)
    
    # Pre-extract all positions and vectors to NumPy arrays
    pos_exit = motl_exit.df[['x', 'y', 'z']].values
    pos_exit2 = motl_exit2.df[['x', 'y', 'z']].values
    pos_entry = motl_entry.df[['x', 'y', 'z']].values
    pos_entry2 = motl_entry2.df[['x', 'y', 'z']].values
    
    # Compute direction vectors for all particles at once
    vec_exit = pos_exit2 - pos_exit
    vec_exit = vec_exit / np.linalg.norm(vec_exit, axis=1, keepdims=True)
    
    vec_entry = pos_entry2 - pos_entry
    vec_entry = vec_entry / np.linalg.norm(vec_entry, axis=1, keepdims=True)
    
    # For small matrices, CPU might be faster
    if num_particles < 100:
        print(f"  Particle count ({num_particles}) is small, using optimized CPU version")
        return calculate_probabilities_all_connexions(
            motl, motl_exit, motl_exit2, motl_entry, motl_entry2, lo=lo
        )
    
    # Try GPU, but fall back to CPU if there are issues
    try:
        # Convert to PyTorch tensors
        pos_exit_t = torch.from_numpy(pos_exit).float()
        pos_exit2_t = torch.from_numpy(pos_exit2).float()
        pos_entry_t = torch.from_numpy(pos_entry).float()
        pos_entry2_t = torch.from_numpy(pos_entry2).float()
        vec_exit_t = torch.from_numpy(vec_exit).float()
        vec_entry_t = torch.from_numpy(vec_entry).float()
        
        # Move to device
        if device in ('cuda', 'mps'):
            pos_exit_t = pos_exit_t.to(device)
            pos_exit2_t = pos_exit2_t.to(device)
            pos_entry_t = pos_entry_t.to(device)
            pos_entry2_t = pos_entry2_t.to(device)
            vec_exit_t = vec_exit_t.to(device)
            vec_entry_t = vec_entry_t.to(device)
        
        # Initialize probability tensor
        if device in ('cuda', 'mps'):
            probs_t = torch.zeros((num_particles, num_particles, 4), device=device)
        else:
            probs_t = torch.zeros((num_particles, num_particles, 4))
        
        # Case 0: entry-entry (pos_entry, vec_entry) -> (pos_entry, vec_entry)
        probs_t[:, :, 0] = _torch_probability_matrix(
            pos_a=pos_entry_t, vec_a=vec_entry_t,
            pos_b=pos_entry_t, vec_b=vec_entry_t,
            lo=lo, lp=lp
        )
        
        # Case 1: exit-entry (pos_exit, vec_exit) -> (pos_entry, vec_entry)
        probs_t[:, :, 1] = _torch_probability_matrix(
            pos_a=pos_exit_t, vec_a=vec_exit_t,
            pos_b=pos_entry_t, vec_b=vec_entry_t,
            lo=lo, lp=lp
        )
        
        # Case 2: entry-exit (pos_entry, vec_entry) -> (pos_exit, vec_exit)
        probs_t[:, :, 2] = _torch_probability_matrix(
            pos_a=pos_entry_t, vec_a=vec_entry_t,
            pos_b=pos_exit_t, vec_b=vec_exit_t,
            lo=lo, lp=lp
        )
        
        # Case 3: exit-exit (pos_exit, vec_exit) -> (pos_exit, vec_exit)
        probs_t[:, :, 3] = _torch_probability_matrix(
            pos_a=pos_exit_t, vec_a=vec_exit_t,
            pos_b=pos_exit_t, vec_b=vec_exit_t,
            lo=lo, lp=lp
        )
        
        # Set diagonal to 0 (same particle connection is not valid)
        if device in ('cuda', 'mps'):
            eye_mask = torch.eye(num_particles, device=device, dtype=torch.bool)
        else:
            eye_mask = torch.eye(num_particles, dtype=torch.bool)
        probs_t[eye_mask] = 0.0
        
        # Convert back to numpy array
        probs = probs_t.cpu().numpy()
        
    except Exception as e:
        print(f"  GPU computation failed ({e}), falling back to CPU")
        return calculate_probabilities_all_connexions(
            motl, motl_exit, motl_exit2, motl_entry, motl_entry2, lo=lo
        )
    
    return probs


def calculate_probabilities_batch_gpu(cluster_data_list, lo=150, lp=500):
    """
    Process multiple clusters in a single GPU batch for maximum throughput.
    
    This function batches multiple clusters together and processes them all on the GPU
    at once, which can be significantly faster than processing them individually.
    
    Args:
        cluster_data_list: List of tuples, each containing:
            (motl, motl_exit, motl_exit2, motl_entry, motl_entry2)
            Each element is a cryomotl object for one cluster
        lo: Contour length parameter
        lp: Persistence length parameter
    
    Returns:
        List of probability arrays, one per cluster
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GPU acceleration")
    
    import torch
    
    # Determine device
    device = DEVICE
    
    # Test device availability
    if device == 'mps':
        try:
            test_tensor = torch.tensor([1.0], device='mps')
            del test_tensor
        except Exception as e:
            print(f"  Warning: MPS test failed ({e}), falling back to CPU")
            device = 'cpu'
    elif device == 'cuda':
        try:
            test_tensor = torch.tensor([1.0]).cuda()
            del test_tensor
        except Exception as e:
            print(f"  Warning: CUDA test failed ({e}), falling back to CPU")
            device = 'cpu'
    
    if device == 'cpu':
        # Fall back to sequential CPU processing
        print("  Batched GPU processing failed, using sequential CPU...")
        results = []
        for motl, motl_exit, motl_exit2, motl_entry, motl_entry2 in cluster_data_list:
            probs = calculate_probabilities_all_connexions(
                motl, motl_exit, motl_exit2, motl_entry, motl_entry2, lo=lo
            )
            results.append(probs)
        return results
    
    print(f"  Processing {len(cluster_data_list)} clusters in batch on {device}...")
    
    # Get max cluster size for padding
    max_size = 0
    cluster_sizes = []
    for motl, motl_exit, motl_exit2, motl_entry, motl_entry2 in cluster_data_list:
        size = len(motl.df)
        cluster_sizes.append(size)
        max_size = max(max_size, size)
    
    padded_size = max_size
    
    # Prepare batched tensors
    batch_size = len(cluster_data_list)
    
    # Create padded tensors on device
    if device in ('cuda', 'mps'):
        probs_batch = torch.zeros((batch_size, padded_size, padded_size, 4), device=device)
    else:
        probs_batch = torch.zeros((batch_size, padded_size, padded_size, 4))
    
    # Process each cluster and fill the batch
    for batch_idx, (motl, motl_exit, motl_exit2, motl_entry, motl_entry2) in enumerate(cluster_data_list):
        n = cluster_sizes[batch_idx]
        
        # Extract data for this cluster
        pos_exit = motl_exit.df[['x', 'y', 'z']].values
        pos_exit2 = motl_exit2.df[['x', 'y', 'z']].values
        pos_entry = motl_entry.df[['x', 'y', 'z']].values
        pos_entry2 = motl_entry2.df[['x', 'y', 'z']].values
        
        vec_exit = pos_exit2 - pos_exit
        vec_exit = vec_exit / np.linalg.norm(vec_exit, axis=1, keepdims=True)
        vec_entry = pos_entry2 - pos_entry
        vec_entry = vec_entry / np.linalg.norm(vec_entry, axis=1, keepdims=True)
        
        # Convert to tensors
        pos_exit_t = torch.from_numpy(pos_exit).float()
        pos_exit2_t = torch.from_numpy(pos_exit2).float()
        pos_entry_t = torch.from_numpy(pos_entry).float()
        pos_entry2_t = torch.from_numpy(pos_entry2).float()
        vec_exit_t = torch.from_numpy(vec_exit).float()
        vec_entry_t = torch.from_numpy(vec_entry).float()
        
        # Move to device
        if device in ('cuda', 'mps'):
            pos_exit_t = pos_exit_t.to(device)
            pos_exit2_t = pos_exit2_t.to(device)
            pos_entry_t = pos_entry_t.to(device)
            pos_entry2_t = pos_entry2_t.to(device)
            vec_exit_t = vec_exit_t.to(device)
            vec_entry_t = vec_entry_t.to(device)
        
        # Compute probabilities for each case
        # Case 0: entry-entry
        probs_batch[batch_idx, :n, :n, 0] = _torch_probability_matrix(
            pos_entry_t, vec_entry_t, pos_entry_t, vec_entry_t, lo, lp
        ).cpu()
        
        # Case 1: exit-entry
        probs_batch[batch_idx, :n, :n, 1] = _torch_probability_matrix(
            pos_exit_t, vec_exit_t, pos_entry_t, vec_entry_t, lo, lp
        ).cpu()
        
        # Case 2: entry-exit
        probs_batch[batch_idx, :n, :n, 2] = _torch_probability_matrix(
            pos_entry_t, vec_entry_t, pos_exit_t, vec_exit_t, lo, lp
        ).cpu()
        
        # Case 3: exit-exit
        probs_batch[batch_idx, :n, :n, 3] = _torch_probability_matrix(
            pos_exit_t, vec_exit_t, pos_exit_t, vec_exit_t, lo, lp
        ).cpu()
        
        # Set diagonal to 0
        eye_mask = np.eye(n, dtype=bool)
        probs_batch[batch_idx, :n, :n, :][eye_mask] = 0.0
    
    # Move batch back to CPU and extract individual results
    probs_batch = probs_batch.cpu().numpy()
    
    results = []
    for batch_idx in range(batch_size):
        n = cluster_sizes[batch_idx]
        results.append(probs_batch[batch_idx, :n, :n, :])
    
    return results


def get_gpu_info():
    """
    Get information about available GPU hardware.
    
    Returns:
        dict: Dictionary with GPU availability and device info
    """
    info = {
        'torch_available': TORCH_AVAILABLE,
        'device': DEVICE,
        'cuda_available': torch.cuda.is_available() if TORCH_AVAILABLE else False,
        'mps_available': torch.backends.mps.is_available() if TORCH_AVAILABLE else False,
    }
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_device_count'] = torch.cuda.device_count()
    
    return info


def combine_cluster_linker_files(output_path_linker: str, output_filename: str = None) -> str:
    """
    Combine all cluster linker MOTL files into a single file with cluster_id column.
    
    Args:
        output_path_linker: Directory containing cluster linker files
        output_filename: Optional output filename (default: 'all_linkers.em')
    
    Returns:
        str: Path to the combined file
    """
    import glob
    
    # Find all linker files
    linker_pattern = os.path.join(output_path_linker, 'motl_tomo*_linkers.em')
    linker_files = glob.glob(linker_pattern)
    
    if not linker_files:
        print(f"No linker files found in {output_path_linker}")
        return None
    
    # Collect all dataframes with cluster_id
    all_dfs = []
    for filepath in linker_files:
        try:
            motl = cryomotl.EmMotl(input_motl=filepath)
            df = motl.df.copy()
            
            # Extract cluster_id from filename (motl_tomo{tomo_id}_cluster{cluster}_linkers.em)
            filename = os.path.basename(filepath)
            # Parse cluster from filename
            cluster_str = filename.split('_cluster')[1].split('_')[0]
            tomo_str = filename.split('_tomo')[1].split('_')[0]
            
            df['cluster_id'] = int(cluster_str)
            df['tomo_id_original'] = int(tomo_str)
            
            all_dfs.append(df)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    if not all_dfs:
        print("No valid linker data to combine")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Create combined MOTL
    combined_motl = cryomotl.Motl(combined_df)
    
    # Write combined file
    if output_filename is None:
        output_filename = 'all_linkers.em'
    
    output_path = os.path.join(output_path_linker, output_filename)
    combined_motl.write_out(output_path)
    
    print(f"Combined {len(linker_files)} cluster files into {output_path}")
    print(f"Total linkers: {len(combined_df)}")
    
    return output_path
