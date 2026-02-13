import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import matplotlib

import warnings
warnings.filterwarnings('ignore')
from cryocat import cryomotl
from cryocat import cryomap
import pandas as pd
import networkx as nx
import pickle

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


def calculate_probabilities(pos_selected, vector_selected, pos_current, vector_current, lo=500 / 1.971):
    """
    Calculate the probability of connection between two particles.

    Args:
    - pos_selected (numpy.ndarray): Position of the selected particle.
    - vector_selected (numpy.ndarray): Direction vector of the selected particle.
    - pos_current (numpy.ndarray): Position of the current particle.
    - vector_current (numpy.ndarray): Direction vector of the current particle.
    - !!! lo (float) default: (500 AA / 1.971AA/voxel) for the current case!: Persistence length used to estimate the bending energy

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
    
    probability = prob_linker_length(L=distance, lo=lo) * prob_bending_energy(L=distance, theta=theta_half)

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

def calculate_probabilities_all_connexions (motl, motl_exit, motl_exit2, motl_entry, motl_entry2, lo=lo):
    num_particles = len(motl.df)
    probs = np.zeros((num_particles, num_particles, 4))
    cases = np.zeros((num_particles, num_particles, 4))  # NxNx4 matrix to store all cases
    
    # Loop over particles
    for index, row in motl.df.iterrows():
        # Extract particle information for exit and entry
        pos_selected_exit = motl_exit.df.loc[index, ['x', 'y', 'z']].values
        vector_selected_exit = (motl_exit2.df.loc[index, ['x', 'y', 'z']] - pos_selected_exit) / np.linalg.norm(motl_exit2.df.loc[index, ['x', 'y', 'z']] - pos_selected_exit)
        
        pos_selected_entry = motl_entry.df.loc[index, ['x', 'y', 'z']].values
        vector_selected_entry = (motl_entry2.df.loc[index, ['x', 'y', 'z']] - pos_selected_entry) / np.linalg.norm(motl_entry2.df.loc[index, ['x', 'y', 'z']] - pos_selected_entry)
    
        for idx, row_entry in motl_entry.df.iterrows():
            if idx != index:
                pos_current_entry = row_entry[['x', 'y', 'z']].values
                vector_current_entry = (motl_entry2.df.loc[idx, ['x', 'y', 'z']] - pos_current_entry) / np.linalg.norm(motl_entry2.df.loc[idx, ['x', 'y', 'z']] - pos_current_entry)
    
                # Calculate probabilities and cases for entry-entry, entry-exit, exit-entry, exit-exit cases
                # entry-entry case
                probability_entry_entry = calculate_probabilities(pos_selected=pos_selected_entry,
                                                                  vector_selected=vector_selected_entry, 
                                                                  pos_current=pos_current_entry, 
                                                                  vector_current=vector_current_entry,
                                                                 lo=lo)
                probs[index, idx, 0] = probability_entry_entry
                
                # exit-entry case
                probability_exit_entry = calculate_probabilities(pos_selected=pos_selected_exit,
                                                                  vector_selected=vector_selected_exit, 
                                                                  pos_current=pos_current_entry, 
                                                                  vector_current=vector_current_entry,
                                                                  lo=lo)
                probs[index, idx, 1] = probability_exit_entry
                
                # EXIT positions
                row_exit = motl_exit.df.loc[idx]
                pos_current_exit = row_exit[['x', 'y', 'z']].values
                vector_current_exit = (motl_exit2.df.loc[idx, ['x', 'y', 'z']] - pos_current_exit) / np.linalg.norm(motl_exit2.df.loc[idx, ['x', 'y', 'z']] - pos_current_exit)
    
                # entry-exit case
                probability_entry_exit = calculate_probabilities(pos_selected=pos_selected_entry,
                                                                  vector_selected=vector_selected_entry, 
                                                                  pos_current=pos_current_exit, 
                                                                  vector_current=vector_current_exit,
                                                                  lo=lo)
                probs[index, idx, 2] = probability_entry_exit
                
                # exit-exit case
                probability_exit_exit = calculate_probabilities(pos_selected=pos_selected_exit,
                                                                 vector_selected=vector_selected_exit, 
                                                                 pos_current=pos_current_exit, 
                                                                 vector_current=vector_current_exit,
                                                                 lo=lo)
                probs[index, idx, 3] = probability_exit_exit
            
    return probs  
