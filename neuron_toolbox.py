import nest
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Any
from IPython.display import Image, display
from numpy import diag, array, zeros 
import random
import numpy as np
import numpy as np, heapq, random

SCALING_FACTOR = 1.0
k= 10.0 # for default
m= np.zeros(3)
dt = 0.01
positions=[np.zeros(3)]

gsl = 16 #grid side length


area2models = {
    
    "V1": ["iaf_cond_alpha","aeif_cond_alpha","iaf_cond_exp","hh_psc_alpha","iaf_cond_alpha","iaf_cond_beta_gap"],
    
    "V2": ["aeif_cond_alpha","iaf_cond_exp","aeif_cond_alpha","hh_psc_alpha","iaf_cond_alpha"],
    
    "V3": ["aeif_cond_alpha","iaf_cond_exp","aeif_cond_alpha","hh_psc_alpha","iaf_cond_alpha"], 
    
    "V4": ["aeif_cond_alpha","iaf_cond_exp","hh_psc_alpha","gif_cond_exp","iaf_cond_alpha","iaf_cond_beta_gap"],
    
    "V5": ["iaf_cond_alpha","aeif_cond_alpha","hh_psc_alpha","izhikevich","iaf_cond_beta_gap"],
    
    "V6": ["iaf_cond_alpha","aeif_cond_alpha","hh_psc_alpha","iaf_psc_delta","iaf_cond_alpha","iaf_cond_beta_gap"]
    
}





### Helper functions

def extract_connections(pre, post):

    conn = nest.GetConnections(pre, post)
    df = pd.DataFrame({
        "pre_gid":  conn.source,
        "post_gid": conn.target,
        "weight":   nest.GetStatus(conn, "weight"),
        "delay":    nest.GetStatus(conn, "delay")
    })
    return df


def to_adjacency(df, pre_ids, post_ids, attr="weight"):
    idx = {gid: i for i, gid in enumerate(pre_ids)}
    jdx = {gid: j for j, gid in enumerate(post_ids)}
    mat = np.zeros((len(pre_ids), len(post_ids)))
    for _, row in df.iterrows():
        i, j = idx[row.pre_gid], jdx[row.post_gid]
        mat[i, j] = row[attr]
    return mat


def plot_point_clusters(
    clusters,
    colors= None,
    marker_size = 20,
    cmap = 'tab10',
    xlabel = 'X',
    ylabel = 'Y',
    zlabel = 'Z',
    title = None,
    edgecolor='k', alpha=0.8,linewidths=1.5
):
    """
    Plots multiple 3D point clouds in distinct colors.

    Args:
        clusters (List[np.ndarray]): 
            List of arrays, each of shape (N_i, 3), containing (x,y,z) points.
        colors (List[str], optional):
            List of colors (matplotlib format) to use for each cluster. 
            If None, a categorical colormap is used.
        marker_size (float, optional):
            Size of scatter markers.
        cmap (str, optional):
            Name of categorical colormap (used if colors is None).
        xlabel, ylabel, zlabel (str, optional):
            Axis labels.
        title (str, optional):
            Plot title.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = len(clusters)
    
    if colors is None:
        cmap_obj = plt.get_cmap(cmap)
        colors = [cmap_obj(i / max(n-1, 1)) for i in range(n)]
    
    for pts, col in zip(clusters, colors):
        pts = np.asarray(pts)
        if pts.size == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=[col], s=marker_size, edgecolor=edgecolor, alpha=alpha,linewidths=linewidths)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if title:
        ax.set_title(title)
    plt.show()
    

def plot_point_clusters_normalized(
        clusters,
        colors=None,
        marker_size=20,
        cmap='tab10',
        xlabel='X', ylabel='Y', zlabel='Z',
        title=None,
        edgecolor='k', alpha=0.8, linewidths=1.5):

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    n = len(clusters)
    if colors is None:
        cmap_obj = plt.get_cmap(cmap)
        colors = [cmap_obj(i / max(n - 1, 1)) for i in range(n)]

    mins = np.full(3,  np.inf)
    maxs = np.full(3, -np.inf)

    for pts, col in zip(clusters, colors):
        pts = np.asarray(pts)
        if pts.size == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=[col], s=marker_size,
                   edgecolor=edgecolor, alpha=alpha, linewidths=linewidths)

        mins = np.minimum(mins, pts.min(axis=0))
        maxs = np.maximum(maxs, pts.max(axis=0))

    span   = max(maxs - mins)
    centre = (maxs + mins) / 2

    ax.set_xlim(centre[0] - span / 2, centre[0] + span / 2)
    ax.set_ylim(centre[1] - span / 2, centre[1] + span / 2)
    ax.set_zlim(centre[2] - span / 2, centre[2] + span / 2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

# may be usefull later, I dunno.

def make_div_curl(f1, f2, f3, h = 1e-5):

    def div(x, y, z):
        df1dx = (f1(x + h, y, z) - f1(x - h, y, z)) / (2*h)
        df2dy = (f2(x, y + h, z) - f2(x, y - h, z)) / (2*h)
        df3dz = (f3(x, y, z + h) - f3(x, y, z - h)) / (2*h)
        return df1dx + df2dy + df3dz

    def curl(x, y, z):
        df3dy = (f3(x, y + h, z) - f3(x, y - h, z)) / (2*h)
        df2dz = (f2(x, y, z + h) - f2(x, y, z - h)) / (2*h)
        c1 = df3dy - df2dz

        df1dz = (f1(x, y, z + h) - f1(x, y, z - h)) / (2*h)
        df3dx = (f3(x + h, y, z) - f3(x - h, y, z)) / (2*h)
        c2 = df1dz - df3dx

        df2dx = (f2(x + h, y, z) - f2(x - h, y, z)) / (2*h)
        df1dy = (f1(x, y + h, z) - f1(x, y - h, z)) / (2*h)
        c3 = df2dx - df1dy

        return np.array([c1, c2, c3])

    return div, curl


###### GEOMETRIC NUMPY FUNCTIONS ######


def circle3d(
    n=10, 
    r=1.0, 
    theta=0, 
    phi=0, 
    m=(0,0,0), 
    name="Circle", 
    plot=False
    ):
    """
    Generate points on a rotated and translated 3D circle.

    Args:
        n (int): Number of points to sample along the circle.
        r (float): Radius of the circle.
        theta (float): Rotation angle around the X-axis, in degrees.
        phi (float):   Rotation angle around the Y-axis, in degrees.
        m (tuple):     3-tuple giving the translation offset (x, y, z).
        name (str):    Title for the plot if `plot=True`.
        plot (bool):   If True, displays a 3D plot of the circle.

    Returns:
        np.ndarray: Array of shape (n, 3) containing the (x, y, z) coordinates.
    """

    
    X_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x_coords = SCALING_FACTOR * r * np.cos(X_angles)
    y_coords = SCALING_FACTOR * r * np.sin(X_angles)
    z_coords = np.zeros(n)

    theta = np.deg2rad(theta)
    phi   = np.deg2rad(phi)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta),  np.cos(theta)]])
    Ry = np.array([[ np.cos(phi), 0, np.sin(phi)],
                   [0,           1, 0         ],
                   [-np.sin(phi),0, np.cos(phi)]])
    rotM = Ry @ Rx

    pts = np.vstack((x_coords, y_coords, z_coords)).T
    pts = pts @ rotM
    pts = pts + np.array(m)

    if plot:
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot(pts[:,0], pts[:,1], pts[:,2], lw=2)
        max_val = np.max(np.abs(pts))
        ax.set_xlim(-max_val*1.2, max_val*1.2)
        ax.set_ylim(-max_val*1.2, max_val*1.2)
        ax.set_zlim(-max_val*1.2, max_val*1.2)
        ax.set_title(name)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        plt.show()

    return pts


def create_cone(
    m=np.zeros(3, dtype=np.float32),
    n=1500,
    inner_radius=0.1,
    outer_radius=0.2,
    height=0.6,
    rot_theta=0.0,
    rot_phi=0.0,
    plot=False
    ):
    #~*~*~*~* START *~*~*~*~#
    """
    Generate random points within a truncated cone, apply rotation and translation.

    Args:
        m (np.ndarray): 3-element center offset (x, y, z) of the cone base.
        n (int): Number of points to generate.
        inner_radius (float): Radius of the top (smaller) circle.
        outer_radius (float): Radius of the bottom (larger) circle.
        height (float): Vertical height of the cone.
        rot_theta (float): Rotation angle around the X-axis in degrees.
        rot_phi (float): Rotation angle around the Y-axis in degrees.
        plot (bool): If True, display a 3D scatter plot of generated points.

    Returns:
        np.ndarray: Array of shape (n, 3) containing the (x, y, z) coordinates of points.
    """
    
    #~*~*~*~* END *~*~*~*~#
    

    inner_r = inner_radius * SCALING_FACTOR  
    outer_r = outer_radius * SCALING_FACTOR
    h = height * SCALING_FACTOR
    
    ang = np.random.uniform(0, 2*np.pi, n)
    z = np.random.uniform(m[2], m[2] + h, n)
    
    r = inner_r + (outer_r - inner_r) * ((z - m[2]) / h)
    
    x = m[0] + r * np.cos(ang)
    y = m[1] + r * np.sin(ang)
    
    # randomized on the surface.
    th = np.deg2rad(rot_theta)
    ph = np.deg2rad(rot_phi)
    
    Rx = np.array([[1,         0,          0       ],
                   [0, np.cos(th), -np.sin(th)],
                   [0, np.sin(th),  np.cos(th)]])
    Ry = np.array([[ np.cos(ph), 0, np.sin(ph)],
                   [          0, 1,          0],
                   [-np.sin(ph), 0, np.cos(ph)]])
    rotM = Ry @ Rx
    
    points = np.column_stack((x, y, z))
    points = points @ rotM.T
    
    if plot:
        from mpl_toolkits.mplot3d import Axes3D 
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2], s=2)
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        plt.show()
    
    return points



def blob_positions(
    n = 10,
    m = np.zeros(3),
    r = 1.0,
    scaling_factor = 1.0,
    plot=False,
    name="Blob"
    ):
        #~*~*~*~* START  *~*~*~*~#

    """
    Generate n random points uniformly distributed inside a sphere.

    Args:
        n (int): Number of points.
        m (np.ndarray): 3-vector specifying the sphere center.
        r (float): Radius of the sphere.
        scaling_factor (float): Global scaling to apply to all coordinates.

    Returns:
        np.ndarray: Array of shape (n,3) with the generated (x,y,z) positions.
    """

    pos_norm = np.random.normal(size=(n, 3))
    pos_norm /= np.linalg.norm(pos_norm, axis=1, keepdims=True)
    
    u = np.random.rand(n, 1)
    r_scaling = r * u**(1/3)
    
    pts = m + pos_norm * r_scaling
    pts *= scaling_factor
    

    if plot:
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')

        ax.scatter(pts[:,0], pts[:,1], pts[:,2])

        lim = 1.1*np.abs(pts).max()
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim),
               title=name, xlabel='X', ylabel='Y', zlabel='Z')
        plt.show()
    return pts

def set_axes_equal(ax):
        #~*~*~*~* START  *~*~*~*~#

    """
    Adjust a 3D Matplotlib axis so that the X, Y and Z axes are scaled equally.

    This avoids distortion in 3D plots by ensuring that one unit in X, Y, and Z
    appear the same length on screen.

    Args:
        ax (mpl_toolkits.mplot3d.axes3d.Axes3D): The 3D axes to adjust.
    """
        #~*~*~*~* END  *~*~*~*~#

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d(x_middle - max_range/2, x_middle + max_range/2)
    ax.set_ylim3d(y_middle - max_range/2, y_middle + max_range/2)
    ax.set_zlim3d(z_middle - max_range/2, z_middle + max_range/2)

    
    

def create_Grid(m=np.zeros(3, dtype=np.float32),
                grid_size_list=[28, 15, 10],
                rot_theta=0.0, 
                rot_phi=0.0, 
                plot=False):
        #~*~*~*~* START  *~*~*~*~#

    """
    Generate a stack of 2D grid layers in 3D space and optionally plot them.

    Each layer is a k×k grid of points lying in a plane at increasing offsets
    along the first coordinate. The entire stack is then rotated and translated.

    Args:
        m (np.ndarray):       3-vector translation offset applied after rotation.
        grid_size_list (list): List of integers [d1, d2, d3,...], where each
                               entry di defines the grid resolution (di×di)
                               for the layer at depth index i.
        rot_theta (float):    Rotation angle around the X-axis in degrees.
        rot_phi (float):      Rotation angle around the Y-axis in degrees.
        plot (bool):          If True, display a 3D scatter of all layers.

    Returns:
        list of np.ndarray:   A list where each element is an (di*di, 3) array
                              of 3D coordinates for layer i.
    """
        #~*~*~*~* END  *~*~*~*~#

    
    th = np.deg2rad(rot_theta)
    ph = np.deg2rad(rot_phi)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(th), -np.sin(th)],
                   [0, np.sin(th),  np.cos(th)]])
    Ry = np.array([[ np.cos(ph), 0, np.sin(ph)],
                   [          0, 1,          0],
                   [-np.sin(ph), 0, np.cos(ph)]])
    rotM = Ry @ Rx

    node_layers = []
    for d, k in enumerate(grid_size_list):
        pts = np.array([[d + m[0], w + m[1], n + m[2]]
                        for w in range(k) for n in range(k)])
        node_layers.append(pts @ rotM.T)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for layer_pts in node_layers:
            ax.scatter(layer_pts[:, 0], layer_pts[:, 1], layer_pts[:, 2], s=1)
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        try:
            ax.set_box_aspect((1,1,1))
        except AttributeError:
            set_axes_equal(ax)
        plt.show()

    return node_layers

def create_Grid_z(m=np.zeros(3, dtype=np.float32),
                grid_size_list=[28, 15, 10],
                rot_theta=0.0, 
                rot_phi=0.0, 
                plot=False):
    
    th = np.deg2rad(rot_theta)
    ph = np.deg2rad(rot_phi)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(th), -np.sin(th)],
                   [0, np.sin(th),  np.cos(th)]])

    Ry = np.array([[ np.cos(ph), 0, np.sin(ph)],
                   [          0, 1,          0],
                   [-np.sin(ph), 0, np.cos(ph)]])

    rotM = Ry @ Rx

    node_layers = []
    for d, k in enumerate(grid_size_list):
        pts = np.array([[w, n, d] for w in range(k) for n in range(k)])  # jetzt XY-Gitter in Z-Tiefe d
        pts = pts @ rotM.T + m
        node_layers.append(pts)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for layer_pts in node_layers:
            ax.scatter(layer_pts[:, 0], layer_pts[:, 1], layer_pts[:, 2], s=1)
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        try:
            ax.set_box_aspect((1,1,1))
        except AttributeError:
            set_axes_equal(ax)
        plt.show()

    return node_layers


# can use a mask.
# better for more uniformly sized clusters. here even low prob. types form clusters approx this size. 
# they stay just rare according to their prob. or so I strongly assume. because of the nature of this algorithm.
# this holds for < (space_dim**space_dim - 1 - log(base=space_dim,x=( space_dim**space_dim))) = 23 types (in 3D) 
# this pattern should break because the clusters form from the pigion hole principle. or so I imagine
def wave_collapse_old(mask=None, 
                  dims=(2,2,2), 
                  sparse_holes = 0,
                  type_array=[0,1],
                  probability_vector=np.array([0.3,0.7]), 
                  start_pos=np.zeros(3),
                  sparsity_factor=0.7,
                seed=42):
    
    """
    Perform a 3D "wave function collapse" to assign discrete labels to grid cells, 
    forming clusters based on local entropy and neighbor constraints.

    Parameters:
        mask (np.ndarray or None): Initial occupancy mask of shape dims. 
            0 = uncollapsed, -1 = blocked. If None, a zero mask is created.
        dims (tuple[int]): Grid dimensions if mask is None.
        sparse_holes (int): Number of random positions to block (-1) for holes.
        type_array (list[int]): List of possible labels.
        probability_vector (np.ndarray): Initial probability for each label.
        start_pos (array-like): Starting coordinate to seed collapse.
        sparsity_factor (float): Controls influence of existing neighbor labels.
        seed (int): Base seed for random number generators.

    Returns:
        np.ndarray: Array of shape dims with assigned labels for each cell.
    """
    
    
    
    # Initialize random seeds for reproducibility
    random.seed(seed+3)
    np.random.seed(seed+7)

    # Normalize probability vector
    probability_vector = np.asarray(probability_vector, dtype=float)
    probability_vector /= probability_vector.sum()
    
    # Create mask if not provided: 0=uncollapsed, -1=blocked
    if mask is None:
        mask = np.zeros(dims, dtype=np.int32)
    dims=mask.shape
    
    # High initial entropy for all cells
    entropy_matrix = np.full(shape=dims, fill_value=1000.0, dtype=np.float32)
    
    # Store collapsed labels; -1 indicates not yet collapsed
    collapsed_nodes = np.full(shape=dims, fill_value=-1, dtype=np.int32)
    
    
    D0, D1, D2 = mask.shape
    
    
    # Randomly block positions to create holes if desired
    if(sparse_holes!=0):
        for _ in range(0,sparse_holes,1):
            x = random.randint(0,D0-1)
            y = random.randint(0,D1-1)
            z = random.randint(0,D2-1)
            mask[x][y][z] = -1
        
    def check_neighbors(i, j, k):
        undef_neighbors = []
        def_neighbors = []
        """
        Return lists of uncollapsed and collapsed neighbor coordinates in all directions.
        """ 
        shifts = [(dx, dy, dz)         # Examine all 26 neighbors in 3D
              for dx in (-1, 0, 1) 
              for dy in (-1, 0, 1) 
              for dz in (-1, 0, 1) 
              if not (dx == dy == dz == 0)]        
        for di, dj, dk in shifts:
            ni, nj, nk = i+di, j+dj, k+dk
            if 0 <= ni < D0 and 0 <= nj < D1 and 0 <= nk < D2:
                if mask[ni, nj, nk] == 0:
                    undef_neighbors.append((ni, nj, nk))
                elif mask[ni, nj, nk] == 1:
                    def_neighbors.append((ni, nj, nk))
                else:
                    continue  # ignore blocked cells

                    
        return undef_neighbors, def_neighbors
    

    def entropy(p, base=None):
        """Compute Shannon entropy of probability vector p."""
        p = np.asarray(p, dtype=float)
        p += 0.0001
        p/=np.sum(p)# p > 0
        log_p = np.log(p) if base is None else np.log(p) / np.log(base)
        return -np.sum(p * log_p)
    
    def collapse(i, j, k):        
        """
        Collapse the cell at (i,j,k) by sampling a label based on local probabilities and neighbor influence.
        """
        if(mask[i][j][k]):#if already collapsed
            pass #TODO for invariants
        
        # Copy and adjust probabilities based on collapsed neighbors

        
        l = len(probability_vector)
        l1 = int((1-sparsity_factor)*l) 
        # Boost probability for each neighbor label

        list_of_uncollapsed_neighbors, list_of_collapsed_neighbors = check_neighbors(i,j,k)
        
        neighbor_labels = []
        for x,y,z in list_of_collapsed_neighbors:
            neighbor_labels.append(collapsed_nodes[x][y][z])
        
        #possible_values = np.setdiff1d(type_array,neighbor_labels) #regel zu hart für Klusterbildung
        
        private_probability_vector = probability_vector.copy()
        
        for m in neighbor_labels:# label gleichzeitig index für probability vektor
            # private_probability_vector anpassen
            p = private_probability_vector[m] / (l + l1)
            p_self = l1 * p
            p_rest = p
            p_redistribution = np.full(shape=(l,),fill_value=p_rest)
            p_redistribution[m] += p_self
            private_probability_vector[m] = 0.0 #setze wert im original zurück
            private_probability_vector+=p_redistribution
        private_probability_vector/=private_probability_vector.sum()#danach um sequenziellen Einfluss zu minimieren
        #print("#######",np.sum(private_probability_vector))#just a check
        choice = np.random.choice(type_array,p=private_probability_vector)#damit kein ablaufmuster entsteht
        collapsed_nodes[i][j][k] = choice
        mask[i][j][k]=1
        entropy_matrix[i][j][k] = 0.0
        # Update entropy for uncollapsed neighbors

        #after collapsed
        # Anpassung der Entropie der Nachbarn

        for ni, nj, nk in list_of_uncollapsed_neighbors:
            _, coll_nbrs = check_neighbors(ni, nj, nk)
            neigh_labels = [collapsed_nodes[x,y,z] for x,y,z in coll_nbrs]

            allowed = np.setdiff1d(type_array, neigh_labels)

            p = np.zeros_like(probability_vector, dtype=float)
            idxs = [type_array.index(a) for a in allowed]
            p[idxs] = probability_vector[idxs]
            if p.sum() > 0:
                p /= p.sum()
            else:
                p[:] = 1 / len(p)
            entropy_matrix[ni, nj, nk] = entropy(p, base=2)
        
            
    def _next():
        """Select the next uncollapsed cell with minimal non-zero entropy."""
        candidates = np.argwhere(mask == 0)
        if candidates.size == 0:
            return None
        valid = [(tuple(idx), entropy_matrix[tuple(idx)]) 
                 for idx in candidates 
                 if entropy_matrix[tuple(idx)] > 0]
        if not valid:
            return None
        next_idx = min(valid, key=lambda x: x[1])[0]
        return next_idx
    
    # Main collapse loop
    while True:
        nxt = _next()
        if nxt is None:
            break
        collapse(*nxt)

    return np.array(collapsed_nodes)#, entropy_matrix


# newer version, much faster on big clusters but doesnt use entropy to determine the next cell to collapse.
# chooses randomly instead.
# für minikolonne mit >80–100 Neuronen und makrokolonne mit 4000–12000 Neuronen



def wave_collapse(dims, type_array, probability_vector,
                       sparsity_factor=.7, seed=0, sparse_holes=0):
    NEIGHBOR_OFFSETS = np.array([(dx,dy,dz)
      for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)
      if (dx,dy,dz)!=(0,0,0)], dtype=np.int8)
    random.seed(seed+3); np.random.seed(seed+7)
    probability_vector = np.asarray(probability_vector, float); probability_vector /= probability_vector.sum()
    mask  = np.zeros(dims, np.int8)
    if sparse_holes:                       
        idx = np.random.choice(mask.size, sparse_holes, replace=False)
        mask.ravel()[idx] = -1

    collapsed = -np.ones(dims, np.int16)
    H = np.full(dims, 999., np.float32)

    heap = [(H[i,j,k],i,j,k) for i in range(dims[0])
                               for j in range(dims[1])
                               for k in range(dims[2]) if mask[i,j,k]==0]
    heapq.heapify(heap)

    while heap:
        h,i,j,k = heapq.heappop(heap)
        if mask[i,j,k]:                    
            continue

        neigh = NEIGHBOR_OFFSETS + (i,j,k)
        good  = ((neigh[:,0] >= 0) & (neigh[:,0] < dims[0]) &
                 (neigh[:,1] >= 0) & (neigh[:,1] < dims[1]) &
                 (neigh[:,2] >= 0) & (neigh[:,2] < dims[2]))
        neigh = neigh[good]
        labels = collapsed[tuple(neigh.T)]
        labels = labels[labels >= 0]              

        if labels.size:
            counts = np.bincount(labels, minlength=len(type_array))
            boost  = counts * (1-sparsity_factor)/len(type_array)
            p = probability_vector + boost
            p /= p.sum()
        else:
            p = probability_vector

        choice = np.random.choice(type_array, p=p)
        collapsed[i,j,k] = choice
        mask[i,j,k] = 1
        H[i,j,k] = 0.0

        for ni,nj,nk in neigh:
            if mask[ni,nj,nk]==0:
                H[ni,nj,nk] = 1.0  
                heapq.heappush(heap,(H[ni,nj,nk],ni,nj,nk))

    return collapsed

# Runtime is of the old algorithm is roughly quadratic–cubic, but it’s intended to create patches for micro-clusters.
# A grid of size 10×20×20 already corresponds to about 4K neurons.
# sure it takes time, but if you want to have "random structures of higher quality" - whatever that means -
# then you should create miniclusters with old wfc and maybe patch them together. because the runtime here
# is horrible.


# Each natural number in the grid will eventually serve as a class label for neurons.
# That mapping will be implemented in the corresponding function.

# -1 indicates a blocked cell where no neuron can be placed.
#  0 indicates an available cell that could hold a neuron but hasn’t been assigned yet.
#    (You could choose to assign a neuron to 0 if desired.)
# This scheme reserves space in the grid for later hypothetical neurogeneration,
# ensuring that these slots persist under any transformations.

# Next step: transform the grid and then convert each label into actual neuron placements.

# You can clearly see how “channels” of like-typed neurons emerge.
# It works exactly as intended!

def generate_qube(grid_size_list=(10, 10, 10)):
    """
    Generate a 3D grid of points within a cube from -1 to 1 in each axis.

    Args:
        grid_size_list (tuple of int): Number of samples along each axis (nx, ny, nz).

    Returns:
        np.ndarray: An array of shape (nx*ny*nz, 3) containing the (x, y, z) coordinates.
    """
    nx, ny, nz = grid_size_list
    xs = np.linspace(-1, 1, nx)
    ys = np.linspace(-1, 1, ny)
    zs = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    return np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T 

def transform_points(
    points,
    m = np.zeros(3, dtype=float),
    rot_theta = 0.0,
    rot_phi = 0.0,
    transform_matrix = np.eye(3),
    plot = False
    ):
    """
    Apply rotation, linear transformation, and translation to a set of 3D points.

    Args:
        points (np.ndarray): Array of shape (N,3) containing the original points.
        m (np.ndarray):      Translation offset as a 3-vector.
        rot_theta (float):   Rotation angle around the X-axis in degrees.
        rot_phi (float):     Rotation angle around the Y-axis in degrees.
        transform_matrix (np.ndarray): 3×3 matrix to apply after rotation.
        plot (bool):         If True, display a 3D scatter of the transformed points.

    Returns:
        np.ndarray: Array of shape (N,3) with transformed coordinates.
    """
    pts = np.asarray(points, dtype=float)
    
    th = np.deg2rad(rot_theta)
    ph = np.deg2rad(rot_phi)
    Rx = np.array([[1,          0,           0],
                   [0, np.cos(th), -np.sin(th)],
                   [0, np.sin(th),  np.cos(th)]])
    Ry = np.array([[ np.cos(ph), 0, np.sin(ph)],
                   [          0, 1,          0],
                   [-np.sin(ph), 0, np.cos(ph)]])
    rotM = Ry @ Rx
    
    rotated = (rotM @ pts.T).T
            
    transformed = (transform_matrix @ rotated.T).T
    
    result = transformed + np.asarray(m, dtype=float)
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(result[:,0], result[:,1], result[:,2], s=2)
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        plt.show()
    
    return result

def field(
        positions: np.ndarray,
        f1 = lambda x, y, z: np.zeros_like(x),
        f2 = lambda x, y, z: np.zeros_like(x),
        f3 = lambda x, y, z: np.zeros_like(x),
        plot=False,    
        normalize = False,
        color_by_length = False,
        cmap = 'viridis',
        arrow_length = 0.1
    ):
    """
    Compute a 3D vector field at given positions and optionally visualize it.

    Args:
        positions (np.ndarray): An (N,3) array of (x,y,z) coordinates.
        f1, f2, f3 (callable):  Functions f1(x,y,z), f2(x,y,z), f3(x,y,z) defining
                                the three vector components at each point.
        plot (bool):           If True, display a 3D quiver plot of the field.
        normalize (bool):      If True, normalize each vector to unit length before plotting.
        color_by_length (bool): If True, color arrows by their original lengths.
        cmap (str):             Name of the Matplotlib colormap to use when coloring.
        arrow_length (float):   Scaling factor for arrow lengths in the plot.

    Returns:
        np.ndarray: An (N,3) array of the computed vectors [v1, v2, v3].
    """
    
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    v1 = f1(x, y, z)
    v2 = f2(x, y, z)
    v3 = f3(x, y, z)
    vectors = np.column_stack((v1, v2, v3))
    if plot:
        lengths = np.linalg.norm(vectors, axis=1)
        U, V, W = v1.copy(), v2.copy(), v3.copy()
        if normalize:
            nonzero = lengths > 0
            U[nonzero] /= lengths[nonzero]
            V[nonzero] /= lengths[nonzero]
            W[nonzero] /= lengths[nonzero]
            U *= arrow_length
            V *= arrow_length
            W *= arrow_length
        else:
            U *= arrow_length
            V *= arrow_length
            W *= arrow_length

        color_args = {}
        if color_by_length:
            normed = (lengths - lengths.min()) / (lengths.ptp() if lengths.ptp()>0 else 1)
            cmap_obj = plt.get_cmap(cmap)
            colors = cmap_obj(normed)
            color_args['color'] = colors

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(
            x, y, z,
            U, V, W,
            **color_args,
            length=1.0,
            normalize=False
        )
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        plt.show()
    return vectors

# This is already a derivative; the next derivative would represent acceleration.
# Perhaps negative acceleration as deviation from the mean could help create suitable attractors.
# meh, nah. 
# Remember: t * d/dx
# and d/dx can be approximated as (f(x+h) - f(x)) / h for each point. I could use this one deriv. step later too.

def generate_torus(
        R = 1.0,
        r = 0.3,
        grid_size_list = (100, 40),
    rot_theta=0,
    rot_phi=0,
    plot=False
    ):
    """
    Generate a set of 3-D points on the surface of a torus.

    The torus lies in the XY-plane, centred at the origin.  The major
    radius R is the distance from the origin to the centre of the tube,
    the minor radius r is the tube radius.

    Args:
        R (float):            Major radius (distance from origin to tube centre).
        r (float):            Minor radius (tube radius).
        grid_size_list (tuple of int):
                              Number of samples along the two angular
                              coordinates (n_theta, n_phi).  Higher numbers
                              produce a denser point cloud.

    Returns:
        np.ndarray: Array of shape (n_theta * n_phi, 3) with (x, y, z)
                    coordinates on the torus surface.
    """
    n_theta, n_phi = grid_size_list

    
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)  
    phi   = np.linspace(0.0, 2.0 * np.pi, n_phi,   endpoint=False)  

    a, b = np.meshgrid(theta, phi, indexing='ij')  

    x = (R + r * np.cos(b)) * np.cos(a)
    y = (R + r * np.cos(b)) * np.sin(a)
    z =  r * np.sin(b)

    return transform_points(np.vstack((x.ravel(), y.ravel(), z.ravel())).T,
                            rot_theta=rot_theta, rot_phi=rot_phi, plot=plot)


def field_flow_iteration(
                        position=np.zeros(3),
                        f1 = lambda x, y, z: np.zeros_like(x),
                        f2 = lambda x, y, z: np.zeros_like(x),
                        f3 = lambda x, y, z: np.zeros_like(x)):
    """
    Compute a single time-step update of positions moving in a vector field.

    Args:
        positions (np.ndarray): Array of shape (N,3) with current positions.
        dt (float):            Time step size.
        f1, f2, f3 (callable): Functions defining the vector field components.

    Returns:
        np.ndarray: Updated positions after dt using Euler integration.
    """
    velocity = field(
        positions=positions,
        plot=False,
        f1=f1,
        f2=f2,
        f3=f3
    )
    return positions + dt * velocity


def simulate_vector_field_flow(
            initial_positions = np.zeros((3,3)),
            dt = 0.001,
            f1 = lambda x, y, z: np.zeros_like(x),
            f2 = lambda x, y, z: np.zeros_like(x),
            f3 = lambda x, y, z: np.zeros_like(x),
            num_steps = 1,
            plot= True,
            scatter_size = 1.0,
            create_object = False #2 Modi, False liefert lediglich die letzte Position, also Endergebnis
                                  # True liefert Liste aller Zwischenschritte
            ):
    """
    Simulate the flow of points through a 3D vector field over multiple time steps.

    Args:
        initial_positions (np.ndarray): Starting positions, shape (N,3).
        dt (float):                    Time step size for Euler integration.
        f1, f2, f3 (callable):         Functions defining the vector field components.
        num_steps (int):               Number of integration steps.
        plot (bool):                   If True, display scatter plots at each step.
        scatter_size (float):          Size of points in the scatter plot.
        create_object (bool):          If True, return a list of all intermediate
                                       position arrays; otherwise return only final positions.

    Returns:
        np.ndarray or List[np.ndarray]:
            Final positions array if create_object=False, otherwise a list of
            position arrays at each time step.
    """
    
    positions = initial_positions.copy()
    snapshots = []
    
    if(create_object):
        snapshots.append(positions)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    
    for _ in range(num_steps):
        if plot:
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=scatter_size)
        velocity = field(positions, f1=f1, f2=f2, f3=f3, plot=False)
        positions = positions + dt * velocity
        if(create_object):
            snapshots.append(positions)

    if plot:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    if(create_object):
        return snapshots
    return positions

def add_local_attractor(
    m = np.zeros(3),
    k = 15.0,      # Sog-Stärke
    sigma = 0.9,   # Gauß-Breite
    f1 = lambda x, y, z: np.zeros_like(x),
    f2 = lambda x, y, z: np.zeros_like(x),
    f3 = lambda x, y, z: np.zeros_like(x),
    repulsive=False
):
    """
    Adds a local attractor parametrized by the values above. returns a modified version of the original function.
    The influence of attractors decays exponentially with distance. Like this spherical one.
    """
    x0, y0, z0 = m
    s = 1.0 if repulsive else -1.0

    def w(r):
        return s*np.exp(-0.5*(r/sigma)**2)

    def f1_mod(x, y, z):
        dx = x - x0
        r = np.sqrt(dx*dx + (y-y0)**2 + (z-z0)**2)
        sink = -k * s * w(r) * dx
        return f1(x, y, z) + sink

    def f2_mod(x, y, z):
        dy = y - y0
        r = np.sqrt((x-x0)**2 + dy*dy + (z-z0)**2)
        sink = -k * s * w(r) * dy
        return f2(x, y, z) + sink

    def f3_mod(x, y, z):
        dz = z - z0
        r = np.sqrt((x-x0)**2 + (y-y0)**2 + dz*dz)
        sink = -k * s * w(r) * dz
        return f3(x, y, z) + sink

    return f1_mod, f2_mod, f3_mod

def find_positions_by_type(
    volume,
    types
):
    """
    For a 3D integer array, find the coordinates of all voxels matching each of the given types.

    Args:
        volume (np.ndarray):
            A 3D NumPy array of dtype int (shape (n, m, o)).
        types (List[int]):
            A list of integer “types” to search for in `volume`.

    Returns:
        Dict[int, np.ndarray]:
            A dict mapping each type → an array of shape (K, 3), where K is the number
            of occurrences of that type, and each row is the (i, j, k) index in `volume`.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D array, got ndim={volume.ndim}")

    positions_by_type: Dict[int, np.ndarray] = {}
    for t in types:
        # np.argwhere returns an (K,3) array of coordinates where volume == t
        coords = np.argwhere(volume == t)
        positions_by_type[t] = coords

    return positions_by_type



def cluster_and_flow(
    grid_size,
    m,
    rot_theta,
    rot_phi,
    transform_matrix,
    wave_params,
    types,
    flow_functions,
    dt,
    num_steps,
    plot_clusters = True,
    title="Cluster-Flows"
):
    """
    1) Create a cubic grid
    2) Apply a one‐time transformation (rotate/translate/scale)
    3) Generate the `neuron_type_array` via `wave_collapse`
    4) Split the grid into sub‐arrays by type
    5) Simulate each sub‐array under its own flow field
    6) Plot all resulting clusters (optional)
    7) Return the list of final position arrays
    """
    # 2
    grid = generate_qube(grid_size)
    pts  = transform_points(grid, m=m, rot_theta=rot_theta,
                            rot_phi=rot_phi, transform_matrix=transform_matrix,
                            plot=False)
    # 3
    neuron_type_array = wave_collapse(
        dims = grid_size,
        sparse_holes     = wave_params['sparse_holes'],
        sparsity_factor  = wave_params['sparsity_factor'],
        probability_vector = wave_params['probability_vector'],
        type_array       = types
    )
    idx = find_positions_by_type(neuron_type_array, types)
    # 4
    nx, ny, nz = grid_size
    slice_size = ny * nz
    row_size   = nz
    arrays = []
    for t in types:
        tripels   = idx[t]
        flat_idx  = tripels[:,0]*slice_size + tripels[:,1]*row_size + tripels[:,2]
        arrays.append(pts[flat_idx])
    # 5
    clusters = []
    for (f1,f2,f3), cluster_pts in zip(flow_functions, arrays):
        final_positions = simulate_vector_field_flow(
            initial_positions=cluster_pts,
            dt=dt, f1=f1, f2=f2, f3=f3,
            num_steps=num_steps,
            plot=False, scatter_size=1.0,
            create_object=False
        )
        clusters.append(final_positions)
    # 6
    if plot_clusters:
        plot_point_clusters(clusters, title=title)
    # 7
    return clusters


####### NEST #######


def create_CCW(
    positions,
    model='iaf_psc_alpha',
    plot=False,
    k=10.0,
    bidirectional = False,
    conn_dict_ex = {
        'rule': 'one_to_one'
    },
    syn_spec_ex = {
        'synapse_model': 'static_synapse',
        'weight': 30.0,
        'delay': 1.0
    },
    conn_dict_inh = {
        'rule': 'pairwise_bernoulli',
        'p': 1.0,
        'allow_autapses': False
    },
    syn_spec_inh = {
        'synapse_model': 'static_synapse',
        'weight': -k * nest.spatial.distance,
        'delay': 1.0
    },
    ):
    #~*~*~*~* START  *~*~*~*~#
    """
    Create a NEST neuron population at specified 3D positions.

    Args:
        conn_dict_ex, syn_spec_ex, conn_dict_inh, syn_spec_inh: dicts for the exh/inh synapses
        k(float): pA/mm
        plot(bool): the usual
        bidirectional(bool): creates exhib. Connections clockwise AND counterclockwise
        positions (np.ndarray): Array of shape (N,3) with (x,y,z) coordinates.
        model (str):           Name of the NEST neuron model to instantiate.

    Returns:
        list: List of NEST node IDs corresponding to the created neurons.
    """    
    #~*~*~*~* END  *~*~*~*~#
    
    
    positions=positions.tolist()
    nodes = nest.Create(model=model,positions=nest.spatial.free(pos=positions))
    
    if(plot):
        nest.PlotLayer(nodes)

    
    # connect the ring
    count=0

    for i in nodes:
        nest.Connect(i,nodes[(count+1)%len(nodes)],conn_dict_ex,syn_spec_ex)
        if(bidirectional):
            nest.Connect(nodes[(count+1)%len(nodes)],i,conn_dict_ex,syn_spec_ex)
            # for the case you don't want a signalgenerator but rather only one active area which can freely
            # slide in every direction on the circle 
        count+=1

    nest.Connect(nodes, nodes, conn_dict_inh, syn_spec_inh)

    # klappt.
    return nodes


def CCW_spike_recorder(ccw):
        
    #~*~*~*~* START  *~*~*~*~#
    """
    simply attaches spike recorder to every neuron in the CCW
    returns a list of tuples countaining theta->[n][0] (the angle relative to (1,0))
    in radians and the corresponding recorder->[n][1]
    """
    #~*~*~*~* END  *~*~*~*~#

    length = len(ccw)
    recorder_list = []
    for i, neuron in enumerate(ccw):
        theta = (i/length) * 2*np.pi
        spikerecorder = nest.Create("spike_recorder")
        nest.Connect(neuron, spikerecorder)
        recorder_list.append((theta,spikerecorder))#damit klar ist, in welche Richtung der Agent läuft
    return recorder_list


def connect_cone(
    cone_points,
    k=10.0,
    model='iaf_psc_alpha',
    # 80 % exhib
    conn_dict_ex = {
        'rule': 'pairwise_bernoulli',
        'p': 0.8,
        'allow_autapses': False
    },
    syn_spec_ex = {
        'synapse_model': 'static_synapse',
        'weight': (((nest.spatial.target_pos.z - nest.spatial.source_pos.z)**2)**0.5),
        'delay': 1.0
    },
    # 20 % inhib
    conn_dict_inh = {
        'rule': 'pairwise_bernoulli',
        'p': 0.2,
        'allow_autapses': False
    },
    syn_spec_inh = {
        'synapse_model': 'static_synapse',
        'weight': -k * (
            ((nest.spatial.target_pos.x - nest.spatial.source_pos.x)**2
           + (nest.spatial.target_pos.y - nest.spatial.source_pos.y)**2)
           ** 0.5# dependency on x-y euclidean distance
        ),
        'delay': 1.0
    }
    ):
     #~*~*~*~* START *~*~*~*~#

    """
    Create a recurrent network on cone-shaped positions with both excitatory and inhibitory connections.

    Args:
        cone_points (np.ndarray): Array of shape (N,3) with (x,y,z) coordinates of cone.
        k (float):              Scaling factor for inhibitory synaptic weights.
        conn_dict_ex (dict):    Excitatory connection rule for nest.Connect.
        syn_spec_ex (dict):     Excitatory synapse specifications; weight func of z-distance.
        conn_dict_inh (dict):   Inhibitory connection rule for nest.Connect.
        syn_spec_inh (dict):    Inhibitory synapse specifications; weight func of xy-distance.

    Returns:
        list: NEST node IDs of the created cone network.
    """
    #~*~*~*~* END *~*~*~*~#
    
    nodes = nest.Create(model=model, positions=nest.spatial.free(pos=cone_points.tolist()))
    nest.Connect(nodes, nodes, conn_dict_ex, syn_spec_ex)
    nest.Connect(nodes, nodes, conn_dict_inh, syn_spec_inh)
    return nodes


def connect_cone_ccw(
        cone, 
        ccw,              
        syn_spec_strong={'synapse_model':'static_synapse', 'weight':10.0, 'delay':1.0}, 
        syn_spec_weak  ={'synapse_model':'static_synapse', 'weight': 2.0, 'delay':1.0}, 
        angle_width_deg=15.0
    ):
        #~*~*~*~* START  *~*~*~*~#

    """
    Connects two NEST populations based on their azimuthal (XY-plane) angular proximity.

    Each neuron in `cone_nodes` is connected to each neuron in `ccw_nodes`.
    If the angular difference between their positions (in the XY-plane) is within
    `angle_width_deg`, the connection uses `syn_spec_strong`, otherwise `syn_spec_weak`.

    Args:
        cone_nodes (List[int]): List of NEST node IDs for the "cone" population.
        ccw_nodes  (List[int]): List of NEST node IDs for the "ccw" population.
        angle_width_deg (float): Angular window (in degrees) for strong connections.
        syn_spec_strong (dict): Synapse specification for connections within the angular window.
        syn_spec_weak   (dict): Synapse specification for connections outside the window.
        conn_spec       (dict): Connection rule dictionary for nest.Connect.

    Returns:
        None: Connections are created in the NEST kernel; nothing is returned.
    """
        #~*~*~*~* END  *~*~*~*~#

    
    # costly preprocessing
    conn_spec   = {'rule': 'one_to_one'}
    angle_width = np.deg2rad(angle_width_deg)


    pos_ccw  = nest.GetPosition(ccw)
    pos_cone = nest.GetPosition(cone)

    theta_ccw  = [ (np.arctan2(y, x) + 2*np.pi) % (2*np.pi) for x, y, _ in pos_ccw ]
    theta_cone = [ (np.arctan2(y, x) + 2*np.pi) % (2*np.pi) for x, y, _ in pos_cone ]

    for idx_i, i in enumerate(ccw):
        ti = theta_ccw[idx_i]
        for idx_j, j in enumerate(cone):
            tj = theta_cone[idx_j]
            delta = abs(tj - ti)
            delta = min(delta, 2*np.pi - delta)
            if delta < angle_width:
                nest.Connect(j, i, conn_spec, syn_spec_strong)
            else:
                nest.Connect(j, i, conn_spec, syn_spec_weak)


def create_blob_population(
    positions,
    conn_ex = {'rule': 'pairwise_bernoulli', 'p': 0.8, 'allow_autapses': False},
    syn_ex  = {'synapse_model': 'static_synapse', 'weight': 2.0, 'delay': 1.0},
    conn_in = {'rule': 'pairwise_bernoulli', 'p': 0.2, 'allow_autapses': False},
    syn_in  = {'synapse_model': 'static_synapse','weight': -10.0,'delay': 1.0},
    plot= False,
    neuron_type = "iaf_psc_alpha"
    ):
        #~*~*~*~* START  *~*~*~*~#

    """
    Create and connect a NEST population (“blob”) at specified 3D positions.

    Args:
        positions (np.ndarray): Array of shape (N,3) with neuron coordinates.
        conn_ex (dict):       Excitatory connection rule for nest.Connect.
        syn_ex (dict):        Excitatory synapse parameters.
        conn_in (dict):       Inhibitory connection rule for nest.Connect.
        syn_in (dict):        Inhibitory synapse parameters.
        neuron_type (str):    Name of the NEST neuron model to instantiate.
        plot (bool):          If True, show a 3D scatter of positions and PlotLayer.

    Returns:
        list: The list of NEST node IDs created.
    """
        #~*~*~*~* END  *~*~*~*~#

    blob_pop = nest.Create(
        neuron_type,
        positions.shape[0],
        positions=nest.spatial.free(positions.tolist())
    )

    nest.Connect(blob_pop, blob_pop, conn_ex, syn_ex)
    nest.Connect(blob_pop, blob_pop, conn_in, syn_in)

    if plot:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(positions[:,0], positions[:,1], positions[:,2], s=2)
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        plt.show()
        nest.PlotLayer(blob_pop)
    return blob_pop

def blob(n = 100,
    m = np.zeros(3),
    r = 1.0,
    scaling_factor = 1.0,
    conn_ex = {'rule': 'pairwise_bernoulli', 'p': 0.8, 'allow_autapses': False},
    syn_ex  = {'synapse_model': 'static_synapse', 'weight': 2.0, 'delay': 1.0},
    conn_in = {'rule': 'pairwise_bernoulli', 'p': 0.2, 'allow_autapses': False},
    syn_in  = {'synapse_model': 'static_synapse','weight': -10.0,'delay': 1.0},
    plot= False,
    neuron_type = "iaf_psc_alpha"    
        ):
            #~*~*~*~* START  *~*~*~*~#
    """
    Create a spatially distributed “blob” of NEST neurons and interconnect them
    with specified excitatory and inhibitory rules.

    This function first generates `n` random 3D points within a sphere of radius `r`
    centered at `m`, scaled by `scaling_factor`. It then instantiates a NEST population
    of size `n` at these coordinates and connects the neurons pairwise according to
    the provided excitatory and inhibitory connection parameters.

    Args:
        n (int): Number of neurons to generate in the blob.
        m (np.ndarray): 3-vector specifying the center of the sphere.
        r (float): Radius of the sphere used for point generation.
        scaling_factor (float): Uniform scaling factor applied to all coordinates.
        conn_ex (dict): NEST connection rule for excitatory synapses.
        syn_ex (dict): Parameters for excitatory synapses (model, weight, delay).
        conn_in (dict): NEST connection rule for inhibitory synapses.
        syn_in (dict): Parameters for inhibitory synapses (model, weight, delay).
        plot (bool): If True, display a 3D scatter of neuron positions and invoke
                     NEST’s PlotLayer.
        neuron_type (str): Name of the NEST neuron model to create (e.g., "iaf_psc_alpha").

    Returns:
        list: List of NEST node IDs corresponding to the created neurons.
    """
    
            #~*~*~*~* END  *~*~*~*~#

    
    
    # Creates Blob.
    pos = blob_positions(
        n=n,
        m=np.zeros(3),
        r=1.0,
        scaling_factor=SCALING_FACTOR
    )
    blob_pop = create_blob_population(
        positions=pos,
        conn_ex=conn_ex,
        syn_ex=syn_ex,
        conn_in=conn_in,
        syn_in=syn_in,
        plot=plot,
        neuron_type = neuron_type
    )
    
    
    return blob_pop


def connect_blob_cone(
    blob,
    cone,
    m = np.zeros(3),
    angle_width_deg = 10.0,
    conn_ex_c = {'rule': 'pairwise_bernoulli', 'p': 0.6, 'allow_autapses': False},
    conn_in_c  = {'rule': 'pairwise_bernoulli', 'p': 0.4, 'allow_autapses': False},
    generic_conn = {'rule': 'pairwise_bernoulli', 'p': 0.01, 'allow_autapses': False},
    generic_syn = {'synapse_model': 'static_synapse', 'weight': 1.0, 'delay': 1.5}
):
        #~*~*~*~* START  *~*~*~*~#

    """
    Connects two NEST populations ("blob" → "cone" and back) using:
      1. generic blob→cone connections (generic_conn, generic_syn)
      2. angle‐dependent cone→blob connections:
         - if Δθ ≤ angle_width_deg: excitatory (conn_ex_c, syn_ex_c)
         - else: inhibitory (conn_in_c, syn_in_c), with delay clipped to ≥1 ms

    """
        #~*~*~*~* END  *~*~*~*~#

    nest.Connect(blob, cone, generic_conn, generic_syn)


    half = np.deg2rad(angle_width_deg)
    pos_blob = np.array(nest.GetPosition(blob))
    pos_cone = np.array(nest.GetPosition(cone))
    th_blob  = (np.arctan2(pos_blob[:,1],pos_blob[:,0]) + 2*np.pi)% (2*np.pi)
    th_cone  = (np.arctan2(pos_cone[:,1],pos_cone[:,0]) + 2*np.pi)% (2*np.pi)
    blob_ids = nest.GetStatus(blob, 'global_id')
    cone_ids = nest.GetStatus(cone, 'global_id')

    pre_exc, post_exc, w_exc, d_exc = [], [], [], []
    pre_inh, post_inh, w_inh, d_inh = [], [], [], []

    for bid, tb, pB in zip(blob_ids, th_blob, pos_blob):
        for cid, tc, pC in zip(cone_ids, th_cone, pos_cone):
            delta = abs(tc - tb)
            delta = min(delta, 2*np.pi - delta)
            if delta <= half:
                pre_exc.append(cid)
                post_exc.append(bid)

                dist = np.linalg.norm(pC - pB)
                w_exc.append(2.4 + 1.2 * dist)
                d_exc.append(1.0)
            else:
                pre_inh.append(cid)
                post_inh.append(bid)
                dx, dy = pC[0] - m[0], pC[1] - m[1]
                w_inh.append(1.0 + dx*dx + dy*dy)
                raw = np.random.exponential(scale=2.0) * np.linalg.norm(pC - pB)
                d_inh.append(max(raw, 1.0))

    if pre_exc:
        nest.Connect(
            pre_exc, post_exc,
            {'rule':'one_to_one'},
            {'synapse_model':'static_synapse',
             'weight': np.array(w_exc),
             'delay':  np.array(d_exc)}
        )
    if pre_inh:
        nest.Connect(
            pre_inh, post_inh,
            {'rule':'one_to_one'},
            {'synapse_model':'static_synapse',
             'weight': np.array(w_inh),
             'delay':  np.array(d_inh)}  # now >= 1
        )


def grid2visual(
                grid,
                K=10,
                m=np.array([0.0,0.0,0.0]),
                syn_dict_ex = {
                    "synapse_model": "static_synapse",
                    "weight": 50.0,    
                    "delay": 1.0       
                },
                syn_dict_in = {
                    "synapse_model": "static_synapse",
                    "weight": -40.0,   
                    "delay": 1.0       
                },
                conn_dict_ex = {
                    "rule": "pairwise_bernoulli",
                    "p": 0.7,                
                    "allow_autapses": False
                },
                syn_spec_ex = {
                    "synapse_model": "static_synapse",
                    "weight": 30.0,          
                    "delay": 1.5             
                },
                conn_dict_inh = {
                    "rule": "pairwise_bernoulli",
                    "p": 0.3,                
                    "allow_autapses": False
                },
                syn_spec_inh = {
                    "synapse_model": "static_synapse",
                    "weight": -60.0,         
                    "delay": 1.5             
                },
                conn_dict_layer = {
                    "rule": "all_to_all",    
                    "allow_autapses": False
                },
                syn_dict_layer = {
                    "synapse_model": "stdp_synapse",
                    "alpha": 1.0,            
                    "lambda": 0.01,          
                    "tau_plus": 20.0         
                }
    ):
        #~*~*~*~* START  *~*~*~*~#

    """
    Build a layered spiking neural network from a list of 3D grid layers,
    apply feed-forward Poisson input and noise to the first layer, connect
    successive layers with STDP synapses, and project the final layer into
    a visual blob population.

    Args:
        grid: A list of 3D coordinate arrays, one per layer (shape (Ni, 3)).
        K: Unused placeholder parameter for future extensions.
        syn_dict_ex: Excitatory synapse spec for Poisson input to the first layer.
        syn_dict_in: Inhibitory synapse spec for noise input to the first layer.
        conn_dict_ex: Connection rule for the final layer → blob excitatory mapping.
        syn_spec_ex: Synapse spec for the final layer → blob excitatory mapping.
        conn_dict_inh: Connection rule for the final layer → blob inhibitory mapping.
        syn_spec_inh: Synapse spec for the final layer → blob inhibitory mapping.
        conn_dict_layer: Connection rule for inter-layer connections.
        syn_dict_layer: Synapse spec (STDP) for inter-layer connections.

    Returns:
        poisson_generators: IDs of Poisson generators providing excitatory input.
        noise_generators: IDs of Poisson generators providing inhibitory noise.
        populations: List of neuron ID lists, one for each grid layer.
        blob_pop: Neuron ID list of the final visual blob population.
    """
        #~*~*~*~* END  *~*~*~*~#

    
    length = 0
    populations = []
    poisson_generators = []
    noise_generators = []
    
    for i,layer in enumerate(grid):
        length+=1
        pop = nest.Create("iaf_psc_alpha",len(layer.tolist()),positions=nest.spatial.free(layer.tolist()))
        if i==0:
            for neuron in pop:
                ex = nest.Create("poisson_generator")
                noise = nest.Create("poisson_generator")
                nest.Connect(ex, neuron, syn_spec=syn_dict_ex)
                nest.Connect(noise, neuron, syn_spec=syn_dict_in)
                poisson_generators.append(ex)
                noise_generators.append(noise)
                #simnulate the receptive field through poisson generators 
        populations.append(pop)
       
        
    
    blob_pop = blob(n=800,plot=False,m=m)
    
    
    for idx in range(len(populations)-1):# all except for the last layer
        nest.Connect(
            populations[idx],
            populations[idx+1],
            conn_spec=conn_dict_layer,
            syn_spec=syn_dict_layer
        )
    
    # Final layer -> visual blob: inhibitory mapping
    
    nest.Connect(populations[-1], blob_pop, conn_dict_ex, syn_spec_ex)
    nest.Connect(populations[-1], blob_pop, conn_dict_inh, syn_spec_inh)
    
    return poisson_generators, noise_generators, populations, blob_pop


def input_to_receptive_field(image_array, poisson_generators, max_rate=200.0):
    """
    Map a grayscale image into Poisson input rates for a corresponding array of spike generators.

    Each pixel (0–255) is linearly scaled to a firing rate between 0 and `max_rate` Hz,
    then assigned to the matching Poisson generator.

    Args:
        image_array (np.ndarray):
            Input image of arbitrary shape, with values in [0, 255].
        poisson_generators (List[int]):
            List of NEST Poisson generator IDs; length must match image_array.size.
        max_rate (float):
            Maximum firing rate (Hz) corresponding to pixel value 255.

    Notes:
        - If `image_array.size` ≠ `len(poisson_generators)`, the mismatch is currently ignored.
          You may wish to add error handling for length mismatches.
    """
    if image_array.size != len(poisson_generators):
        print("POISSON_GENERATOR ANZAHL ENTSPRICHT NICHT DER GRÖßE DES IMAGE_ARRAYS!!!ELF\nTipp: Funktion input_to_receptive_field ist das Problem\n")
        pass # da es noch genug zu tun gibt, überspringe ich jetzt sowas. WITH GREAT POWER COMES GREAT RESPONSIBILITY


    flat_image = image_array.flatten()
    scaled_rates = (flat_image / 255.0) * max_rate

    for i, rate in enumerate(scaled_rates):#the timing scares me.
        nest.SetStatus([poisson_generators[i]], {'rate': float(rate)})



def clusters_to_neurons(
    clusters, 
    neuron_models
    ):
    """
    Create NEST neuron populations from clusters of 3D coordinates.

    Args:
        clusters (List[np.ndarray]):
            A list of NumPy arrays, each of shape (K_i, 3), representing the (x, y, z)
            positions of K_i neurons in cluster i.
        neuron_models (List[str]):
            A list of NEST neuron model names (e.g. "iaf_psc_alpha", "izhikevich",
            "hh_psc_alpha"), one for each cluster. Must have the same length as `clusters`.

    Returns:
        List[nest.NodeCollection]:
            A list of NodeCollections, one per cluster, containing the created neurons
            of the specified model placed at the given positions.
    """
    populations = []
    for idx, pts in enumerate(clusters):
        model = neuron_models[idx]

        n_cells = pts.shape[0]

        nodes = nest.Create(n=n_cells, model=model,positions=nest.spatial.free(pos=pts.tolist()))


        populations.append(nodes)
    return populations

def xy_distance(a, b):
    # ignore z: lateral distance
    return np.linalg.norm(a[:2] - b[:2])

def eye_lgn_layer(gsl = 16,plot=False): # grid side length
    eye_layer_size = [gsl,gsl,gsl,gsl,gsl,gsl,gsl,gsl] #r,g,b,light,ganglia_r,ganglia_g, ganglia_b

    lgn_layer_size = [gsl,gsl,gsl,gsl,gsl] #not so sure anymore about 2x2 sight split. 
    # maybe just attach to different layers
    # r,g,b,light,merge_layer-> project
    # gsl grid side length


    eye_1 = create_Grid(m = np.array([1,gsl,10]),grid_size_list = eye_layer_size, plot = False, rot_phi = 0)
    eye_2 = create_Grid(m = np.array([1,-2*gsl,10]),grid_size_list = eye_layer_size, plot = False, rot_phi = 0)


    LGN_1 = create_Grid(m = np.array([20,gsl,15]),grid_size_list = lgn_layer_size, plot = False, rot_phi = 0)
    LGN_2 = create_Grid(m = np.array([20,-2*gsl,15]),grid_size_list = lgn_layer_size, plot = False, rot_phi = 0)



    V1_projection = create_Grid(m = np.array([10,0,0]),grid_size_list=[gsl],rot_theta=0.0,rot_phi=90)



    if plot:
        eyes = eye_1 + eye_2
        plot_point_clusters(eyes,marker_size = 10, alpha = 0.5,linewidths=0.1)
        plot_point_clusters_normalized(eyes,marker_size = 2, alpha = 0.5,linewidths=0.1)
        lgns = LGN_1 + LGN_2
        plot_point_clusters(lgns,marker_size = 10, alpha = 0.5,linewidths=0.1)
        plot_point_clusters_normalized(lgns,marker_size = 2, alpha = 0.5,linewidths=0.1)
    return [eye_1,eye_2,LGN_1,LGN_2,V1_projection]


def vis_cortex_pos(
    gsl          = 16,           # grid side length
    col_height   = 2.0,          
    cells_per_col= 140,          
    rot_xy       = (0.0, 90.0),        
    grid_size        = None,
    rot_theta        = 0,
    rot_phi          = 0,
    transform_matrix = np.diag([1,1,1]),
    dt               = 0.01,
    area_specs = None,
    flow_functions = None,
    experiments = None,
    plot=False):

    if grid_size is None:
        grid_size=np.array([gsl,gsl,gsl])
    if area_specs is None:
        area_specs = {
            "V1": dict(offset_x=10, inner=0.10, outer=0.15),
            "V2": dict(offset_x= 8, inner=0.15, outer=0.20),
            "V3": dict(offset_x= 6, inner=0.20, outer=0.30),
            "V4": dict(offset_x= 4, inner=0.30, outer=0.40),
            "V5": dict(offset_x= 2, inner=0.40, outer=0.45),
            "V6": dict(offset_x= 0, inner=0.45, outer=0.50),
        }
    if experiments is None:
        experiments = {
            "V1C": dict(
                m               = np.array([0,0,-9]),
                wave_params     = dict(sparse_holes=0,
                                       sparsity_factor=0.8,
                                       probability_vector=[0.35,0.25,0.10,0.05,0.20,0.05]),
                types           = [0,1,2,3,4,5],
                num_steps       = 10,
                title           = "V1 computational body"
            ),

            "V2C": dict(
                m               = np.array([0,0,-7]),
                wave_params     = dict(sparse_holes=0,
                                       sparsity_factor=0.8,
                                       probability_vector=[0.30,0.15,0.20,0.05,0.25]),
                types           = [0,1,2,3,4],
                num_steps       = 10,
                title           = "V2 computational body"
            ),
                "V3C": dict(
                m               = np.array([0,0,-5]),
                wave_params     = dict(sparse_holes=0,
                                       sparsity_factor=0.8,
                                       probability_vector=[0.3,0.15,0.15,0.05,0.2]),
                types           = [0,1,2,3,4],
                num_steps       = 10,
                title           = "V3 computational body"
            ),
                "V4C": dict(
                m               = np.array([0,0,-3]),
                wave_params     = dict(sparse_holes=0,
                                       sparsity_factor=0.8,
                                       probability_vector=[0.25,0.25,0.1,0.05,0.15,0.1]),
                types           = [0,1,2,3,4,5],
                num_steps       = 10,
                title           = "V4 computational body"
            ),
                "V5C": dict(
                m               = np.array([0,0,-1]),
                wave_params     = dict(sparse_holes=0,
                                       sparsity_factor=0.8,
                                       probability_vector=[0.45,0.25,0.1,0.1,0.1]),
                types           = [0,1,2,3,4],
                num_steps       = 10,
                title           = "V5 computational body"
            ),
                "V6C": dict(
                m               = np.array([0,0,1]),
                wave_params     = dict(sparse_holes=0,
                                       sparsity_factor=0.8,
                                       probability_vector=[0.35,0.25,0.1,0.05,0.2,0.05]),
                types           = [0,1,2,3,4,5],
                num_steps       = 10,
                title           = "V6 computational body"
            )
        }


    if flow_functions is None:
        f1 = lambda x,y,z : 1.0/(0.3*x**2 + 0.3*y**2 + 0.15)
        flow_functions = [(lambda x,y,z:x,
                           lambda x,y,z:y,
                           f1)] * 6  # same tripples, list operation



    results = {}
    all_columns  = {} 



    for name, spec in area_specs.items():

        grid_layers   = create_Grid(
            m           = np.array([spec["offset_x"], 0, 0]),
            grid_size_list=[gsl],
            rot_theta   = rot_xy[0],
            rot_phi     = rot_xy[1],
            plot        = False
        )
        projection_xy = grid_layers[0]

        columns = []
        for center in projection_xy:
            col_pts = create_cone(
                m             = center,
                n             = cells_per_col,
                inner_radius  = spec["inner"],
                outer_radius  = spec["outer"],
                height        = col_height,
                rot_theta     = 0.0,
                rot_phi       = 0.0,
                plot          = False
            )
            columns.append(col_pts)
        all_columns[name] = columns




    for name, cfg in experiments.items():
        res = cluster_and_flow(
            grid_size      = grid_size,
            m              = cfg["m"],
            rot_theta      = rot_theta,
            rot_phi        = rot_phi,
            transform_matrix = transform_matrix,
            wave_params    = cfg["wave_params"],
            types          = cfg["types"],
            flow_functions = flow_functions,
            dt             = dt,
            num_steps      = cfg["num_steps"],
            plot_clusters  = plot,
            title          = cfg["title"]
        )
        results[name]=res


    if plot:
        for name, cols in all_columns.items():
            plot_point_clusters(cols,
                                marker_size = 2,
                                alpha       = 0.6,
                                linewidths  = 0.1,
                                title       = f"{name}-Säulen ({len(cols)} Stück)")
    return results, all_columns



def vis2neurons(
    Vn_pop,
    area2models=None,
    plot=False
    ):
    
    model_alias = {"iaf_cond_beta_gap": "iaf_cond_alpha"}
    if(area2models is None):
        area2models = {
            "V1": ["iaf_cond_alpha","aeif_cond_alpha","iaf_cond_exp",
                   "hh_psc_alpha","iaf_cond_alpha","iaf_cond_beta_gap"],
            "V2": ["aeif_cond_alpha","iaf_cond_exp","aeif_cond_alpha",
                   "hh_psc_alpha","iaf_cond_alpha"],
            "V3": ["aeif_cond_alpha","iaf_cond_exp","aeif_cond_alpha",
                   "hh_psc_alpha","iaf_cond_alpha"],               
            "V4": ["aeif_cond_alpha","iaf_cond_exp","hh_psc_alpha",
                   "gif_cond_exp","iaf_cond_alpha","iaf_cond_beta_gap"],
            "V5": ["iaf_cond_alpha","aeif_cond_alpha","hh_psc_alpha",
                   "izhikevich","iaf_cond_beta_gap"],
            "V6": ["iaf_cond_alpha","aeif_cond_alpha","hh_psc_alpha",
                   "iaf_psc_delta","iaf_cond_alpha","iaf_cond_beta_gap"]
        }
    Vn_neurons = {}
    for exp_name, clusters in Vn_pop.items():
        models = [model_alias.get(m, m) for m in area2models[exp_name[:2]]]
        models = (models * ((len(clusters) + len(models) - 1) // len(models)))[:len(clusters)]
        Vn_neurons[exp_name] = clusters_to_neurons(clusters, models)
    if(plot):
        for exp_name, cluster_pops in Vn_neurons.items():    
            for pop in cluster_pops:                         
                nest.PlotLayer(pop)
    return Vn_neurons

def generate_direction_similarity_matrix(
    pop1=None,
    pop2=None,
    vecfield1=(
        lambda x, y, z: 1.5 * x ** 2 + y + z,
        lambda x, y, z: 1.3 * x ** 3 - y + z,
        lambda x, y, z: 1.7 * x ** 2 + y - z,
    ),
    vecfield2=(
        lambda x, y, z: 4.7 * x - np.sin(y),
        lambda x, y, z: -1.9 * x ** 2 - y + 2,
        lambda x, y, z: 1.7 * y ** 2 + x - y,
    ),
    eps: float = 1e-12,
    plot: bool = False,
    cmap: str = "plasma",
    arrow_length: float = 0.2,
):
    """
    Compute a cosine-similarity weight matrix W between two 3-D vector fields.
    The fields are evaluated on point clouds *pop1* (sources) and *pop2*
    (targets).  If either population is *None*, a default 4 × 4 × 4 grid is
    generated and transformed with hard-coded parameters.

    Parameters
    ----------
    pop1, pop2 : np.ndarray | None
        Arrays with shape (N, 3) and (M, 3) giving the coordinates of the
        source and target neurons.  When *None*, a default grid is built via
        ``generate_qube`` and ``transform_points`` (see example).
    vecfield1, vecfield2 : tuple(callable, callable, callable)
        Component functions (f₁, f₂, f₃) mapping (x, y, z) → ℝ for each field.
    eps : float
        Floor for vector norms to avoid division by zero.
    plot : bool
        Forwarded to ``field`` to display 3-D quiver plots.
    cmap : str
        Matplotlib colour map used when *plot=True*.
    arrow_length : float
        Arrow scale in the quiver plots.

    Returns
    -------
    numpy.ndarray
        Weight matrix W with shape (len(pop2), len(pop1)); entries are cosine
        similarities in [-1, 1].

    Example
    -------
    >>> W = generate_direction_similarity_matrix()        # defaults
    >>> nest.Connect(pop1_nodes, pop2_nodes,
    ...              syn_spec={'weight': W.astype(np.float32)})
    """
    # ---------------------------------------------------------------------
    if pop1 is None:
        grid = generate_qube((4, 4, 4))
        pop1 = transform_points(
            grid, [1, 2, 3], 120, 120, transform_matrix=np.eye(3), plot=plot
        )
    if pop2 is None:
        grid = generate_qube((4, 4, 4))
        pop2 = transform_points(
            grid, [3, 3, 1], 24, 77, transform_matrix=np.eye(3), plot=plot
        )

    f1v1, f2v1, f3v1 = vecfield1
    f1v2, f2v2, f3v2 = vecfield2

    vec1 = field(pop1, plot=plot, normalize=True, cmap=cmap,
                 arrow_length=arrow_length, f1=f1v1, f2=f2v1, f3=f3v1)
    vec2 = field(pop2, plot=plot, normalize=True, cmap=cmap,
                 arrow_length=arrow_length, f1=f1v2, f2=f2v2, f3=f3v2)

    u1 = vec1 / np.clip(np.linalg.norm(vec1, axis=1, keepdims=True), eps, None)
    u2 = vec2 / np.clip(np.linalg.norm(vec2, axis=1, keepdims=True), eps, None)

    return u2 @ u1.T


def generate_direction_similarity_matrix2(
    grid_shape=(4, 4, 4),
    transform1=None,
    transform2=None,
    stretch1=(1, 1, 1),
    stretch2=(1, 1, 1),
    vecfield1=(
        lambda x, y, z: 1.5 * x ** 2 + y + z,
        lambda x, y, z: 1.3 * x ** 3 - y + z,
        lambda x, y, z: 1.7 * x ** 2 + y - z,
    ),
    vecfield2=(
        lambda x, y, z: 4.7 * x - np.sin(y),
        lambda x, y, z: -1.9 * x ** 2 - y + 2,
        lambda x, y, z: 1.7 * y ** 2 + x - y,
    ),
    eps=1e-12,
    plot=False,
    cmap="plasma",
    arrow_length=0.2,
):
    """
    Generate a cosine-similarity weight matrix between two 3-D vector fields
    evaluated on two separately transformed point grids.  See full docstring
    below for details.
    """
    if transform1 is None:
        transform1 = dict(m=[1, 2, 3], rot_theta=120, rot_phi=120)
    if transform2 is None:
        transform2 = dict(m=[3, 3, 1], rot_theta=24, rot_phi=77)

    grid = generate_qube(grid_shape)                       
    pop1 = transform_points(
        grid,
        transform1["m"],
        transform1["rot_theta"],
        transform1["rot_phi"],
        transform_matrix=np.diag(stretch1),
        plot=plot,
    )
    pop2 = transform_points(
        grid,
        transform2["m"],
        transform2["rot_theta"],
        transform2["rot_phi"],
        transform_matrix=np.diag(stretch2),
        plot=plot,
    )

    f1v1, f2v1, f3v1 = vecfield1
    f1v2, f2v2, f3v2 = vecfield2

    vec1 = field(pop1, plot=plot, normalize=True, cmap=cmap,
                 arrow_length=arrow_length, f1=f1v1, f2=f2v1, f3=f3v1)
    vec2 = field(pop2, plot=plot, normalize=True, cmap=cmap,
                 arrow_length=arrow_length, f1=f1v2, f2=f2v2, f3=f3v2)

    u1 = vec1 / np.clip(np.linalg.norm(vec1, 1, keepdims=True), eps, None)
    u2 = vec2 / np.clip(np.linalg.norm(vec2, 1, keepdims=True), eps, None)

    return u2 @ u1.T


