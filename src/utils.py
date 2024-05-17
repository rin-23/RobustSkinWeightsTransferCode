import igl
import numpy as np
import math
import scipy as sp

def find_closest_point_on_surface(P, V, F):
    """
    Given a number of points find their closest points on the surface of the V,F mesh

    Args:
        P: #P by 3, where every row is a point coordinate
        V: #V by 3 mesh vertices
        F: #F by 3 mesh triangles indices
    Returns:
        sqrD #P smallest squared distances
        I #P primitive indices corresponding to smallest distances
        C #P by 3 closest points
        B #P by 3 of the barycentric coordinates of the closest point
    """
    
    sqrD,I,C = igl.point_mesh_squared_distance(P, V, F)

    F_closest = F[I,:]
    V1 = V[F_closest[:,0],:]
    V2 = V[F_closest[:,1],:]
    V3 = V[F_closest[:,2],:]

    B = igl.barycentric_coordinates_tri(C, V1, V2, V3)

    return sqrD,I,C,B

def interpolate_attribute_from_bary(A,B,I,F):
    """
    Interpolate per-vertex attributes A via barycentric coordinates B of the F[I,:] vertices

    Args:
        A: #V by N per-vertex attributes
        B  #B by 3 array of the barycentric coordinates of some points
        I  #B primitive indices containing the closest point
        F: #F by 3 mesh triangle indices
    Returns:
        A_out #B interpolated attributes
    """
    F_closest = F[I,:]
    a1 = A[F_closest[:,0],:]
    a2 = A[F_closest[:,1],:]
    a3 = A[F_closest[:,2],:]

    b1 = B[:,0]
    b2 = B[:,1]
    b3 = B[:,2]

    b1 = b1.reshape(-1,1)
    b2 = b2.reshape(-1,1)
    b3 = b3.reshape(-1,1)
    
    A_out = a1*b1 + a2*b2 + a3*b3

    return A_out

def normalize_vec(v):
    return v/np.linalg.norm(v)


def find_matches_closest_surface(V1, F1, N1, V2, F2, N2, W1, dDISTANCE_THRESHOLD_SQRD, dANGLE_THRESHOLD_DEGREES):
    """
    For each vertex on the target mesh find a match on the source mesh.

    Args:
        V1: #V1 by 3 source mesh vertices
        F1: #F1 by 3 source mesh triangles indices
        N1: #V1 by 3 source mesh normals
        
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        N2: #V2 by 3 target mesh normals
        
        W1: #V1 by num_bones source mesh skin weights

        dDISTANCE_THRESHOLD_SQRD: scalar distance threshold
        dANGLE_THRESHOLD_DEGREES: scalar normal threshold

    Returns:
        Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
        W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
    """

    Matched = np.zeros(shape = (V2.shape[0]), dtype=bool)
    sqrD,I,C,B = find_closest_point_on_surface(V2,V1,F1)
    
    # for each closest point on the source, interpolate its per-vertex attributes(skin weights and normals) 
    # using the barycentric coordinates
    W2 = interpolate_attribute_from_bary(W1,B,I,F1)
    N1_match_interpolated = interpolate_attribute_from_bary(N1,B,I,F1)
    
    # check that the closest point passes our distance and normal thresholds
    for RowIdx in range(0, V2.shape[0]):
        n1 = normalize_vec(N1_match_interpolated[RowIdx,:])
        n2 = normalize_vec(N2[RowIdx, :])
        rad_angle = np.arccos(np.dot(n1, n2))
        deg_angle = math.degrees(rad_angle)
        if sqrD[RowIdx] <= dDISTANCE_THRESHOLD_SQRD and deg_angle <= dANGLE_THRESHOLD_DEGREES:
            Matched[RowIdx] = True

    return Matched, W2

def is_valid_array(sparse_matrix):
    has_invalid_numbers = np.isnan(sparse_matrix.data).any() or np.isinf(sparse_matrix.data).any()
    return not has_invalid_numbers

def inpaint(V2, F2, W2, Matched):
    """
    Inpaint weights for all the vertices on the target mesh for which  we didnt 
    find a good match on the source (i.e. Matched[i] == False).

    Args:
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
        Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh

    Returns:
        W_inpainted: #V2 by num_bones, final skinning weights where we inpainted weights for all vertices i where Matched[i] == False
        success: true if inpainting succeeded, false otherwise
    """

    # Compute the laplacian
    L = 2*igl.cotmatrix(V2, F2)
    M = igl.massmatrix(V2, F2, igl.MASSMATRIX_TYPE_VORONOI)
    Minv = sp.sparse.diags(1 / M.diagonal())

    is_valid = is_valid_array(L)
    if (not is_valid):
        print("[Error] Laplacian is invalid:")

    is_valid = is_valid_array(Minv)
    if (not is_valid):
        print("[Error] Mass matrix is invalid:")

    Q = -L + L*Minv*L

    is_valid = is_valid_array(Q)
    if (not is_valid):
        print("[Error] System matrix is invalid:")
    
    Aeq = sp.sparse.csc_matrix((0, 0))
    Beq = np.array([])
    B = np.zeros(shape = (L.shape[0], W2.shape[1]))

    b = np.array(range(0, int(V2.shape[0])), dtype=int)
    b = b[Matched]
    bc = W2[Matched,:]

    results, W_inpainted = igl.min_quad_with_fixed(Q, B, b, bc, Aeq, Beq, True)

    return W_inpainted, results

def smooth(V2, F2, W2, Matched, dDISTANCE_THRESHOLD, num_smooth_iter_steps=10, smooth_alpha=0.2):
    """
    Smooth weights in the areas for which weights were inpainted and also their close neighbours.

    Args:
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        W2: #V2 by num_bones skinning weights
        Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
        dDISTANCE_THRESHOLD_SQRD: scalar distance threshold
        num_smooth_iter_steps: scalar number of smoothing steps
        smooth_alpha: scalar the smoothing strength

    Returns:
        W2_smoothed: #V2 by num_bones new smoothed weights
        VIDs_to_smooth: 1D array of vertex IDs for which smoothing was applied
    """

    NotMatched = ~Matched
    VIDs_to_smooth = np.array(NotMatched, copy=True)

    adj_list = igl.adjacency_list(F2)

    def get_points_within_distance(V, VID, distance=dDISTANCE_THRESHOLD):
        """
        Get all neighbours of vertex VID within dDISTANCE_THRESHOLD
        """

        queue = []
        queue.append(VID)
        while len(queue) != 0:
            vv = queue.pop()
            neigh = adj_list[vv]
            for nn in neigh:
                if ~VIDs_to_smooth[nn] and np.linalg.norm(V[VID,:]-V[nn]) < distance:
                    VIDs_to_smooth[nn] = True
                    if nn not in queue:
                        queue.append(nn)
                        

    for i in range(0, V2.shape[0]):
        if NotMatched[i]:
            get_points_within_distance(V2, i)

    W2_smoothed = np.array(W2, copy=True)
    for step_idx in range(0, num_smooth_iter_steps):
        for i in range(0, V2.shape[0]):
            if VIDs_to_smooth[i]:
                neigh = adj_list[i]
                num = len(neigh)
                weight = W2_smoothed[i,:]

                new_weight = (1-smooth_alpha)*weight
                for influence_idx in neigh:
                    weight_connected = W2_smoothed[influence_idx,:]
                    new_weight += (weight_connected / num) * smooth_alpha
                
                W2_smoothed[i,:] = new_weight

    return W2_smoothed, VIDs_to_smooth
   
