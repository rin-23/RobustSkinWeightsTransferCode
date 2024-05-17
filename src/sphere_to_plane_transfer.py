from utils import find_matches_closest_surface, inpaint, smooth
import polyscope as ps
import igl
import numpy as np
import os 

# Initialize polyscope
ps.init()

# Get the directory of the current file
current_folder = os.path.dirname(os.path.abspath(__file__))

# Load the source mesh
V, F = igl.read_triangle_mesh(current_folder+"/../meshes/sphere.obj")
V1, F1,_,_ = igl.remove_unreferenced(V,F)
if (V.shape[0] != V1.shape[0]):
    print("[Warning] Source mesh has unreferenced vertices which were removed")
N1 = igl.per_vertex_normals(V1, F1)

# Load the target mesh
V, F = igl.read_triangle_mesh(current_folder+"/../meshes/grid.obj")
V2, F2,_,_ = igl.remove_unreferenced(V,F)
if (V.shape[0] != V2.shape[0]):
    print("[Warning] Target mesh has unreferenced vertices which were removed")
N2 = igl.per_vertex_normals(V2, F2)

# You can setup your own skin weights matrix W \in R^(|V1| x num_bones) here 
# W = np.load("source_skinweights.npy")

# For now, generate simple per-vertex data (can be skinning weights but can be any scalar data)
W = np.ones((V1.shape[0], 2)) # our simple rig has only 2 bones
W[:, 0] = 0.3 # first bone has an influence of 0.3 on all vertices
W[:, 1] = 0.7 # second bone has an influence of 0.7 on all vertices

num_bones = W.shape[1] 

# Register source and target Mesh geometries, plus their Normals
ps.register_surface_mesh("SourceMesh", V1, F1, smooth_shade=True)
ps.register_surface_mesh("TargetMesh", V2, F2, smooth_shade=True)
ps.get_surface_mesh("SourceMesh").add_vector_quantity("Normals", N1, defined_on='vertices', color=(0.2, 0.5, 0.5))
ps.get_surface_mesh("TargetMesh").add_vector_quantity("Normals", N2, defined_on='vertices', color=(0.2, 0.5, 0.5))

#
# Section 3.1 Closest Point Matching
#
dDISTANCE_THRESHOLD = 0.05*igl.bounding_box_diagonal(V2) # threshold distance D
dDISTANCE_THRESHOLD_SQRD = dDISTANCE_THRESHOLD *dDISTANCE_THRESHOLD
dANGLE_THRESHOLD_DEGREES = 30 # threshold angle theta in degrees

# for every vertex on the target mesh find the closest point on the source mesh and copy weights over
Matched, SkinWeights_interpolated = find_matches_closest_surface(V1,F1,N1,V2,F2,N2,W,dDISTANCE_THRESHOLD_SQRD,dANGLE_THRESHOLD_DEGREES)

# visualize vertices for which we found a match
ps.get_surface_mesh("TargetMesh").add_scalar_quantity("Matched", Matched, defined_on='vertices', cmap='blues')


#
# Section 3.2 Skinning Weights Inpainting
#
InpaintedWeights, success = inpaint(V2, F2, SkinWeights_interpolated, Matched)

if (success):
    # Visualize the weights for each bone
    ps.get_surface_mesh("TargetMesh").add_scalar_quantity("Bone1", InpaintedWeights[:,0], defined_on='vertices', cmap='blues')
    ps.get_surface_mesh("TargetMesh").add_scalar_quantity("Bone2", InpaintedWeights[:,1], defined_on='vertices', cmap='blues')

    # Optional smoothing
    SmoothedInpaintedWeights, VIDs_to_smooth = smooth(V2, F2, InpaintedWeights, Matched, dDISTANCE_THRESHOLD, num_smooth_iter_steps=10, smooth_alpha=0.2)
    ps.get_surface_mesh("TargetMesh").add_scalar_quantity("VIDs_to_smooth", VIDs_to_smooth, defined_on='vertices', cmap='blues')

    # Visualize the smoothed weights for each bone
    ps.get_surface_mesh("TargetMesh").add_scalar_quantity("SmoothedBone1", InpaintedWeights[:,0], defined_on='vertices', cmap='blues')
    ps.get_surface_mesh("TargetMesh").add_scalar_quantity("SmoothedBone2", InpaintedWeights[:,1], defined_on='vertices', cmap='blues')

    ps.show()

else:
    print("[Error] Inpainting failed.")