#include <igl/read_triangle_mesh.h>
#include <igl/per_vertex_normals.h>
#include <igl/bounding_box_diagonal.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/barycentric_coordinates.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/slice_mask.h>
#include <igl/adjacency_list.h>

#include <cmath>
#include <vector>
#include <queue>
#include <set>


/**
 * Given a number of points find their closest points on the surface of the V,F mesh
 * 
 *  P: #P by 3, where every row is a point coordinate
 *  V: #V by 3 mesh vertices
 *  F: #F by 3 mesh triangles indices
 *  sqrD #P smallest squared distances
 *  I #P primitive indices corresponding to smallest distances
 *  C #P by 3 closest points
 *  B #P by 3 of the barycentric coordinates of the closest point
 */
void find_closest_point_on_surface(const Eigen::MatrixXd& P, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, 
                                   Eigen::VectorXd& sqrD, Eigen::VectorXi& I, Eigen::MatrixXd& C, Eigen::MatrixXd& B)
{
    igl::point_mesh_squared_distance(P, V, F, sqrD, I, C);

    Eigen::MatrixXi F_closest = F(I, Eigen::all);
    Eigen::MatrixXd V1 = V(F_closest(Eigen::all, 0), Eigen::all);
    Eigen::MatrixXd V2 = V(F_closest(Eigen::all, 1), Eigen::all);
    Eigen::MatrixXd V3 = V(F_closest(Eigen::all, 2), Eigen::all);

    igl::barycentric_coordinates(C, V1, V2, V3, B);
}


/** 
 * Interpolate per-vertex attributes A via barycentric coordinates B of the F[I,:] vertices
 * 
 *  A: #V by N per-vertex attributes
 *  B  #B by 3 array of the barycentric coordinates of some points
 *  I  #B primitive indices containing the closest point
 *  F: #F by 3 mesh triangle indices
 *  A_out #B interpolated attributes
 */
void interpolate_attribute_from_bary(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                                     const Eigen::VectorXi& I, const Eigen::MatrixXi& F, 
                                     Eigen::MatrixXd& A_out)
{
    Eigen::MatrixXi F_closest = F(I, Eigen::all);
    
    Eigen::MatrixXd a1 = A(F_closest.col(0), Eigen::all);
    Eigen::MatrixXd a2 = A(F_closest(Eigen::all, 1), Eigen::all);
    Eigen::MatrixXd a3 = A(F_closest(Eigen::all, 2), Eigen::all);

    Eigen::VectorXd b1 = B(Eigen::all, 0);
    Eigen::VectorXd b2 = B(Eigen::all, 1);
    Eigen::VectorXd b3 = B(Eigen::all, 2);

    a1.array().colwise() *= b1.array();
    a2.array().colwise() *= b2.array();
    a3.array().colwise() *= b3.array();

    A_out = a1 + a2 + a3;
}


/**
 * For each vertex on the target mesh find a match on the source mesh.
 * 
 *  V1: #V1 by 3 source mesh vertices
 *  F1: #F1 by 3 source mesh triangles indices
 *  N1: #V1 by 3 source mesh normals
 *  V2: #V2 by 3 target mesh vertices
 *  F2: #F2 by 3 target mesh triangles indices
 *  N2: #V2 by 3 target mesh normals
 *  W1: #V1 by num_bones source mesh skin weights
 *  dDISTANCE_THRESHOLD_SQRD: distance threshold
 *  dANGLE_THRESHOLD_DEGREES: normal threshold
 *  Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
 *  W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
 */
void find_matches_closest_surface(const Eigen::MatrixXd& V1, const Eigen::MatrixXi& F1, const Eigen::MatrixXd& N1, 
                                  const Eigen::MatrixXd& V2, const Eigen::MatrixXi& F2, const Eigen::MatrixXd& N2, 
                                  const Eigen::MatrixXd& W1, 
                                  double dDISTANCE_THRESHOLD_SQRD, 
                                  double dANGLE_THRESHOLD_DEGREES,
                                  Eigen::MatrixXd& W2,
                                  Eigen::Array<bool,Eigen::Dynamic,1>& Matched)
{
    Matched = Eigen::Array<bool,Eigen::Dynamic,1>::Constant(V2.rows(), false);
    Eigen::VectorXd sqrD; 
    Eigen::VectorXi I;
    Eigen::MatrixXd C, B;
    find_closest_point_on_surface(V2, V1, F1, sqrD, I, C, B);
    
    // for each closest point on the source, interpolate its per-vertex attributes(skin weights and normals) 
    // using the barycentric coordinates
    interpolate_attribute_from_bary(W1, B, I, F1, W2);

    Eigen::MatrixXd N1_match_interpolated;
    interpolate_attribute_from_bary(N1, B, I, F1, N1_match_interpolated);
    
    // check that the closest point passes our distance and normal thresholds
    for (int RowIdx = 0; RowIdx < V2.rows(); ++RowIdx)
    {
        Eigen::VectorXd n1 = N1_match_interpolated.row(RowIdx);
        n1.normalize();

        Eigen::VectorXd n2 = N2.row(RowIdx);
        n2.normalize();

        const double rad_angle = acos(n1.dot(n2));
        const double deg_angle = rad_angle * (180.0 / M_PI);

        if (sqrD(RowIdx) <= dDISTANCE_THRESHOLD_SQRD and deg_angle <= dANGLE_THRESHOLD_DEGREES)
        {
            Matched(RowIdx) = true;
        }
    }
}


/**
 * Inpaint weights for all the vertices on the target mesh for which  we didnt 
 * find a good match on the source (i.e. Matched[i] == False).
 * 
 *  V2: #V2 by 3 target mesh vertices
 *  F2: #F2 by 3 target mesh triangles indices
 *  W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
 *  Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
 *  W_inpainted: #V2 by num_bones, final skinning weights where we inpainted weights for all vertices i where Matched[i] == False
 *  success: true if inpainting succeeded, false otherwise
 */
bool inpaint(const Eigen::MatrixXd& V2, const Eigen::MatrixXi& F2, const Eigen::MatrixXd& W2, const Eigen::Array<bool,Eigen::Dynamic,1>& Matched, Eigen::MatrixXd& W_inpainted)
{
    // Compute the laplacian

    Eigen::SparseMatrix<double> L, M, Minv;
    igl::cotmatrix(V2, F2, L);
    igl::massmatrix(V2, F2, igl::MassMatrixType::MASSMATRIX_TYPE_VORONOI, M);
    igl::invert_diag(M, Minv);

    Eigen::SparseMatrix<double> Q = -L + L*Minv*L;
    
    Eigen::SparseMatrix<double> Aeq;

    Eigen::VectorXd Beq;

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(L.rows(), W2.cols());

    Eigen::VectorXi b_all = Eigen::VectorXi::LinSpaced(V2.rows(), 0, V2.rows()-1);
      
    Eigen::VectorXi b;
    igl::slice_mask(b_all, Matched, 1, b);
    
    Eigen::MatrixXd bc;
    igl::slice_mask(W2, Matched, 1, bc);    
    
    igl::min_quad_with_fixed_data<double> mqwf;
    igl::min_quad_with_fixed_precompute(Q,b,Aeq,true,mqwf);
    
    bool result = igl::min_quad_with_fixed_solve(mqwf,B,bc,Beq,W_inpainted);

    return result;
}


/**
 * Smooth weights in the areas for which weights were inpainted and also their close neighbours.
 * 
 *  V2: #V2 by 3 target mesh vertices
 *  F2: #F2 by 3 target mesh triangles indices
 *  W2: #V2 by num_bones skinning weights
 *  Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
 *  dDISTANCE_THRESHOLD_SQRD: scalar distance threshold
 *  num_smooth_iter_steps: scalar number of smoothing steps
 *  smooth_alpha: scalar the smoothing strength      
 *  W2_smoothed: #V2 by num_bones new smoothed weights
 *  VIDs_to_smooth: 1D array of vertex IDs for which smoothing was applied
 */
void smooth(Eigen::MatrixXd& W2_smoothed,
            Eigen::Array<bool,Eigen::Dynamic,1>& VIDs_to_smooth,
            const Eigen::MatrixXd& V2, 
            const Eigen::MatrixXi& F2, 
            const Eigen::MatrixXd& W2, 
            const Eigen::Array<bool,Eigen::Dynamic,1>& Matched, 
            const double dDISTANCE_THRESHOLD, 
            const double num_smooth_iter_steps=10, 
            const double smooth_alpha=0.2)
{
    Eigen::Array<bool,Eigen::Dynamic,1> NotMatched = Matched == false;
    VIDs_to_smooth = NotMatched; //.array(NotMatched, copy=True)

    std::vector<std::vector<int> > adj_list;
    igl::adjacency_list(F2, adj_list);

    auto get_points_within_distance = [&](const Eigen::MatrixXd& V, const int VID, const double distance)
    {
        // Get all neighbours of vertex VID within dDISTANCE_THRESHOLD   
        std::queue<int> queue;
        queue.push(VID);

        std::set<int> visited;
        visited.insert(VID);
        while (!queue.empty())
        {
            const int vv = queue.front();
            queue.pop();
            
            auto neigh = adj_list[vv];
            for (auto nn : neigh)
            {
                if (!VIDs_to_smooth[nn] && (V.row(VID) - V.row(nn)).norm() < distance)
                {
                    VIDs_to_smooth[nn] = true;
                    if (visited.find(nn) == visited.end())
                    {
                        queue.push(nn);
                        visited.insert(nn);
                    }
                }
            }
        }
    };

    for (int i = 0; i < V2.rows(); ++i)
    {
        if (NotMatched[i])
        {
            get_points_within_distance(V2, i, dDISTANCE_THRESHOLD);
        }
    }

    W2_smoothed = W2;

    for (int step_idx = 0;  step_idx < num_smooth_iter_steps; ++step_idx)
    {
        for (int i = 0; i < V2.rows(); ++i)
        {
            if (VIDs_to_smooth[i])
            {
                auto neigh = adj_list[i];
                int num = neigh.size();
                Eigen::VectorXd weight = W2_smoothed.row(i);

                Eigen::VectorXd new_weight = (1.0-smooth_alpha)*weight;

                for (auto influence_idx : neigh)
                {
                    Eigen::VectorXd weight_connected = W2_smoothed.row(influence_idx);
                    new_weight = new_weight + (smooth_alpha/num) * weight_connected;
                }
                
                W2_smoothed.row(i) = new_weight; 
            }
        }
    }
} 


/**
 *  Perform robust weight transfer from the source mesh to the target mesh as described 
 *  in https://www.dgp.toronto.edu/~rinat/projects/RobustSkinWeightsTransfer/index.html.
 *   
 *  For every vertex on the target mesh, find the closest point on the surface of the source mesh. If that point 
 *  is within the SearchRadius, and their normals differ by less than the NormalThreshold, then we directly copy the 
 *  weights from the source point to the target mesh vertex. For all the vertices we didn't copy the weights directly, 
 *  automatically compute the smooth weights.
 *  
 *  V1: #V1 by 3 source mesh vertices
 *  F1: #F1 by 3 source mesh triangles indices
 *  W1: #V1 by num_bones source mesh skin weights
 *   
 *  V2: #V2 by 3 target mesh vertices
 *  F2: #F2 by 3 target mesh triangles indices
 *  W2: #V2 by num_bones target mesh skin weights that were transferred from the source mesh
 * 
 *  SearchRadius: Radius for searching the closest point.
 *  NormalThreshold: Maximum angle (in degrees) difference between target and source point normals to be considred a match. 
 * 
 *  num_smooth_iter_steps: The number of optional post-processing smoothing iterations applied to the vertices without the match.
 *  smooth_alpha: The strength of each post-processing smoothing iteration.
 */
bool robust_skin_weights_transfer(const Eigen::MatrixXd& V1, 
                                  const Eigen::MatrixXi& F1, 
                                  const Eigen::MatrixXd& W1,
                                  const Eigen::MatrixXd& V2, 
                                  const Eigen::MatrixXi& F2,
                                  Eigen::MatrixXd& W2,
                                  const double SearchRadius,
                                  const double NormalThreshold,
                                  const int num_smooth_iter_steps, 
                                  const double smooth_alpha)
{
    Eigen::MatrixXd N1, N2;
    igl::per_vertex_normals(V1, F1, N1);
    igl::per_vertex_normals(V2, F2, N2);

    const int num_bones = W1.cols(); 

    const double SearchRadiusSqrd = SearchRadius * SearchRadius;

    Eigen::Array<bool,Eigen::Dynamic,1> Matched; 
    Eigen::MatrixXd SkinWeights_interpolated;
    find_matches_closest_surface(V1, F1, N1, V2, F2, N2, W1, SearchRadiusSqrd, NormalThreshold, SkinWeights_interpolated, Matched);

    Eigen::MatrixXd InpaintedWeights;
    const bool success = inpaint(V2, F2, SkinWeights_interpolated, Matched, InpaintedWeights);

    if (success)
    {
        Eigen::MatrixXd SmoothedInpaintedWeights;
        Eigen::Array<bool,Eigen::Dynamic,1> VIDs_to_smooth; 
        smooth(SmoothedInpaintedWeights, VIDs_to_smooth, V2, F2, InpaintedWeights, Matched, SearchRadius, num_smooth_iter_steps, smooth_alpha);
        
        W2 = SmoothedInpaintedWeights;

        return true;
    }
    else 
    {
        return false;
    }
}

/** EXAMPLE RUN */
int main(int argc, char *argv[])
{ 
    // Load source and target meshes
    Eigen::MatrixXd V1, V2;
    Eigen::MatrixXi F1, F2; 
    igl::read_triangle_mesh("../../meshes/sphere.obj", V1, F1);
    igl::read_triangle_mesh("../../meshes/grid.obj", V2, F2);

    // Dummy skinning weights
    Eigen::MatrixXd W1 = Eigen::MatrixXd::Zero(V1.rows(), 2);
    W1.col(0) = 0.3 * Eigen::VectorXd::Ones(V1.rows());
    W1.col(1) = 0.7 * Eigen::VectorXd::Ones(V1.rows());

    const double dDISTANCE_THRESHOLD = 0.05*igl::bounding_box_diagonal(V2); // threshold distance D
    const double dANGLE_THRESHOLD_DEGREES = 30; // threshold angle theta in degrees

    Eigen::MatrixXd InpaintedWeights;
    const bool success = robust_skin_weights_transfer(V1, F1, W1,
                                                      V2, F2, InpaintedWeights,
                                                      dDISTANCE_THRESHOLD,
                                                      dANGLE_THRESHOLD_DEGREES, 
                                                      10, 0.2);

    if (success)
    {
        std::cout << "Computation Succeeded" << std::endl;
        std::cout << InpaintedWeights << std::endl;
    }
    else
    {
        std::cout << "Computation Failed" << std::endl;
    }
}
