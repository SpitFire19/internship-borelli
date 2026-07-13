#include "simulation.h"

int main() {
    vector<int> point_counts = {100, 250};
    vector<int> window_sizes = {50, 100, 150, 200, 250, 300, 400, 500};
    std::cerr <<"Benchmarking recalculating window for every new point\n";
    for(int n_points_after : point_counts) {
        for(int window_size : window_sizes) {
            SimulationState state(dim);
            vector<vector<double>> window;
            int id_start = 1;
            for(int i = 0; i < window_size; ++i) {
                vector<double> pt = rand_point_dD(dim);
                window.push_back(pt);
            }
            vector<Point> points_after;
            for(int i = 0; i < n_points_after; ++i) {
                vector<double> pt = rand_point_dD(dim);
                Point p = Point(dim, pt.begin(), pt.begin() + dim);
                points_after.push_back(p);
            }
            std::vector<Point> window_cgal;
            for(auto pt : window) {
                Point p = Point(dim, pt.begin(), pt.begin() + dim);
                window_cgal.push_back(p);
            }
            state.build_cech_delaunay_complex(window_cgal);
            auto vertex_handles = state.get_vertices_by_id();

            std::deque<Vertex_handle> window_pts(vertex_handles.begin(), vertex_handles.end());

            auto start = std::chrono::high_resolution_clock::now();
            for(int i = 0; i < n_points_after; ++i) {
                Point pt = points_after[i];
                Vertex_handle vh = state.add_triangulation_point(pt);
                window_pts.push_back(vh);    
                
                if(window_pts.size() > window_size) {
                    Vertex_handle vh = window_pts.front();
                    window_pts.pop_front();
                    int removed_point_id = state.get_vertex_id(vh);
                    state.dt.remove(vh);
                    state.remove_point_from_state(removed_point_id);
                }
                Simplex_tree st_recomputed = state.build_reference_complex();
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cerr << "Time for k=" << n_points_after <<" "<<", h="<<window_size << " : " << elapsed.count() << " seconds\n";
            state.clear();
        }
    }
    return 0;
}