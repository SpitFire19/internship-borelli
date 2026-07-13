#include "geometry_utils.h"


#define DEBUG false // set to true to print detailed messages on inserted and deleted simplices

#define MOD_INSERT 0
#define MOD_ERASE 1

std::random_device rd;
std::mt19937 gen(42);
double low = -5.0;
double high = 5.0;
std::uniform_real_distribution<double> dist(low, high); // Unif([-3, 3])



/*
g++ delaunay_insert_points.cpp \
    -std=c++17 \
    -I/path/to/gudhi-devel/src \
    -I/usr/include/eigen3 \
    -lgmp -lmpfr

*/

std::map<vector<int>, int> simplex_count;

std::map<DT::Vertex_handle,int> vertex_ids; // map handle to id (a number) 
std::map<int,DT::Vertex_handle> id_to_vertex; // reverse map to facilitate access operations
std::unordered_map<int,std::vector<double>> vertex_coords; // map vetex id to its coordinates

int next_vertex_id = 1; // used to provide a unique id to each new vertex

void remove_simplex_list(std::vector<std::vector<int>> simplices_to_remove, Simplex_tree &st) {
    std::sort(simplices_to_remove.begin(),simplices_to_remove.end(),
        [](const auto& a, const auto& b){
            return a.size() > b.size(); 
        });
    bool progress = true;
    while(progress) {
        progress = false;

        for(auto it = simplices_to_remove.begin(); it != simplices_to_remove.end();) {
            auto sh = st.find(*it);

            if(sh == st.null_simplex()) {
                it = simplices_to_remove.erase(it);
                continue;
            }
            if(!st.has_children(sh)) {
                st.remove_maximal_simplex(sh);
                it = simplices_to_remove.erase(it);
                progress = true;
            }
            else {
                ++it;
            }   
        }
    }
    if(!simplices_to_remove.empty()) {
        std::cerr << "Gudhi st, Failed to remove "
              << simplices_to_remove.size()
              << " simplices\n";
        std::cerr <<"====================\n";
        print_simplex_list(simplices_to_remove);
        std::cerr <<"====================\n";
    }
}

void remove_simplex_list(std::vector<std::vector<int>> simplices_to_remove, SimplexTree<int>& st_custom, int mode) {
    std::sort(simplices_to_remove.begin(),simplices_to_remove.end(),
        [](const auto& a, const auto& b){
            return a.size() > b.size(); 
        });
    bool progress = true;
    while(progress) {
        progress = false;

        for(auto it = simplices_to_remove.begin(); it != simplices_to_remove.end();) {
            bool sh = st_custom.contains(*it);

            if(sh) {
                it = simplices_to_remove.erase(it);
                continue;
            }
            if(!st_custom.has_children(*it)) {
                st_custom.remove_maximal_simplex(*it);
                it = simplices_to_remove.erase(it);
                progress = true;
            }
            else {
                ++it;
            }   
        }
    }
    
    if(!simplices_to_remove.empty()) {
        std::cerr << "Failed to remove "
              << simplices_to_remove.size()
              << " simplices during";
              if(mode == MOD_INSERT) {
                cout <<"insert\n";
              } 
              if(mode == MOD_ERASE) {
                cout << "erase\n";
              }
    }
}


void simple_remove(std::vector<std::vector<int>> simplices_to_remove, SimplexTree<int>&st_custom, int mode) {
    std::sort(simplices_to_remove.begin(),simplices_to_remove.end(),
        [](const auto& a, const auto& b){
            return a.size() > b.size(); 
        });
        for(auto simplex : simplices_to_remove) {
            st_custom.remove_maximal_simplex(simplex);
        }
        for(auto simplex: simplices_to_remove) {
            int delete_fail = 0;
            if(st_custom.contains(simplex)) {
                if(mode == MOD_INSERT) {
                    cout <<"DELETE FAIL during insert" << endl;
                } 
                if(mode == MOD_ERASE) {
                    cout <<"DELETE FAIL during erase" << endl;
                }
                std::cout<<"====================\n";
                vector<vector<int>> vc = {simplex};
                print_simplex_list({simplex});
                std::cout<<"====================\n";
                delete_fail++;
            }
            if(delete_fail !=0)
                cout <<"TOTAL: " << delete_fail << endl;
        }
}


// debug util function: compare two simplex trees and provide an 
// analysis of simplex count mismatches
// launch only for small amount of simplices
void compare_deltas(SimplexTree<int> &st_custom, Simplex_tree&st) {
    auto sd_true = simplex_count_by_dim(st);
    for(int d = 0; d <= dim; ++d) {
        int sd_pred =  st_custom.num_simplices(d);
        int delta = sd_pred - sd_true[d];
        cout << '\n' << delta;
        if(true) {
            if(delta != 0) cout << "\nDIFF found\n";
            cout <<"\nTrue Simplex tree simplices\n";
            auto simplex_list = simplex_list_by_dim(st, d);
            print_simplex_list(simplex_list);
            cout <<"Predicted simplex tree simplices\n";
            st_custom.print_simplices_of_dimension(d);
        }
    }
    cout << endl;
}


double check_ECC(SimplexTree<int>&st_custom, Simplex_tree&st_recalc, double tmax) {
    int n_values = 100;
    int vals_correct = 0;

    for(int i = 0; i < n_values; ++i) {
        double t = tmax * i / n_values;
        int predicted =  st_custom.ECC(t);
        int recalculated = ECC(st_recalc, t);
        if(predicted == recalculated) {
            vals_correct++;
        }
    }

    return (vals_correct * 1.0) / n_values;
}
// With no recalc:
// k = 10000
// h=50: 69.3s
// h=500: 539.7s

// k = 1000
//h=50: 7.13s
//h=500: 52.717s

// With recalc:
// k = 1000
//h=50: 17.6s
//h=500: 228.1s

vector<double> rand_point_dD(int dim) {
    vector<double> point;
    point.reserve(dim);
    for(int i = 0; i < dim; ++i) {
        double coord = dist(gen);
        point.push_back(coord);
    }
    return point;
}

int main() {
    DT dt(dim);
    int window_size = 100;
    int n_points_after = 1000;
    
    vector<vector<double>> window;
    int id_start = next_vertex_id; 
    for(int i = 0; i < window_size; ++i) {
        vector<double> pt = rand_point_dD(dim);
        vertex_coords[id_start] = pt;
        id_start++;
        window.push_back(pt);
    }
    vector<Point> points_after;
    for(int i = 0; i < n_points_after; ++i) {
        vector<double> pt = rand_point_dD(dim);
        vertex_coords[id_start] = pt;
        id_start++;
        Point p = Point(dim, pt.begin(), pt.begin() + dim);
        points_after.push_back(p);
    }
    std::deque<Vertex_handle> window_pts;
    for(auto pt : window) {
        Point p = Point(dim, pt.begin(), pt.begin() + dim);
        Vertex_handle v = dt.insert(p);
        window_pts.push_back(v);
        vertex_ids[v] = next_vertex_id;
        id_to_vertex[next_vertex_id] = v;
        next_vertex_id++;
    }
    
    auto st = build_cech_delaunay_complex(dt);
    SimplexTree<int> st_custom;
    st_custom.from_gudhi(st);
    simplex_count.clear();
    for (auto cell = dt.finite_full_cells_begin();
         cell != dt.finite_full_cells_end(); ++cell) {
            std::vector<int> maximal_simplex;
            maximal_simplex.reserve(dt.current_dimension() + 1);
            for (int i = 0; i <= dt.current_dimension(); ++i)
                maximal_simplex.push_back(vertex_ids.at(cell->vertex(i)));

            std::sort(maximal_simplex.begin(), maximal_simplex.end());
            simplex_set faces;
            generate_faces(maximal_simplex, faces);
            for(auto face : faces) {
                ++simplex_count[face];
            }
    }
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < n_points_after; ++i) {
        cout << "Iter "<< i << endl;
        Point pt = points_after[i];
        Vertex_handle vh = insert_point(dt, st_custom, pt, vertex_ids, id_to_vertex, vertex_coords);
        if(window_pts.size() > window_size) {
            Vertex_handle vh = window_pts.front();
            window_pts.pop_front();
            erase_point(dt, st_custom, vh, vertex_ids, id_to_vertex, vertex_coords);
        }
        Simplex_tree st_recomputed = build_cech_delaunay_complex(dt);
        double correct_ratio = check_ECC(st_custom, st_recomputed, 5);
        if(correct_ratio != 1.00) {
            cout <<"ALARM\n";
        }
        cout << "ECC is correct on " << correct_ratio * 100.0 <<" % values\n";
        bool mismatch = false;
        auto simplex_dim = simplex_count_by_dim(st_recomputed);
        for(int d = 0; d <= dim;d++) {
            int a = st_custom.num_simplices(d);
            if(a != simplex_dim[d]) {
                mismatch = true;
            }
        }
        if(!mismatch) {
            cout << "Simplex count by dim OK\n";
        } else {
            cout << "Simplex count by dim NOT OK\n";
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cerr << "Time for k=" << n_points_after <<" "<<", h="<<window_size << " : " << elapsed.count() << " seconds\n";
    return 0;
}