#include <vector>
#include <iostream>
#include <random>
#include <deque>
#include <string>
#include <cmath>

#include <CGAL/Epick_d.h>
#include <CGAL/Delaunay_triangulation.h>
#include <CGAL/Cartesian_d.h>
#include <CGAL/Min_sphere_of_spheres_d.h>
#include <CGAL/Min_sphere_of_spheres_d_traits_d.h>
#include <gudhi/Simplex_tree.h>
#include <gudhi/Alpha_complex.h>
#include <Eigen/Core>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/draw_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include "st.h" // custom simplex tree implementation

#define DEBUG false // set to true to print detailed messages on inserted and deleted simplices

#define MOD_INSERT 0
#define MOD_ERASE 1

std::random_device rd;
std::mt19937 gen(42);
double low = -5.0;
double high = 5.0;
std::uniform_real_distribution<double> dist(low, high); // Unif([-3, 3])

const int dim = 3;
typedef CGAL::Epick_d<CGAL::Dynamic_dimension_tag> K;
typedef CGAL::Delaunay_triangulation<K> DT;

typedef DT::Full_cell_handle Full_cell_handle;
typedef DT::Point Point;
typedef DT::Vertex_handle Vertex_handle;
typedef Gudhi::Simplex_tree<> Simplex_tree;

typedef double FT;

using K2 = CGAL::Exact_predicates_inexact_constructions_kernel;
using DT2 = CGAL::Delaunay_triangulation_2<K2>;

typedef CGAL::Cartesian_d<FT> K_cartesian;

// class used to define the smallest enclosing sphere for a simplex
typedef CGAL::Min_sphere_of_spheres_d_traits_d<
    K_cartesian,
    FT,
    dim
> MEB_Traits;

typedef CGAL::Min_sphere_of_spheres_d<MEB_Traits > Min_sphere;
typedef MEB_Traits::Sphere Sphere;
typedef Gudhi::alpha_complex::Alpha_complex<K> Alpha_complex;

using std::vector;
using std::cout;
using std::endl;

/*
g++ delaunay_insert_points.cpp \
    -std=c++17 \
    -I/path/to/gudhi-devel/src \
    -I/usr/include/eigen3 \
    -lgmp -lmpfr

*/

// we need to store and search for simplices efficiently, thus the 
// custom hash function
struct VectorHash {
    std::size_t operator()(const std::vector<int>& v) const {
        std::size_t seed = 0;
        for(int x : v) {
            seed ^= std::hash<int>{}(x)
                + 0x9e3779b9
                + (seed << 6)
                + (seed >> 2);
        }
        return seed;
    }
};

using simplex_map = std::unordered_map<std::vector<int>,int,VectorHash>;
using simplex_set = std::unordered_set<std::vector<int>,VectorHash>;
using simplex_map_double = std::unordered_map<std::vector<int>,double,VectorHash>;


std::vector<double> point_to_vector(const DT::Point& p) {
    std::vector<double> v;
    v.reserve(p.dimension());

    for (int i = 0; i < p.dimension(); ++i)
        v.push_back(CGAL::to_double(p.cartesian(i)));

    return v;
}

// utility function for calculating filtration values of simplices
double meb_radius_squared(const std::vector<std::vector<double>>& pts) {
    std::vector<Sphere> spheres;
    if(pts.size() == 1) return 0.0;
    for(const auto& v : pts) {
        K_cartesian::Point_d p(v.size(),v.begin(),v.end());
        spheres.push_back(Sphere(p, 0.0));
    }
    Min_sphere ms(spheres.begin(),spheres.end());
    return ms.radius() * ms.radius();
}
// generate all sub-simplices of a simplex
void generate_faces(const std::vector<int>& simplex, simplex_set & faces) {
    int n = simplex.size();
    for(int mask = 1; mask < (1 << n); ++mask) {
        std::vector<int> face;
        face.reserve(n);
        for(int i = 0; i < n; ++i) {
            if(mask & (1 << i))
                face.push_back(simplex[i]);
        }
        std::sort(face.begin(), face.end());
        faces.insert(face);
    }
}


class SimulationState {
private:
    int next_vertex_id = 1;
    std::map<Vertex_handle, int> vertex_ids;
    std::map<int, Vertex_handle> id_to_vertex;
    std::unordered_map<int, std::vector<double>> vertex_coords;

public:
    DT dt;
    SimplexTree<int> st;
    std::map<std::vector<int>, int> simplex_count;

    SimulationState(int dim):dt(dim) {}

    Vertex_handle add_triangulation_point(Point pt) {
        Vertex_handle vh = this->dt.insert(pt);
        
        this->vertex_ids[vh] = this->next_vertex_id;
        this->id_to_vertex[this->next_vertex_id] = vh;
        this->vertex_coords[this->next_vertex_id] = point_to_vector(pt);
        ++this->next_vertex_id;
        return vh;
    }
    bool check_state() const {
        bool ok = true;

        if (vertex_ids.size() != id_to_vertex.size()) {
            std::cerr << "vertex_ids.size() = " << vertex_ids.size()
                      << ", id_to_vertex.size() = " << id_to_vertex.size() << '\n';
            ok = false;
        }

        if (vertex_ids.size() != vertex_coords.size()) {
            std::cerr << "vertex_ids.size() = " << vertex_ids.size()
                      << ", vertex_coords.size() = " << vertex_coords.size() << '\n';
            ok = false;
        }

        if (vertex_ids.size() != dt.number_of_vertices()) {
            std::cerr << "vertex_ids.size() = " << vertex_ids.size()
                      << ", dt.number_of_vertices() = " << dt.number_of_vertices() << '\n';
            ok = false;
        }

        // Check vertex_ids -> id_to_vertex
        for (const auto& [vh, id] : vertex_ids) {

            auto it = id_to_vertex.find(id);
            if (it == id_to_vertex.end()) {
                std::cerr << "Missing id_to_vertex entry for id " << id << '\n';
                ok = false;
                continue;
            }

            if (it->second != vh) {
                std::cerr << "Inconsistent mapping for id " << id << '\n';
                ok = false;
            }

            if (!vertex_coords.count(id)) {
                std::cerr << "Missing coordinates for id " << id << '\n';
                ok = false;
            }
        }
        // Check id_to_vertex -> vertex_ids
        for (const auto& [id, vh] : id_to_vertex) {

            auto it = vertex_ids.find(vh);
            if (it == vertex_ids.end()) {
                std::cerr << "Missing vertex_ids entry for id " << id << '\n';
                ok = false;
                continue;
            }

            if (it->second != id) {
                std::cerr << "Reverse mapping broken for id " << id << '\n';
                ok = false;
            }
        }

        return ok;
    }
    void remove_point_from_state(int id) {
        auto vh = id_to_vertex.at(id);
        vertex_ids.erase(vh);
        id_to_vertex.erase(id);
        vertex_coords.erase(id);
    }
    Vertex_handle get_vertex_by_id(int id) const {
        return id_to_vertex.at(id);
    }
    int get_vertex_id(Vertex_handle vh) const {
        if (this->vertex_ids.find(vh) == this->vertex_ids.end()) {
            std::cout << "Missing handle!\n";
        }
        return vertex_ids.at(vh);
    }
    vector<double> get_vertex_coords(Vertex_handle vh) {
        int id = this->vertex_ids.at(vh);
        return vertex_coords.at(id);
    }
    vector<double> get_vertex_coords_by_id(int id) {
        return vertex_coords.at(id);
    }
    void build_cech_delaunay_complex(const vector<Point>& points) {
        this->clear();
        Simplex_tree st_gudhi;
        std::set<std::vector<int>> all_faces;

        for(auto pt : points) {
            this->add_triangulation_point(pt);
        }
        for (auto cell = this->dt.finite_full_cells_begin();
             cell != this->dt.finite_full_cells_end(); ++cell)
        {
            std::vector<int> maximal_simplex;
            maximal_simplex.reserve(this->dt.current_dimension() + 1);

            for (int i = 0; i <= this->dt.current_dimension(); ++i)
                maximal_simplex.push_back(this->get_vertex_id(cell->vertex(i)));

            std::sort(maximal_simplex.begin(), maximal_simplex.end());

            simplex_set local_faces;
            generate_faces(maximal_simplex, local_faces);

            all_faces.insert(local_faces.begin(), local_faces.end());
        }

        for (const auto& simplex : all_faces) {
            std::vector<std::vector<double>> pts;
            pts.reserve(simplex.size());

            for (int id : simplex) {
                const auto& p = this->get_vertex_by_id(id)->point();
                pts.emplace_back(p.cartesian_begin(), p.cartesian_end());
            }
            double filtration = meb_radius_squared(pts);
            st_gudhi.insert_simplex(simplex, filtration);
        }
        st_gudhi.make_filtration_non_decreasing();
        this->st.from_gudhi(st_gudhi);


        for (auto cell = this->dt.finite_full_cells_begin();
         cell != this->dt.finite_full_cells_end(); ++cell) {
            std::vector<int> maximal_simplex;
            maximal_simplex.reserve(this->dt.current_dimension() + 1);
            for (int i = 0; i <= this->dt.current_dimension(); ++i)
                maximal_simplex.push_back(this->get_vertex_id(cell->vertex(i)));

            std::sort(maximal_simplex.begin(), maximal_simplex.end());
            simplex_set faces;
            generate_faces(maximal_simplex, faces);
            for(auto face : faces) {
                ++this->simplex_count[face];
            }
        }
    }

    vector<Vertex_handle> get_vertices_by_id() const {
        vector<std::pair<int, Vertex_handle>> entries;
        entries.reserve(id_to_vertex.size());

        for (const auto& kv : id_to_vertex)
            entries.push_back(kv);

        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });

        vector<Vertex_handle> vertices;
        vertices.reserve(entries.size());

        for (const auto& [id, vh] : entries)
            vertices.push_back(vh);

        return vertices;
    }

    void clear() {
        this->dt.clear();
        this->st.clear();
        this->simplex_count.clear();
        this->vertex_ids.clear();
        this->id_to_vertex.clear();
        this->vertex_coords.clear();
        this->next_vertex_id = 1;
    }
    ~SimulationState() {
        this->clear();
    }
};


// returns true if a full cell contains infinite point i.e. lies on the convex hull
// and false otherwise
bool is_infinite_cell(DT& dt,Full_cell_handle cell) {
    for(int i = 0;i <= dt.current_dimension();++i)
    {
        if(dt.is_infinite(cell->vertex(i)))
        {
            return true;
        }
    }
    return false;
}


// returns chi(t) - the value of Euler Characteristic curve at filtation t
int ECC(const Simplex_tree& st, double t) {
    int chi = 0;
    for(auto sh : st.filtration_simplex_range()) {
        if(st.filtration(sh) > t)
            break;
        int dim = st.dimension(sh);
        chi += (dim % 2 == 0 ? 1 : -1);
    }
    return chi;
}


int simplex_count_of_dimension(Simplex_tree&st, int d) {
    for(auto sh : st.filtration_simplex_range()) {
        int dim = 0;
        for (auto v : st.simplex_vertex_range(sh)) {
            dim++;
        }
    }
    return dim-1;
}

void print_simplex_list(const std::vector<std::vector<int>>&simplex_list) {
    for(auto simplex : simplex_list) {
        for(auto v : simplex) {
            cout << v << " ";
        }
        cout << endl;
    }
}

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

vector<vector<int>> get_vertex_cofaces(const Simplex_tree& st, int vertex_id) {
    std::vector<std::vector<int>> cofaces;

    auto sh = st.find({vertex_id});
    if (sh == st.null_simplex())
        return cofaces;

    for (auto cf : st.cofaces_simplex_range(sh, 0)) {
        std::vector<int> simplex;

        for (auto v : st.simplex_vertex_range(cf))
            simplex.push_back(v);

        std::sort(simplex.begin(), simplex.end());
        cofaces.push_back(std::move(simplex));
    }
    return cofaces;
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

// returns the amount of simplices from Simplex_tree of every dimension up to maximal 
vector<int> simplex_count_by_dim(Simplex_tree&st) {
    int dim_size = st.dimension();
    vector<int> simplex_by_dim(dim_size + 1);
    for (auto sh : st.complex_simplex_range()) {
        int d = 0;
        for (auto v : st.simplex_vertex_range(sh)) {
            ++d;
        }
        simplex_by_dim[d-1]++;
    }
    return simplex_by_dim;
}

// debug util function: return all the simplices of given dimension of Simplex_tree
vector<vector<int>> simplex_list_by_dim(Simplex_tree&st, int dim) {
    vector<vector<int>> simplex_of_dim;
    for (auto sh : st.complex_simplex_range()) {
        int d = 0;
        vector<int> simplex;
        for (auto v : st.simplex_vertex_range(sh)) {
            simplex.push_back(v);
            d++;
        }
        std::sort(simplex.begin(), simplex.end());
        if(d == dim + 1) {
            simplex_of_dim.push_back(simplex);
        }
    }
    return simplex_of_dim;
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

void update_delta_count(const vector<vector<int>>& inserted_list, const vector<vector<int>>& deleted_list, simplex_map& delta_count) {
    for(const auto& simplex: inserted_list) {
        simplex_set local_faces;
        generate_faces(simplex,local_faces);
        for(const auto& f : local_faces) {
            ++delta_count[f];
        }
    }
    for(const auto& simplex: deleted_list) {
        simplex_set local_faces;
        generate_faces(simplex,local_faces);
        for(const auto& f : local_faces) {
            --delta_count[f];
        }
    }

}

// insert a point into Simplicial complex represented by a 
// custom implementation of a Simplex_tree
// Simplex_tree&st argument is deprecated and is not used for other calculations
Vertex_handle insert_point(
    SimulationState&state,
    Point pt) {
        DT::Locate_type lt;
        DT::Face face(state.dt.maximal_dimension());
        DT::Facet facet;

        Full_cell_handle start = state.dt.locate(pt,lt,face,facet);
        std::vector<Full_cell_handle> conflict_cells;
        simplex_map delta_count;
        state.dt.compute_conflict_zone(pt,start,std::back_inserter(conflict_cells));

        std::vector<std::vector<int>> max_deleted_list;
        std::vector<std::vector<int>> max_inserted_list;
        for(const auto& cell : conflict_cells) {
            if(is_infinite_cell(state.dt, cell)) continue;

            std::vector<int> full_cell;
            for(int i = 0; i <= state.dt.current_dimension(); ++i) {
                auto vh = cell->vertex(i);
                if(state.dt.is_infinite(vh))
                    continue;
                full_cell.push_back(state.get_vertex_id(vh));
            }
            std::sort(full_cell.begin(), full_cell.end());
            max_deleted_list.push_back(full_cell);

        }
        if(DEBUG) {
            cout << "Full cells before insertion\n";
            cout << "===============================\n";
            for (auto cell = state.dt.finite_full_cells_begin();
             cell != state.dt.finite_full_cells_end(); ++cell) {
                std::vector<int> maximal_simplex;
                maximal_simplex.reserve(state.dt.current_dimension() + 1);

                for (int i = 0; i <= state.dt.current_dimension(); ++i)
                    maximal_simplex.push_back(state.get_vertex_id(cell->vertex(i)));

                std::sort(maximal_simplex.begin(), maximal_simplex.end());
                print_simplex_list({maximal_simplex});
            }
            cout << "===============================\n";
            cout<<"Conflict full cells during insertion\n";
            cout << "===============================\n";
            print_simplex_list(max_deleted_list);
            cout << "===============================\n";
        }
        Vertex_handle vh = state.add_triangulation_point(pt);
        
        std::vector<Full_cell_handle> incident_cells;
        state.dt.incident_full_cells(vh,std::back_inserter(incident_cells));
        for(const auto& cell : incident_cells) {
            if(is_infinite_cell(state.dt, cell))continue;
            std::vector<int> simplex;
            for(int i = 0; i <= state.dt.current_dimension(); ++i) {
                auto vh = cell->vertex(i);
                if(state.dt.is_infinite(vh))
                continue;
                simplex.push_back(state.get_vertex_id(vh));
            }
            std::sort(simplex.begin(), simplex.end());
            max_inserted_list.push_back(simplex);
        }
        if(DEBUG) {
            cout << "Full cells after insertion\n";
            cout << "===============================\n";
            for (auto cell = state.dt.finite_full_cells_begin();
             cell != state.dt.finite_full_cells_end(); ++cell) {
                std::vector<int> maximal_simplex;
                maximal_simplex.reserve(state.dt.current_dimension() + 1);

                for (int i = 0; i <= state.dt.current_dimension(); ++i)
                    maximal_simplex.push_back(state.get_vertex_id(cell->vertex(i)));

                std::sort(maximal_simplex.begin(), maximal_simplex.end());
                print_simplex_list({maximal_simplex});
            }
            cout << "===============================\n";

            cout<<"New full cells during insertion\n";
            cout << "===============================\n";
            print_simplex_list(max_inserted_list);
            cout << "===============================\n";

            cout<<"Simplex_count by point\n";
            cout << "===============================\n";
            for(auto kv : state.simplex_count) {
                auto simplex = kv.first;
                if(simplex.size() == 1 && kv.second != 0) {
                    cout << simplex[0] << " " << kv.second << "\n";
                }
            }
            cout << "===============================\n";
        }
        update_delta_count(max_inserted_list, max_deleted_list, delta_count);
        std::vector<std::vector<int>> to_delete;
        std::vector<std::vector<int>> to_insert;
        // cout <<"Simplex/delta list\n";
        for(const auto& kv : delta_count) {
            const auto& f = kv.first;
            int delta = kv.second;
            int old_count = state.simplex_count[f];
            int new_count = old_count + delta;
            state.simplex_count[f] = new_count;
            if(old_count > 0 && new_count == 0) {
                to_delete.push_back(f);
            }
            if(old_count == 0 && new_count > 0) to_insert.push_back(f);
        }

        if(DEBUG) {
            cout <<"Removing simplices during insertion :\n";
            print_simplex_list(to_delete);
        }

        simple_remove(to_delete, state.st, MOD_INSERT);
        if(DEBUG) {
            cout <<"Adding simplices:\n";
            print_simplex_list(to_insert);
        }
        bool flag = false;
        for(const auto& f : to_insert) {
            std::vector<std::vector<double>> points;
            for(int v : f) {
                points.push_back(state.get_vertex_coords_by_id(v));
            }
            double filtration = meb_radius_squared(points);
            if(!state.st.contains(f)) {
                state.st.insert_simplex(f, filtration); 
            } else {
                if(filtration < state.st.filtration(f))
                    state.st.assign_filtration(f, filtration);
            }
        }
        state.st.make_filtration_non_decreasing();
        return vh;
}  

// erase a point from a simplicial complex represnted by a
// custom implementation of a Simplex_tree
void erase_point(SimulationState& state, Vertex_handle& vh_to_remove) {
    
    std::vector<Full_cell_handle> old_incident_cells;
    state.dt.incident_full_cells(vh_to_remove,std::back_inserter(old_incident_cells));

    std::unordered_set<Vertex_handle> neighbor_vertices;
    for(const auto& cell :old_incident_cells) {
        if(is_infinite_cell(state.dt, cell)) continue;
        std::vector<int> simplex;
        for(int i = 0;i <= state.dt.current_dimension();++i) {
            auto vh = cell->vertex(i);
            if(vh != vh_to_remove) neighbor_vertices.insert(vh);
            simplex.push_back(state.get_vertex_id(vh));
        }
        std::sort(simplex.begin(), simplex.end());
        std::unordered_set<std::vector<int>,VectorHash> local_faces;
        generate_faces(simplex,local_faces);
    }

    std::unordered_set<int> cavity_vertex_ids;
    for(const auto& vh : neighbor_vertices) {
        cavity_vertex_ids.insert(state.get_vertex_id(vh));
    }

    std::unordered_map<std::vector<int>,int,VectorHash> boundary_count;

    for(const auto& cell : old_incident_cells) {
        if(is_infinite_cell(state.dt, cell))
            continue;

        std::vector<int> facet;

        for(int i = 0; i <= state.dt.current_dimension(); ++i) {
            auto vh = cell->vertex(i);
            if(vh == vh_to_remove)
                continue;
            facet.push_back(state.get_vertex_id(vh));
        }

        std::sort(facet.begin(),facet.end());
        ++boundary_count[facet];
    }

    std::unordered_set<std::vector<int>,VectorHash> cavity_boundary;

    for(const auto& kv : boundary_count) {
        if(kv.second == 1)
            cavity_boundary.insert(kv.first);
    }
    if(DEBUG) {
        cout << "Full cells before removal \n";
        cout << "===============================\n";
        for (auto cell = state.dt.finite_full_cells_begin();
         cell != state.dt.finite_full_cells_end(); ++cell) {
            std::vector<int> maximal_simplex;
            maximal_simplex.reserve(state.dt.current_dimension() + 1);
            for (int i = 0; i <= state.dt.current_dimension(); ++i)
                maximal_simplex.push_back(state.get_vertex_id(cell->vertex(i)));
            std::sort(maximal_simplex.begin(), maximal_simplex.end());
            print_simplex_list({maximal_simplex});
        }
        cout << "===============================\n";
    }
    const int removed_id = state.get_vertex_id(vh_to_remove);
    state.dt.remove(vh_to_remove);
    auto v = std::vector<int> {removed_id};
    std::vector<std::vector<int>> simplices_to_remove = state.st.cofaces(v);

    if(DEBUG) {
        cout<<"Simplex_count by point\n";
        cout << "===============================\n";
        for(auto kv : state.simplex_count) {
            auto simplex = kv.first;
            if(simplex.size() == 1 && kv.second != 0) {
                cout << simplex[0] << " " << kv.second << "\n";
            }
        }
        cout << "===============================\n";
    }
    //simple_remove(simplices_to_remove, st_custom, MOD_ERASE);
    // remove_simplex_list(simplices_to_remove, st_custom, MOD_ERASE);
    vector<vector<int>> max_deleted_list;
    for(auto simplex : simplices_to_remove) {
        if(simplex.size() == dim + 1) 
            max_deleted_list.push_back(simplex);
    }

    std::unordered_set<Full_cell_handle> new_cells;

    for(const auto& vh : neighbor_vertices) {
        std::vector<Full_cell_handle> local_cells;
        state.dt.incident_full_cells(vh,std::back_inserter(local_cells));
        for(const auto& cell : local_cells) {
            if(is_infinite_cell(state.dt, cell))
                continue;
            bool inside_cavity = true;
            for(int i = 0;i <= state.dt.current_dimension();++i) {
                auto cvh = cell->vertex(i);
                int id = state.get_vertex_id(cvh);

                if(!cavity_vertex_ids.count(id)) {
                    inside_cavity = false;
                    break;
                }
            }
            if(!inside_cavity) continue;
            new_cells.insert(cell);
        }
    }
    std::unordered_set<std::vector<int>,VectorHash> new_simplices;
    vector<vector<int>> max_inserted_list;
    for(const auto& cell : new_cells) {
        if(is_infinite_cell(state.dt, cell))continue;
        std::vector<int> simplex;

        for(int i = 0;i <= state.dt.current_dimension();++i) {
            auto vh = cell->vertex(i);
            if(state.dt.is_infinite(vh))
                continue;
            simplex.push_back(state.get_vertex_id(vh));
        }
        std::sort(simplex.begin(), simplex.end());
      
        if(state.simplex_count[simplex] > 0) {
            // cout <<"An existing cell has been preserved\n";
            continue;
        }
        
        std::unordered_set<std::vector<int>,VectorHash> local_faces;
        max_inserted_list.push_back(simplex);
        generate_faces(simplex,local_faces);
        for(auto face : local_faces) {
            auto new_face = face;
            std::sort(new_face.begin(), new_face.end());
            new_simplices.insert(new_face);
        }
    }
    if(DEBUG) {
        cout << "Full cells after removal \n";
        cout << "===============================\n";
        for (auto cell = state.dt.finite_full_cells_begin();
         cell != state.dt.finite_full_cells_end(); ++cell) {
            std::vector<int> maximal_simplex;
            maximal_simplex.reserve(state.dt.current_dimension() + 1);
            for (int i = 0; i <= state.dt.current_dimension(); ++i)
                maximal_simplex.push_back(state.get_vertex_id(cell->vertex(i)));
            std::sort(maximal_simplex.begin(), maximal_simplex.end());
            print_simplex_list({maximal_simplex});
        }
        cout << "===============================\n";
    }
    std::unordered_map<std::vector<int>,int,VectorHash> old_global_count;
    std::vector<std::vector<int>> simplices_to_insert(new_simplices.begin(), new_simplices.end());
    simplex_map delta_count;
    update_delta_count(max_inserted_list, max_deleted_list, delta_count);
    std::vector<std::vector<int>> to_delete;
    std::vector<std::vector<int>> to_insert;
    // cout <<"Simplex/delta list\n";
        for(const auto& kv : delta_count) {
            const auto& f = kv.first;
            int delta = kv.second;
            int old_count = state.simplex_count[f];
            int new_count = old_count + delta;
            if(new_count > 1 && f.size() == dim + 1) {
                cout << "BAD FULL CELL IN simplex_count during erase \n";
                new_count = 1;
            }
            state.simplex_count[f] = new_count;
            if(old_count > 0 && new_count <= 0) to_delete.push_back(f);
            if(old_count == 0 && new_count > 0) to_insert.push_back(f);
    }

    //simple_remove(to_delete, st_custom, MOD_ERASE);
    state.st.removeSimplexAndCofaces(v);

    for(auto simplex : to_insert) {
        std::vector<std::vector<double>> points;
        for(int v : simplex) {points.push_back(state.get_vertex_coords_by_id(v));}
        bool contains_simplex = state.st.contains(simplex); 
        for(int v : simplex) {
            points.push_back(state.get_vertex_coords_by_id(v));
        }
        double filtration = meb_radius_squared(points);
        std::sort(simplex.begin(), simplex.end());
        if(!contains_simplex) {
            state.st.insert_simplex(simplex, filtration);
        } else {
            state.st.assign_filtration(simplex, filtration);
        }
    }
    
    state.st.make_filtration_non_decreasing();
    state.remove_point_from_state(removed_id);
}

void build_cech_delaunay_complex(SimulationState&state) {
    Simplex_tree st;
    std::set<std::vector<int>> all_faces;

    for (auto cell = state.dt.finite_full_cells_begin();
         cell != state.dt.finite_full_cells_end(); ++cell)
    {
        std::vector<int> maximal_simplex;
        maximal_simplex.reserve(state.dt.current_dimension() + 1);

        for (int i = 0; i <= state.dt.current_dimension(); ++i)
            maximal_simplex.push_back(state.get_vertex_id(cell->vertex(i)));

        std::sort(maximal_simplex.begin(), maximal_simplex.end());

        simplex_set local_faces;
        generate_faces(maximal_simplex, local_faces);

        all_faces.insert(local_faces.begin(), local_faces.end());
    }

    for (const auto& simplex : all_faces) {
        std::vector<std::vector<double>> pts;
        pts.reserve(simplex.size());

        for (int id : simplex) {
            const auto& p = state.get_vertex_by_id(id)->point();
            pts.emplace_back(p.cartesian_begin(), p.cartesian_end());
        }
        double filtration = meb_radius_squared(pts);
        st.insert_simplex(simplex, filtration);
    }

    st.make_filtration_non_decreasing();
    state.st.from_gudhi(st);
}

void copy_delaunay(const DT& src, DT& dst) {
    dst.clear();

    for (auto vh = src.finite_vertices_begin();
         vh != src.finite_vertices_end(); ++vh)
    {
        dst.insert(vh->point());
    }
}

Simplex_tree build_cech_delaunay_complex_tree(const SimulationState&state) {
    Simplex_tree st;
    std::set<std::vector<int>> all_faces;
    
    for (auto cell = state.dt.finite_full_cells_begin();
         cell != state.dt.finite_full_cells_end(); ++cell)
    {
        std::vector<int> maximal_simplex;
        maximal_simplex.reserve(state.dt.current_dimension() + 1);

        for (int i = 0; i <= state.dt.current_dimension(); ++i)
            maximal_simplex.push_back(state.get_vertex_id(cell->vertex(i)));

        std::sort(maximal_simplex.begin(), maximal_simplex.end());

        simplex_set local_faces;
        generate_faces(maximal_simplex, local_faces);

        all_faces.insert(local_faces.begin(), local_faces.end());
    }

    for (const auto& simplex : all_faces) {
        std::vector<std::vector<double>> pts;
        pts.reserve(simplex.size());

        for (int id : simplex) {
            const auto& p = state.get_vertex_by_id(id)->point();
            pts.emplace_back(p.cartesian_begin(), p.cartesian_end());
        }
        double filtration = meb_radius_squared(pts);
        st.insert_simplex(simplex, filtration);
    }

    st.make_filtration_non_decreasing();
    return st;
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
    //int window_size = 500;
    int n_points_after = 100;
    vector<int> point_counts = {100, 250};
    vector<int> window_sizes = {50, 100, 150, 200, 250, 300, 400, 500};
        cout << "OK\n";
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
    cout<<"Ok before build complex\n";
    state.build_cech_delaunay_complex(window_cgal);
    auto vertex_handles = state.get_vertices_by_id();
    
    std::deque<Vertex_handle> window_pts(vertex_handles.begin(), vertex_handles.end());
    
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < n_points_after; ++i) {
        bool ok = state.check_state();
        if(ok) {
            cout <<"State ok\n";
        } else {
            cout << "State not ok\n";
        }
        cout << i << endl;
        Point pt = points_after[i];
        Vertex_handle vh = insert_point(state, pt);
        window_pts.push_back(vh);    
        // Vertex_handle vh = state.dt.insert(pt);
        // state.vertex_ids[vh] = state.next_vertex_id;
        // state.id_to_vertex[state.next_vertex_id] = vh;
        // state.vertex_coords[state.next_vertex_id] = point_to_vector(pt);
        // state.next_vertex_id++;
        
        if(window_pts.size() > window_size) {
            Vertex_handle vh = window_pts.front();
            window_pts.pop_front();
            erase_point(state, vh);
            // state.dt.remove(vh);
            // int removed_id = state.vertex_ids[vh];
            // state.vertex_ids.erase(vh);
            // state.id_to_vertex.erase(removed_id);
            // state.vertex_coords.erase(removed_id);
        }
        Simplex_tree st_recomputed = build_cech_delaunay_complex_tree(state);
        double correct_ratio = check_ECC(state.st, st_recomputed, 5);
        if(correct_ratio != 1.00) {
            cout <<"ALARM\n";
        }
        cout << "ECC is correct on " << correct_ratio * 100.0 <<" % values\n";
        bool mismatch = false;
        auto simplex_dim = simplex_count_by_dim(st_recomputed);
        for(int d = 0; d <= dim;d++) {
            int a = state.st.num_simplices(d);
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

    state.clear();
    }
    }

    return 0;
}