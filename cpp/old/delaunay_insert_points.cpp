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
double low = -3.0;
double high = 3.0;
std::uniform_real_distribution<double> dist(low, high); // Unif([-3, 3])

int dim = 2;
typedef CGAL::Epick_d<CGAL::Dynamic_dimension_tag> K;
typedef CGAL::Delaunay_triangulation<K> DT;

typedef DT::Full_cell_handle Full_cell_handle;
typedef DT::Point Point;
typedef DT::Vertex_handle Vertex_handle;
typedef Gudhi::Simplex_tree<> Simplex_tree;

int const d = 3;
typedef double FT;

using K2 = CGAL::Exact_predicates_inexact_constructions_kernel;
using DT2 = CGAL::Delaunay_triangulation_2<K2>;

typedef CGAL::Cartesian_d<FT> K_cartesian;

// class used to define the smallest enclosing sphere for a simplex
typedef CGAL::Min_sphere_of_spheres_d_traits_d<
    K_cartesian,
    FT,
    2
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

std::map<vector<int>, int> simplex_count;

std::map<DT::Vertex_handle,int> vertex_ids; // map handle to id (a number) 
std::map<int,DT::Vertex_handle> id_to_vertex; // reverse map to facilitate access operations
std::unordered_map<int,std::vector<double>> vertex_coords; // map vetex id to its coordinates

int next_vertex_id = 4; // used to provide a unique id to each new vertex


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

void print_st(Simplex_tree st) {
        for(auto sh : st.filtration_simplex_range()) {

        std::cout << "{ ";

        for(auto v : st.simplex_vertex_range(sh))
            std::cout << v << " ";

        std::cout << "} filtration = "
                  << st.filtration(sh)
                  << std::endl;
    }

}

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

void print_simplex_tree(const Simplex_tree& st) {
    for (auto sh : st.complex_simplex_range()) {
        std::cout << "{";
        bool first = true;
        for (auto v : st.simplex_vertex_range(sh)) {
            if (!first) std::cout << ",";
            std::cout << v;
            first = false;
        }

        std::cout << "} ";
        std::cout << st.filtration(sh) << "\n";
    }
}

vector<std::pair<vector<int>, double>> simplex_tree_to_simplex_list(const Simplex_tree& st) {
    vector<std::pair<vector<int>, double>> simplex_list;
    for (auto sh : st.complex_simplex_range()) {
        vector<int> simplex;
        for (auto v : st.simplex_vertex_range(sh)) {
            simplex.push_back(v);
        }
        std::sort(simplex.begin(), simplex.end());
        simplex_list.push_back(std::make_pair(simplex, st.filtration(sh)));
    }
    return simplex_list;
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
        // cout <<"Simplex/delta list before actual update\n";
        // print_simplex_list(inserted_list);
        // cout<<"======================\n";
        // print_simplex_list(deleted_list);
        // cout<<"======================\n";
    for(const auto& simplex: inserted_list) {
        simplex_set local_faces;
        generate_faces(simplex,local_faces);
        for(const auto& f : local_faces) {
            ++delta_count[f];
            // for(auto v : f) {
            //     cout << v << " ";
            // }
            // cout <<delta_count[f]<< endl;
        }
    }
    for(const auto& simplex: deleted_list) {
        simplex_set local_faces;
        generate_faces(simplex,local_faces);
        for(const auto& f : local_faces) {
            --delta_count[f];
            // for(auto v : f) {
            //     cout << v << " ";
            // }
            // cout <<delta_count[f]<< endl;
        }
    }

}

// insert a point into Simplicial complex represented by a 
// custom implementation of a Simplex_tree
// Simplex_tree&st argument is deprecated and is not used for other calculations
Vertex_handle insert_point(DT& dt,
    SimplexTree<int>& st_custom, 
    Point pt,
    std::map<DT::Vertex_handle,int> &vertex_ids,
    std::map<int,DT::Vertex_handle> &id_to_vertex,
    std::unordered_map<int,std::vector<double>> &vertex_coords) {
        DT::Locate_type lt;
        DT::Face face(dt.maximal_dimension());
        DT::Facet facet;

        Full_cell_handle start = dt.locate(pt,lt,face,facet);
        std::vector<Full_cell_handle> conflict_cells;
        simplex_map delta_count;
        dt.compute_conflict_zone(pt,start,std::back_inserter(conflict_cells));

        std::vector<std::vector<int>> max_deleted_list;
        std::vector<std::vector<int>> max_inserted_list;
        for(const auto& cell : conflict_cells) {
            if(is_infinite_cell(dt, cell)) continue;

            std::vector<int> full_cell;
            for(int i = 0; i <= dt.current_dimension(); ++i) {
                auto vh = cell->vertex(i);
                if(dt.is_infinite(vh))
                    continue;
                full_cell.push_back(vertex_ids[vh]);
            }
            std::sort(full_cell.begin(), full_cell.end());
            max_deleted_list.push_back(full_cell);

        }
        // cout<<"Conflict cells during insertion\n";
        // cout << "===============================\n";
        // print_simplex_list(max_deleted_list);
        // cout << "===============================\n";

        Vertex_handle vh = dt.insert(pt);
        vertex_ids[vh] = next_vertex_id;
        id_to_vertex[next_vertex_id] = vh;
        ++next_vertex_id;
        
        std::vector<Full_cell_handle> incident_cells;
        dt.incident_full_cells(vh,std::back_inserter(incident_cells));
        for(const auto& cell : incident_cells) {
            if(is_infinite_cell(dt, cell))continue;
            std::vector<int> simplex;
            for(int i = 0; i <= dt.current_dimension(); ++i) {
                auto vh = cell->vertex(i);
                if(dt.is_infinite(vh))
                continue;
                simplex.push_back(vertex_ids[vh]);
            }
            std::sort(simplex.begin(), simplex.end());
            max_inserted_list.push_back(simplex);
        }
        update_delta_count(max_inserted_list, max_deleted_list, delta_count);
        std::vector<std::vector<int>> to_delete;
        std::vector<std::vector<int>> to_insert;
        // cout <<"Simplex/delta list\n";
        for(const auto& kv : delta_count) {
            const auto& f = kv.first;
            int delta = kv.second;
            int old_count;
            if(simplex_count.find(f) != simplex_count.end()) {
                old_count = simplex_count[f];
            } else {
                // cout<<"NOT IN SIMPLEX COUNT\n";
                old_count = 0;
            }
            int new_count = old_count + delta;
            simplex_count[f] = new_count;
            // for(auto v : f) {
            //     cout << v << " ";
            // }
            // cout << old_count <<"+" << delta << "="<< new_count << endl;
            if(old_count > 0 && new_count <= 0) to_delete.push_back(f);
            if(old_count == 0 && new_count > 0) to_insert.push_back(f);
        }

        if(DEBUG) {
            cout <<"Removing simplices during insertion :\n";
            print_simplex_list(to_delete);
        }

        simple_remove(to_delete, st_custom, MOD_INSERT);
        if(DEBUG) {
            cout <<"Adding simplices:\n";
            print_simplex_list(to_insert);
        }
        bool flag = false;
        for(const auto& f : to_insert) {
            std::vector<std::vector<double>> points;
            for(int v : f) {
                points.push_back(vertex_coords[v]);
            }
            double filtration = meb_radius_squared(points);
            if(!st_custom.contains(f)) {
                st_custom.insert_simplex(f, filtration); 
            } else {
                if(filtration < st_custom.filtration(f))
                    st_custom.assign_filtration(f, filtration);
            }
        }
        st_custom.make_filtration_non_decreasing();
        return vh;
}  

// erase a point from a simplicial complex represnted by a
// custom implementation of a Simplex_tree
void erase_point(
    DT& dt,
    SimplexTree<int>& st_custom,
    Vertex_handle& vh_to_remove,
    std::map<DT::Vertex_handle,int>& vertex_ids,
    std::map<int,DT::Vertex_handle>& id_to_vertex,
    std::unordered_map<int,std::vector<double>>& vertex_coords) {
    
    std::vector<Full_cell_handle> old_incident_cells;
    dt.incident_full_cells(vh_to_remove,std::back_inserter(old_incident_cells));

    std::unordered_set<Vertex_handle> neighbor_vertices;
    for(const auto& cell :old_incident_cells) {
        std::vector<int> simplex;
        for(int i = 0;i <= dt.current_dimension();++i) {
            auto vh = cell->vertex(i);
            if(vh != vh_to_remove) neighbor_vertices.insert(vh);
            simplex.push_back(vertex_ids[vh]);
        }
        std::sort(simplex.begin(), simplex.end());
        std::unordered_set<std::vector<int>,VectorHash> local_faces;
        generate_faces(simplex,local_faces);
    }

    std::unordered_set<int> cavity_vertex_ids;
    for(const auto& vh : neighbor_vertices) {
        cavity_vertex_ids.insert(vertex_ids[vh]);
    }

    std::unordered_map<std::vector<int>,int,VectorHash> boundary_count;

    for(const auto& cell : old_incident_cells) {
        if(is_infinite_cell(dt, cell))
            continue;

        std::vector<int> facet;

        for(int i = 0; i <= dt.current_dimension(); ++i) {
            auto vh = cell->vertex(i);
            if(vh == vh_to_remove)
                continue;
            facet.push_back(vertex_ids[vh]);
        }

        std::sort(facet.begin(),facet.end());
        ++boundary_count[facet];
    }

    std::unordered_set<std::vector<int>,VectorHash> cavity_boundary;

    for(const auto& kv : boundary_count) {
        if(kv.second == 1)
            cavity_boundary.insert(kv.first);
    }

    int removed_id = vertex_ids[vh_to_remove];
    dt.remove(vh_to_remove);
    auto v = std::vector<int> {vertex_ids[vh_to_remove]};
    std::vector<std::vector<int>> simplices_to_remove = st_custom.cofaces(v);

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
        dt.incident_full_cells(vh,std::back_inserter(local_cells));
        for(const auto& cell : local_cells) {
            if(is_infinite_cell(dt, cell))
                continue;
            bool inside_cavity = true;
            for(int i = 0;i <= dt.current_dimension();++i) {
                auto cvh = cell->vertex(i);
                int id = vertex_ids[cvh];

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
        if(is_infinite_cell(dt, cell))continue;
        std::vector<int> simplex;

        for(int i = 0;i <= dt.current_dimension();++i) {
            auto vh = cell->vertex(i);
            if(dt.is_infinite(vh))
                continue;
            simplex.push_back(vertex_ids[vh]);
        }
        std::sort(simplex.begin(), simplex.end());
        if(simplex_count[simplex] > 0) {
            //cout <<"An existing cell has been preserved\n";
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
            int old_count = simplex_count[f];
            int new_count = old_count + delta;
            if(new_count > 1 && f.size() == 3) {
                cout << "BAD FULL CELL IN simplex_count during erase \n";
                new_count = 1;
            }
            simplex_count[f] = new_count;
            if(old_count > 0 && new_count <= 0) to_delete.push_back(f);
            if(old_count == 0 && new_count > 0) to_insert.push_back(f);
    }

    if(DEBUG) {
        auto to_remove = st_custom.cofaces(v);
        cout <<"REMOVING, simplices " << to_remove.size() << "\n=================\n";
        print_simplex_list(to_remove);
        cout <<"=================\n";
    }
    //simple_remove(to_delete, st_custom, MOD_ERASE);
    st_custom.removeSimplexAndCofaces(v);
    if(DEBUG) {
        cout << "Adding simplexes ("<<to_insert.size()<<")" << endl;
        cout <<"=================\n";
        print_simplex_list(to_insert);
        cout <<"=================\n";
    }
    for(auto simplex : to_insert) {
        std::vector<std::vector<double>> points;
        for(int v : simplex) {points.push_back(vertex_coords[v]);}
        bool contains_simplex = st_custom.contains(simplex); 
        for(int v : simplex) {
            points.push_back(vertex_coords[v]);
        }
        double filtration = meb_radius_squared(points);
        std::sort(simplex.begin(), simplex.end());
        if(!contains_simplex) {
            st_custom.insert_simplex(simplex, filtration);
        } else {
            st_custom.assign_filtration(simplex, filtration);
        }
    }
    
    st_custom.make_filtration_non_decreasing();
    vertex_ids.erase(vh_to_remove);
    id_to_vertex.erase(removed_id);
    vertex_coords.erase(removed_id);
}

void draw_projection(const DT& dt) {
    DT2 dt2;

    for (auto vh = dt.finite_vertices_begin();
         vh != dt.finite_vertices_end(); ++vh)
    {
        const auto& p = vh->point();

        if (p.dimension() < 2)
            continue;

        dt2.insert(K2::Point_2(
            CGAL::to_double(p.cartesian(0)),
            CGAL::to_double(p.cartesian(1))));
    }

    CGAL::draw(dt2);
}

std::vector<double> point_to_vector(const DT::Point& p) {
    std::vector<double> v;
    v.reserve(p.dimension());

    for (int i = 0; i < p.dimension(); ++i)
        v.push_back(CGAL::to_double(p.cartesian(i)));

    return v;
}

Simplex_tree build_cech_delaunay_complex(const DT& dt) {
    Simplex_tree st;

    for (auto cell = dt.finite_full_cells_begin();
         cell != dt.finite_full_cells_end(); ++cell)
    {
        std::vector<int> maximal_simplex;
        maximal_simplex.reserve(dt.current_dimension() + 1);

        for (int i = 0; i <= dt.current_dimension(); ++i)
            maximal_simplex.push_back(vertex_ids.at(cell->vertex(i)));

        std::sort(maximal_simplex.begin(), maximal_simplex.end());

        simplex_set faces;
        generate_faces(maximal_simplex, faces);

        for (const auto& simplex : faces)
        {
            if (st.find(simplex) != st.null_simplex())
                continue;

            std::vector<std::vector<double>> pts;
            pts.reserve(simplex.size());

            for (int id : simplex)
            {
                const auto& p = id_to_vertex.at(id)->point();
                pts.emplace_back(p.cartesian_begin(), p.cartesian_end());
            }
            double r2 = meb_radius_squared(pts);
            st.insert_simplex(simplex, r2);
        }
    }
    // st.make_filtration_non_decreasing();
    return st;
}

vector<std::pair<vector<int>, double>> cech_delaunay(const DT&dt) {

    vector<std::pair<vector<int>, double>> simplex_filt_list;
    Simplex_tree st;
    simplex_set faces;
    std::vector<vector<int>> vect_faces;
    for (auto cell = dt.finite_full_cells_begin();
         cell != dt.finite_full_cells_end(); ++cell)
    {
        std::vector<int> maximal_simplex;
        maximal_simplex.reserve(dt.current_dimension() + 1);

        for (int i = 0; i <= dt.current_dimension(); ++i)
            maximal_simplex.push_back(vertex_ids.at(cell->vertex(i)));

        std::sort(maximal_simplex.begin(), maximal_simplex.end());

        generate_faces(maximal_simplex, faces);
        for(auto face : faces) {
            vect_faces.push_back(face);
        }
    }
    
    std::set<vector<int>> faces_vect;
    for(auto face : vect_faces) {faces_vect.insert(face);}
    for (const auto& simplex : faces_vect) {
            std::vector<std::vector<double>> pts;
            pts.reserve(simplex.size());
            
            for (int id : simplex)
            {
                const auto& p = id_to_vertex.at(id)->point();
                pts.emplace_back(p.cartesian_begin(), p.cartesian_end());
            }
            double r2 = meb_radius_squared(pts);
            if(st.find(simplex) == st.null_simplex()) {
                st.insert_simplex(simplex, r2);
                simplex_filt_list.push_back(make_pair(simplex, r2));
            } else {
                //st.insert_simplex(simplex, r2);
            }
    }
    // st.make_filtration_non_decreasing();
    return simplex_filt_list;
}

void print_simplex_filt_list(vector<std::pair<vector<int>, double>> lst) {
    std::sort(lst.begin(), lst.end(), [](const auto& a, const auto& b){
                    if(a.second != b.second)
                        return a.second > b.second;
                    else
                        return a.first.size() > b.first.size();
                });
    for(auto simplex : lst) {
        cout << "{";
        bool first = true;
        for(auto vertex: simplex.first) {
            if(!first) {
                cout <<", ";
            }
            first = false;
            cout << vertex;
        }
        cout <<"} " << simplex.second << "\n";
    }
}

DT clone_DT(const DT& dt) {
    DT copy(dt.current_dimension());
    for (auto vh = dt.finite_vertices_begin();vh != dt.finite_vertices_end(); ++vh) {
        DT::Vertex_handle vh_copy = copy.insert(vh->point());
    }

    return copy;
}

void copy_delaunay(const DT& src, DT& dst) {
    dst.clear();

    for (auto vh = src.finite_vertices_begin();
         vh != src.finite_vertices_end(); ++vh)
    {
        dst.insert(vh->point());
    }
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

int main() {
    int dim = 2;
    DT dt(dim);
    vector<double> a = {0,0};
    vector<double> b = {1,0};
    vector<double> c = {0,1};
    vector<double> d = {1,1};

    SimplexTree<int> st_custom;
    Point pa = Point(dim, a.begin(), a.begin()+dim);
    Point pb = Point(dim, b.begin(), b.begin()+dim);
    Point pc = Point(dim, c.begin(), c.begin()+dim);
    Point pd = Point(dim, d.begin(), d.begin()+dim);
    
    Vertex_handle v0 = dt.insert(pa);
    Vertex_handle v1 = dt.insert(pb);
    Vertex_handle v2 = dt.insert(pc);
    Vertex_handle v3 = dt.insert(pd);

    st_custom.insert_simplex({0}, 0.0);
    st_custom.insert_simplex({1}, 0.0);
    st_custom.insert_simplex({2}, 0.0);
    st_custom.insert_simplex({3}, 0.0);

    st_custom.insert_simplex({0,1}, 0.25);
    st_custom.insert_simplex({0,2}, 0.25);
    st_custom.insert_simplex({1,3}, 0.25);
    st_custom.insert_simplex({2,3}, 0.25);
    
    st_custom.insert_simplex({1,2}, 0.5);  
    st_custom.insert_simplex({1,2,3}, 0.5);
    st_custom.insert_simplex({0,1,2}, 0.5);
    
    st_custom.make_filtration_non_decreasing();

    simplex_count[{0}] = 1;
    simplex_count[{1}] = 2;
    simplex_count[{2}] = 2;
    simplex_count[{3}] = 1;

    simplex_count[{0,1}] = 1;
    simplex_count[{0,2}] = 1;
    simplex_count[{1,3}] = 1;
    simplex_count[{2,3}] = 1;
    simplex_count[{1,2}] = 2;

    simplex_count[{0,1,2}] = 1;
    simplex_count[{1,2,3}] = 1;

    vertex_ids[v0] = 0;
    id_to_vertex[0] = v0;
    vertex_coords[0] = a;
    
    vertex_ids[v1] = 1;
    id_to_vertex[1] = v1;
    vertex_coords[1] = b;
    
    vertex_ids[v2] = 2;
    id_to_vertex[2] = v2;
    vertex_coords[2] = c;
    
    vertex_ids[v3] = 3;
    id_to_vertex[3] = v3;
    vertex_coords[3] = d;
    

    int count_to_push = 10000;
    const int window_size = 500;
    int num_erases = window_size;
    std::vector<Point> points_to_insert;
    int curr_id = 4;
    for(int i = 0; i < count_to_push; ++i) {
        double x = dist(gen);
        double y = dist(gen);
        std::vector<double> coords = {x, y};
        Point pt(dim,coords.begin(),coords.end());
        vertex_coords[curr_id] = coords;
        curr_id++;
        points_to_insert.push_back(pt);
    }

    std::vector<Point> points = {pa, pb, pc, pd}; 
    auto start = std::chrono::high_resolution_clock::now();

    std::deque<DT::Vertex_handle> window;
    window.push_back(v0);
    window.push_back(v1);
    window.push_back(v2);
    window.push_back(v3);

    bool done = false;
    bool only_delete = false;
    DT dt_prev(dim);
    copy_delaunay(dt, dt_prev);
    std::vector<double> filtrations = {0, 0.001, 0.0025, 0.005,0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5};
    for(int i = 0; i < count_to_push; ++i) {
        auto pt = points_to_insert[i];

        // Insert a new point to a DT, update maps
        if(!only_delete) {
        copy_delaunay(dt, dt_prev);
        Vertex_handle vh = insert_point(dt, st_custom,pt, vertex_ids, id_to_vertex, vertex_coords);
            window.push_back(vh);
        }
        std::vector<Vertex_handle> temp(window.begin(),window.end());
        std::vector<Point> window_points;
        int idx = 0;
        for(auto vh : temp) {
            window_points.push_back(vh->point());
        }
        auto st_recomputed = build_cech_delaunay_complex(dt);

        Simplex_tree window_alpha_st;
        Alpha_complex window_alpha_complex(window_points);
        window_alpha_complex.create_complex(window_alpha_st);
        bool mismatch = false;
        double correct_ratio = check_ECC(st_custom, window_alpha_st, 5);
        if(correct_ratio != 1.00) {
                cout <<"ALARM\n";
        }
        cout << "ECC is correct on " << correct_ratio * 100.0 <<" % values\n";
        /*
        for(double t : filtrations) {
                int predicted =  st_custom.ECC(t);
                int recalculated = ECC(window_alpha_st, t);
                std::cout << predicted <<  " " << recalculated << std::endl;
                if(predicted != recalculated) {
                    cout <<"Euler curve mismatch\n";
                    mismatch = true;
                }
        }
        */
        auto simplex_dim = simplex_count_by_dim(window_alpha_st);
        auto vec = simplex_count_by_dim(st_recomputed);
        for(int d = 0; d <= dim;d++) {
            int a = st_custom.num_simplices(d);
            // std::cout << a << " " << simplex_dim[d] << " " << vec[d] << std::endl;
            if(a != simplex_dim[d]) {
                cout << "MISMATCH IN NUM OF SIMPLICES after insertion" << endl;
                mismatch = true;
            }
        }
        if(mismatch) {
            vector<vector<double>> points_double;
            for(auto pt: window_points) {
                points_double.push_back(point_to_vector(pt));
            }
            cout<<"Predicted tree:\n";
            cout <<"=================\n";
            auto lst_custom = st_custom.simplex_tree_to_simplex_list();
            print_simplex_filt_list(lst_custom);
            cout <<"=================\n";
            cout<<"Recomputed tree\n";
            cout <<"=================\n";
            auto lst = simplex_tree_to_simplex_list(st_recomputed);
            auto simplex_list = cech_delaunay(dt);
            print_simplex_filt_list(simplex_list);
            cout <<"=================\n";
        }
        if(mismatch) {
            draw_projection(dt_prev);
            draw_projection(dt);
        }
        if(window.size() > window_size || only_delete) {
            // only_delete = true; // uncomment to only add points and then to only erase, without mixing two operations
            Vertex_handle vh = window.front();
            auto st_recomputed = build_cech_delaunay_complex(dt);
            auto cofaces = get_vertex_cofaces(st_recomputed, vertex_ids[vh]);
            /*
            cout<<"HAVE TO DELETE THE FOLLOWIING:\n";
            cout<<"=============================\n";
            print_simplex_list(cofaces);
            cout<<"=============================\n";
            */
            window.pop_front();
            copy_delaunay(dt, dt_prev);
            
            erase_point(dt, st_custom, vh, vertex_ids, id_to_vertex, vertex_coords);
            std::vector<Vertex_handle> temp(window.begin(),window.end());

            std::vector<Point> window_points;
            for(auto vh : temp) {
                window_points.push_back(vh->point());
            }
            Alpha_complex window_alpha_complex_new(window_points);
            Simplex_tree window_alpha_st;
            window_alpha_complex_new.create_complex(window_alpha_st);
            st_recomputed = build_cech_delaunay_complex(dt);
            auto vec = simplex_count_by_dim(st_recomputed);
            double correct_ratio = check_ECC(st_custom, window_alpha_st, 5);
            if(correct_ratio != 1.00) {
                cout <<"ALARM\n";
            }
            cout << "ECC is correct on " << correct_ratio * 100.0 <<" % values";
            /*
            bool mismatch = false;
            for(double t : filtrations) {
                int predicted =  st_custom.ECC(t);
                int recalculated = ECC(window_alpha_st, t);
                std::cout << predicted <<  " " << recalculated << std::endl;
                if(predicted != recalculated) {
                    cout <<"Euler curve mismatch\n";
                    mismatch = true;
                }
            }
            */
            
            auto simplex_dim = simplex_count_by_dim(window_alpha_st);
            for(int d = 0; d <= dim;d++) {
                int a = st_custom.num_simplices(d);
                // std::cout << a << " " << simplex_dim[d] << " " << vec[d] << std::endl;
                if(a != simplex_dim[d]) {
                    cout << "MISMATCH IN NUM OF SIMPLICES after removal" << endl;
                    mismatch = true;
                }
            }
            st_recomputed = build_cech_delaunay_complex(dt);
            if(mismatch) {
                    draw_projection(dt_prev);
                    draw_projection(dt);
                    vector<vector<double>> points_double;
                    for(auto pt: window_points) {
                        points_double.push_back(point_to_vector(pt));
                    }
                    cout<<"Predicted tree:\n";
                    cout <<"=================\n";
                    auto lst_custom = st_custom.simplex_tree_to_simplex_list();
                    print_simplex_filt_list(lst_custom);
                    cout <<"=================\n";
                    cout<<"Recomputed tree\n";
                    cout <<"=================\n";
                    auto lst = simplex_tree_to_simplex_list(st_recomputed);
                    auto simplex_list = cech_delaunay(dt);
                    print_simplex_filt_list(simplex_list);
                    cout <<"=================\n";
            }
            cout <<endl;   
        }

    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time for k=" << count_to_push <<" " << " : " << elapsed.count() << " seconds\n";
    points.insert(points.end(), points_to_insert.begin(), points_to_insert.end());
    Alpha_complex alpha_complex(points);
    Simplex_tree alpha_st;
    // print_st(st);
    // std::cout << std::endl << std::endl;
    alpha_complex.create_complex(alpha_st);
    return 0;
}