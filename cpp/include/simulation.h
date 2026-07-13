#pragma once
#include "types.h"
#include "debug.h"
#include "geometry.h"
#include "algorithms.h"

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
        std::set<vector<int>> all_faces;

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
    Vertex_handle insert_point(Point pt) {
        auto start_func = std::chrono::high_resolution_clock::now();
        DT::Locate_type lt;
        DT::Face face(this->dt.maximal_dimension());
        DT::Facet facet;

        Full_cell_handle start = this->dt.locate(pt,lt,face,facet);
        std::vector<Full_cell_handle> conflict_cells;
        simplex_map delta_count;
        this->dt.compute_conflict_zone(pt,start,std::back_inserter(conflict_cells));

        std::vector<std::vector<int>> max_deleted_list;
        std::vector<std::vector<int>> max_inserted_list;
        for(const auto& cell : conflict_cells) {
            if(is_infinite_cell(this->dt, cell)) continue;

            std::vector<int> full_cell;
            for(int i = 0; i <= this->dt.current_dimension(); ++i) {
                auto vh = cell->vertex(i);
                if(this->dt.is_infinite(vh))
                    continue;
                full_cell.push_back(this->get_vertex_id(vh));
            }
            std::sort(full_cell.begin(), full_cell.end());
            max_deleted_list.push_back(full_cell);

        }
        if(DEBUG) {
            cout << "Full cells before insertion\n";
            cout << "===============================\n";
            for (auto cell = this->dt.finite_full_cells_begin();
             cell != this->dt.finite_full_cells_end(); ++cell) {
                std::vector<int> maximal_simplex;
                maximal_simplex.reserve(this->dt.current_dimension() + 1);

                for (int i = 0; i <= this->dt.current_dimension(); ++i)
                    maximal_simplex.push_back(this->get_vertex_id(cell->vertex(i)));

                std::sort(maximal_simplex.begin(), maximal_simplex.end());
                print_simplex_list({maximal_simplex});
            }
            cout << "===============================\n";
            cout<<"Conflict full cells during insertion\n";
            cout << "===============================\n";
            print_simplex_list(max_deleted_list);
            cout << "===============================\n";
        }
        Vertex_handle vh = this->add_triangulation_point(pt);
        
        std::vector<Full_cell_handle> incident_cells;
        this->dt.incident_full_cells(vh,std::back_inserter(incident_cells));
        for(const auto& cell : incident_cells) {
            if(is_infinite_cell(this->dt, cell))continue;
            std::vector<int> simplex;
            for(int i = 0; i <= this->dt.current_dimension(); ++i) {
                auto vh = cell->vertex(i);
                if(this->dt.is_infinite(vh))
                continue;
                simplex.push_back(this->get_vertex_id(vh));
            }
            std::sort(simplex.begin(), simplex.end());
            max_inserted_list.push_back(simplex);
        }
        if(DEBUG) {
            cout << "Full cells after insertion\n";
            cout << "===============================\n";
            for (auto cell = this->dt.finite_full_cells_begin();
             cell != this->dt.finite_full_cells_end(); ++cell) {
                std::vector<int> maximal_simplex;
                maximal_simplex.reserve(this->dt.current_dimension() + 1);

                for (int i = 0; i <= this->dt.current_dimension(); ++i)
                    maximal_simplex.push_back(this->get_vertex_id(cell->vertex(i)));

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
            for(auto kv : this->simplex_count) {
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
            int old_count = this->simplex_count[f];
            int new_count = old_count + delta;
            this->simplex_count[f] = new_count;
            if(old_count > 0 && new_count == 0) {
                to_delete.push_back(f);
            }
            if(old_count == 0 && new_count > 0) to_insert.push_back(f);
        }

        if(DEBUG) {
            cout <<"Removing simplices during insertion :\n";
            print_simplex_list(to_delete);
        }

        simple_remove(to_delete, this->st);
        if(DEBUG) {
            cout <<"Adding simplices:\n";
            print_simplex_list(to_insert);
        }
        bool flag = false;
        for(const auto& f : to_insert) {
            std::vector<std::vector<double>> points;
            for(int v : f) {
                points.push_back(this->get_vertex_coords_by_id(v));
            }
            double filtration = meb_radius_squared(points);
            if(!this->st.contains(f)) {
                this->st.insert_simplex(f, filtration); 
            } else {
                if(filtration < this->st.filtration(f))
                    this->st.assign_filtration(f, filtration);
            }
        }
        auto start_filt = std::chrono::high_resolution_clock::now();
        this->st.make_filtration_non_decreasing();
        auto end_filt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_filt = end_filt - start_filt;
        std::chrono::duration<double> before_filt = start_filt - start_func;
        cout <<"Ratio of make_filt_non_decreasing in runtime for insert: " << elapsed_filt / (elapsed_filt + before_filt) * 100.0 <<"%\n";
        return vh;
}

// erase a point from a simplicial complex represnted by a
// custom implementation of a Simplex_tree
void erase_point(Vertex_handle& vh_to_remove) {
    auto start_func = std::chrono::high_resolution_clock::now();
    std::vector<Full_cell_handle> old_incident_cells;
    this->dt.incident_full_cells(vh_to_remove,std::back_inserter(old_incident_cells));

    std::unordered_set<Vertex_handle> neighbor_vertices;
    for(const auto& cell :old_incident_cells) {
        if(is_infinite_cell(this->dt, cell)) continue;
        std::vector<int> simplex;
        for(int i = 0;i <= this->dt.current_dimension();++i) {
            auto vh = cell->vertex(i);
            if(vh != vh_to_remove) neighbor_vertices.insert(vh);
            simplex.push_back(this->get_vertex_id(vh));
        }
        std::sort(simplex.begin(), simplex.end());
        simplex_set local_faces;
        generate_faces(simplex,local_faces);
    }

    std::unordered_set<int> cavity_vertex_ids;
    for(const auto& vh : neighbor_vertices) {
        cavity_vertex_ids.insert(this->get_vertex_id(vh));
    }

    std::unordered_map<std::vector<int>,int,VectorHash> boundary_count;

    for(const auto& cell : old_incident_cells) {
        if(is_infinite_cell(this->dt, cell))
            continue;

        std::vector<int> facet;

        for(int i = 0; i <= this->dt.current_dimension(); ++i) {
            auto vh = cell->vertex(i);
            if(vh == vh_to_remove)
                continue;
            facet.push_back(this->get_vertex_id(vh));
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
        for (auto cell = this->dt.finite_full_cells_begin();
            cell != this->dt.finite_full_cells_end(); ++cell) {
            std::vector<int> maximal_simplex;
            maximal_simplex.reserve(this->dt.current_dimension() + 1);
            for (int i = 0; i <= this->dt.current_dimension(); ++i)
                maximal_simplex.push_back(this->get_vertex_id(cell->vertex(i)));
            std::sort(maximal_simplex.begin(), maximal_simplex.end());
            print_simplex_list({maximal_simplex});
            }
            cout << "===============================\n";
        }
        const int removed_id = this->get_vertex_id(vh_to_remove);
        this->dt.remove(vh_to_remove);
        auto v = std::vector<int> {removed_id};
        std::vector<std::vector<int>> simplices_to_remove = this->st.cofaces(v);

        if(DEBUG) {
            cout<<"Simplex_count by point\n";
            cout << "===============================\n";
            for(auto kv : this->simplex_count) {
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
            this->dt.incident_full_cells(vh,std::back_inserter(local_cells));
            for(const auto& cell : local_cells) {
                if(is_infinite_cell(this->dt, cell))
                    continue;
                bool inside_cavity = true;
                for(int i = 0;i <= this->dt.current_dimension();++i) {
                    auto cvh = cell->vertex(i);
                    int id = this->get_vertex_id(cvh);

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
            if(is_infinite_cell(this->dt, cell))continue;
            std::vector<int> simplex;

            for(int i = 0;i <= this->dt.current_dimension();++i) {
                auto vh = cell->vertex(i);
                if(this->dt.is_infinite(vh))
                    continue;
                simplex.push_back(this->get_vertex_id(vh));
            }
            std::sort(simplex.begin(), simplex.end());
        
            if(this->simplex_count[simplex] > 0) {
                // cout <<"An existing cell has been preserved\n";
                continue;
            }

            simplex_set local_faces;
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
            for (auto cell = this->dt.finite_full_cells_begin();
             cell != this->dt.finite_full_cells_end(); ++cell) {
                std::vector<int> maximal_simplex;
                maximal_simplex.reserve(this->dt.current_dimension() + 1);
                for (int i = 0; i <= this->dt.current_dimension(); ++i)
                    maximal_simplex.push_back(this->get_vertex_id(cell->vertex(i)));
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
            int old_count = this->simplex_count[f];
            int new_count = old_count + delta;
            if(new_count > 1 && f.size() == dim + 1) {
                cout << "BAD FULL CELL IN simplex_count during erase \n";
                new_count = 1;
            }
            this->simplex_count[f] = new_count;
            if(old_count > 0 && new_count <= 0) to_delete.push_back(f);
            if(old_count == 0 && new_count > 0) to_insert.push_back(f);
        }

        //simple_remove(to_delete, st_custom, MOD_ERASE);
        this->st.removeSimplexAndCofaces(v);

        for(auto simplex : to_insert) {
            std::vector<std::vector<double>> points;
            for(int v : simplex) {points.push_back(this->get_vertex_coords_by_id(v));}
            bool contains_simplex = this->st.contains(simplex); 
            for(int v : simplex) {
                points.push_back(this->get_vertex_coords_by_id(v));
            }
            double filtration = meb_radius_squared(points);
            std::sort(simplex.begin(), simplex.end());
            if(!contains_simplex) {
                this->st.insert_simplex(simplex, filtration);
            } else {
                this->st.assign_filtration(simplex, filtration);
            }
        }
        auto start_filt = std::chrono::high_resolution_clock::now();
        this->st.make_filtration_non_decreasing();
        auto end_filt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_filt = end_filt - start_filt;
        std::chrono::duration<double> before_filt = start_filt - start_func;
        cout <<"Ratio of make_filt_non_decreasing in runtime for erase: " << elapsed_filt / (elapsed_filt + before_filt) * 100.0 <<"%\n";
        this->remove_point_from_state(removed_id);
    }

    // re-build Cech-Delaunay filtration over the triangulation contained in simulation state
    Simplex_tree build_reference_complex() const  {
        Simplex_tree st;
        std::set<std::vector<int>> all_faces;

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
            st.insert_simplex(simplex, filtration);
        }
        st.make_filtration_non_decreasing();
        return st;
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


