#pragma once
#include "types.h"
#include "geometry.h"

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

void simple_remove(std::vector<std::vector<int>> simplices_to_remove, SimplexTree<int>&st_custom, int mode) {
    std::sort(simplices_to_remove.begin(),simplices_to_remove.end(),
        [](const auto& a, const auto& b){
            return a.size() > b.size(); 
        });
    for(auto simplex : simplices_to_remove) {
        st_custom.remove_maximal_simplex(simplex);
    }
}
