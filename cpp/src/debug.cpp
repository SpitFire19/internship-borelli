#include "debug.h"
#include "algorithms.h"

#include <iostream>
#include <utility>


vector<vector<int>> simplex_list_by_dim(
    Simplex_tree& st,
    int dim)
{
    vector<vector<int>> simplex_of_dim;

    for (auto sh : st.complex_simplex_range()) {
        int d = 0;
        vector<int> simplex;

        for (auto v : st.simplex_vertex_range(sh)) {
            simplex.push_back(v);
            ++d;
        }

        std::sort(simplex.begin(), simplex.end());

        if (d == dim + 1) {
            simplex_of_dim.push_back(simplex);
        }
    }

    return simplex_of_dim;
}


void print_simplex_list(
    const vector<vector<int>>& simplex_list)
{
    for (const auto& simplex : simplex_list) {
        for (auto v : simplex) {
            std::cout << v << " ";
        }

        std::cout << std::endl;
    }
}


void compare_deltas(
    SimplexTree<int>& st_custom,
    Simplex_tree& st)
{
    auto sd_true = simplex_count_by_dim(st);

    /*
     * The original code used:
     *
     *     for (int d = 0; d <= dim; ++d)
     *
     * but `dim` is not declared locally in the function.
     *
     * Using st.dimension() makes the function self-contained.
     */
    const int max_dim = st.dimension();

    for (int d = 0; d <= max_dim; ++d) {
        int sd_pred = st_custom.num_simplices(d);
        int delta = sd_pred - sd_true[d];

        std::cout << '\n' << delta;

        if (delta != 0) {
            std::cout << "\nDIFF found\n";
        }

        std::cout << "\nTrue Simplex tree simplices\n";

        auto simplex_list = simplex_list_by_dim(st, d);
        print_simplex_list(simplex_list);

        std::cout << "Predicted simplex tree simplices\n";
        st_custom.print_simplices_of_dimension(d);
    }

    std::cout << std::endl;
}


double check_ECC(
    SimplexTree<int>& st_custom,
    Simplex_tree& st_recalc,
    double tmax)
{
    const int n_values = 100;
    int vals_correct = 0;

    for (int i = 0; i < n_values; ++i) {
        double t = tmax * i / n_values;

        int predicted = st_custom.ECC(t);
        int recalculated = ECC(st_recalc, t);

        if (predicted == recalculated) {
            ++vals_correct;
        }
    }

    return static_cast<double>(vals_correct) / n_values;
}


void copy_delaunay(
    const DT& src,
    DT& dst)
{
    dst.clear();

    for (auto vh = src.finite_vertices_begin();
         vh != src.finite_vertices_end();
         ++vh)
    {
        dst.insert(vh->point());
    }
}


vector<vector<int>> get_vertex_cofaces(
    const Simplex_tree& st,
    int vertex_id)
{
    std::vector<std::vector<int>> cofaces;

    auto sh = st.find({vertex_id});

    if (sh == st.null_simplex()) {
        return cofaces;
    }

    for (auto cf : st.cofaces_simplex_range(sh, 0)) {
        std::vector<int> simplex;

        for (auto v : st.simplex_vertex_range(cf)) {
            simplex.push_back(v);
        }

        std::sort(simplex.begin(), simplex.end());
        cofaces.push_back(std::move(simplex));
    }

    return cofaces;
}