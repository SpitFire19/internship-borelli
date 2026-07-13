#pragma once

#include "types.h"

// Set to true to print detailed messages about inserted
// and deleted simplices.
#define DEBUG false

#define MOD_INSERT 0
#define MOD_ERASE 1


// Return all simplices of the given dimension
// from a GUDHI Simplex_tree.
vector<vector<int>> simplex_list_by_dim(
    Simplex_tree& st,
    int dim
);


// Print a list of simplices to standard output.
void print_simplex_list(
    const vector<vector<int>>& simplex_list
);


// Compare a custom simplex tree with a GUDHI simplex tree
// and print simplex-count mismatches.
//
// Intended only for small numbers of simplices.
void compare_deltas(
    SimplexTree<int>& st_custom,
    Simplex_tree& st
);


// Compare the Euler Characteristic Curves of two simplex trees
// at 100 filtration values between 0 and tmax.
//
// Returns the fraction of matching values in [0, 1].
double check_ECC(
    SimplexTree<int>& st_custom,
    Simplex_tree& st_recalc,
    double tmax
);


// Copy all finite vertices from one Delaunay triangulation
// into another.
void copy_delaunay(
    const DT& src,
    DT& dst
);


// Return all cofaces containing the specified vertex.
vector<vector<int>> get_vertex_cofaces(
    const Simplex_tree& st,
    int vertex_id
);

using std::cout;
using std::endl;