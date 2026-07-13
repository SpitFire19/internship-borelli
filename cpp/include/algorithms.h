#pragma once
#include "types.h"

// Returns chi(t): the value of the Euler Characteristic curve
// at filtration value t.
int ECC(const Simplex_tree& st, double t);

// Returns information about simplex dimension.
// Note: the current implementation does not use parameter d.
int simplex_count_of_dimension(Simplex_tree& st, int d);

// Returns the number of simplices of each dimension,
// from dimension 0 up to the maximal dimension.
vector<int> simplex_count_by_dim(Simplex_tree& st);

// Removes simplices from the custom simplex tree,
// processing higher-dimensional simplices first.
void simple_remove(
    std::vector<std::vector<int>> simplices_to_remove,
    SimplexTree<int>& st_custom
);