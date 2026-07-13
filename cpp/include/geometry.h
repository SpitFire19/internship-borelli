#pragma once
#include "types.h"

// Generate a random point in d-dimensional space
vector<double> rand_point_dD(int dim);

// Convert a Delaunay triangulation point to std::vector<double>
vector<double> point_to_vector(const DT::Point& p);

// Generate all sub-simplices of a simplex
void generate_faces(
    const std::vector<int>& simplex,
    simplex_set& faces
);

// Calculate the squared radius of the minimum enclosing ball
double meb_radius_squared(
    const std::vector<std::vector<double>>& pts
);

// Return true if a full cell contains the infinite point,
// i.e. if it lies on the convex hull
bool is_infinite_cell(
    DT& dt,
    Full_cell_handle cell
);