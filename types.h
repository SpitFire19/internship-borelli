#pragma once
#include <random>
#include <vector>
#include <map>

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

#include "st.h"

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
const int dim = 3;
// class used to define the smallest enclosing sphere for a simplex
typedef CGAL::Min_sphere_of_spheres_d_traits_d<
    K_cartesian,
    FT,
    dim
> MEB_Traits;

typedef CGAL::Min_sphere_of_spheres_d<MEB_Traits > Min_sphere;
typedef MEB_Traits::Sphere Sphere;
typedef Gudhi::alpha_complex::Alpha_complex<K> Alpha_complex;

std::random_device rd;
std::mt19937 gen(42);
double low = -5.0;
double high = 5.0;
std::uniform_real_distribution<double> dist(low, high); // Unif([-5, 5])

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

using std::vector;