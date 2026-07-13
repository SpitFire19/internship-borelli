#include "geometry.h"

namespace {
    std::mt19937 gen(42);
    double low = -5.0;
    double high = 5.0;
    std::uniform_real_distribution<double> dist(low, high);
}

vector<double> rand_point_dD(int dim)
{
    vector<double> point;
    point.reserve(dim);

    for (int i = 0; i < dim; ++i) {
        double coord = dist(gen);
        point.push_back(coord);
    }

    return point;
}


vector<double> point_to_vector(const DT::Point& p)
{
    vector<double> v;
    v.reserve(p.dimension());

    for (int i = 0; i < p.dimension(); ++i) {
        v.push_back(CGAL::to_double(p.cartesian(i)));
    }

    return v;
}


void generate_faces(
    const std::vector<int>& simplex,
    simplex_set& faces)
{
    int n = simplex.size();

    for (int mask = 1; mask < (1 << n); ++mask) {
        vector<int> face;
        face.reserve(n);

        for (int i = 0; i < n; ++i) {
            if (mask & (1 << i)) {
                face.push_back(simplex[i]);
            }
        }

        std::sort(face.begin(), face.end());
        faces.insert(face);
    }
}


double meb_radius_squared(
    const std::vector<std::vector<double>>& pts)
{
    std::vector<Sphere> spheres;

    if (pts.size() == 1) {
        return 0.0;
    }

    for (const auto& v : pts) {
        K_cartesian::Point_d p(v.size(), v.begin(), v.end());
        spheres.push_back(Sphere(p, 0.0));
    }

    Min_sphere ms(spheres.begin(), spheres.end());

    return ms.radius() * ms.radius();
}


bool is_infinite_cell(
    DT& dt,
    Full_cell_handle cell)
{
    for (int i = 0; i <= dt.current_dimension(); ++i) {
        if (dt.is_infinite(cell->vertex(i))) {
            return true;
        }
    }

    return false;
}