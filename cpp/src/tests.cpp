#include<queue>
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "st.h"

SimplexTree st_custom;

void simple_remove(std::vector<std::vector<int>> &simplices_to_remove) {
    std::sort(simplices_to_remove.begin(),simplices_to_remove.end(),
        [](const auto& a, const auto& b){
            return a.size() > b.size(); 
        });
        for(auto simplex : simplices_to_remove) {
            st_custom.remove_maximal_simplex(simplex);
        }

    if(st_custom.contains({3, 4})) {
        std::cout <<"still contains {3, 4}" << std::endl;
    }

}

int main(void) {
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

    st_custom.insert_simplex({3, 4}, 0.5);
    st_custom.insert_simplex({1, 3, 4}, 0.75); 
    st_custom.insert_simplex({4}, 0.0);  
    st_custom.insert_simplex({1, 4}, 0.5); 
    st_custom.insert_simplex({0, 1, 4}, 0.4); 
    st_custom.insert_simplex({0, 4}, 0.8);
    std::vector<std::vector<int>> to_remove = {{1, 3, 4} , {3, 4}};
    simple_remove(to_remove);
    // st_custom.remove_maximal_simplex({1, 3, 4});
    // st_custom.remove_maximal_simplex({3, 4});
    // st_custom.print_simplices_of_dimension(1);
    // st_custom.print_simplices_of_dimension(2);
    // st_custom.print();
    auto cofaces = st_custom.cofaces({0});
    for(auto simplex : cofaces) {
        for(auto v: simplex) {
            std::cout << v << " ";
        }
        std::cout << '\n'; 
    }

}