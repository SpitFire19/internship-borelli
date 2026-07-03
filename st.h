#pragma once
#include<queue>
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <gudhi/Simplex_tree.h>

typedef Gudhi::Simplex_tree<> Simplex_tree;

template <class Label = int>
class SimplexTree {
public:
    struct Node {
        Label label{};
        Node* parent{nullptr};
        std::size_t depth{0}; // root = 0, vertex = 1, edge = 2, ...
        std::map<Label, Node*> children;
        double filtration{0.0};
        Node() = default;
        Node(Label lbl, Node* p, std::size_t d, double filt=0.0) : label(lbl), parent(p), depth(d), filtration(filt) {}
    };

    SimplexTree() : root_(), face_count_(0) {}
    ~SimplexTree() { clear(); }

    // Number of stored simplices, excluding the empty simplex (root).
    std::size_t size() const { return face_count_; }

    void clear() {
        std::vector<Node*> to_delete;
        to_delete.reserve(root_.children.size());
        for (auto& kv : root_.children) to_delete.push_back(kv.second);

        for (Node* child : to_delete) {
            eraseSubtree(child);
        }
        root_.children.clear();
        index_.clear();
        face_count_ = 0;
    }

    // Normalize to strictly increasing unique labels.
    static std::vector<Label> normalize(std::vector<Label> simplex) {
        std::sort(simplex.begin(), simplex.end());
        simplex.erase(std::unique(simplex.begin(), simplex.end()), simplex.end());
        return simplex;
    }

    // Exact search: returns the node representing simplex, or nullptr.
    Node* findNode(std::vector<Label> simplex) const {
        simplex = normalize(std::move(simplex));
        Node* cur = const_cast<Node*>(&root_);
        for (const Label& v : simplex) {
            auto it = cur->children.find(v);
            if (it == cur->children.end()) return nullptr;
            cur = it->second;
        }
        return cur;
    }

    bool contains(std::vector<Label> simplex) const {
        return findNode(std::move(simplex)) != nullptr;
    }

    // Insert one simplex (all its faces are NOT inserted automatically).
    // Returns the node representing the simplex.
    Node* insert_simplex(std::vector<Label> simplex, double filt) {
        simplex = normalize(std::move(simplex));
        Node* cur = &root_;

        for (const Label& v : simplex) {
            auto it = cur->children.find(v);
            if (it == cur->children.end()) {
                Node* created = new Node(v, cur, cur->depth + 1, filt);
                cur->children.emplace(v, created);
                index_[v].push_back(created);
                cur = created;
                ++face_count_;
            } else {
                cur = it->second;
            }
        }
        return cur;
    }

    // Insert simplex together with all non-empty subfaces.
    // Complexity matches the paper's O(2^j Dm) idea.
    void insertClosedSimplex(std::vector<Label> simplex, double filt) {
        simplex = normalize(std::move(simplex));
        std::vector<Label> current;
        current.reserve(simplex.size());
        insertClosedSimplexRec(simplex, filt, 0, current);
    }

    // Locate all facets by checking each codimension-1 face.
    // This is O(j^2 Dm)-style in practice for a j-simplex.
    std::vector<std::vector<Label>> locateFacets(std::vector<Label> simplex) const {
        simplex = normalize(std::move(simplex));
        std::vector<std::vector<Label>> facets;

        if (simplex.size() <= 1) return facets;

        for (std::size_t i = 0; i < simplex.size(); ++i) {
            std::vector<Label> facet;
            facet.reserve(simplex.size() - 1);
            for (std::size_t j = 0; j < simplex.size(); ++j) {
                if (j != i) facet.push_back(simplex[j]);
            }
            if (contains(facet)) facets.push_back(std::move(facet));
        }
        return facets;
    }

    // Locate maximal roots of coface subtrees for a simplex.
    // This follows the paper's label-index idea: scan nodes whose label matches
    // last(simplex), then check whether the simplex labels appear on the path.
    std::vector<Node*> locateCofaceRoots(std::vector<Label> simplex) const {
        simplex = normalize(std::move(simplex));
        std::vector<Node*> roots;
        if (simplex.empty()) return roots;

        auto it = index_.find(simplex.back());
        if (it == index_.end()) return roots;

        for (Node* n : it->second) {
            if (n->depth < simplex.size()) continue;
            if (!pathContainsSubsequence(n, simplex)) continue;

            Node* p = n->parent;
            if (p != nullptr && p != &root_ && pathContainsSubsequence(p, simplex)) {
                continue; // not a subtree root
            }
            roots.push_back(n);
        }
        return roots;
    }

    // Remove a simplex and all its cofaces.
    // Returns false if no coface-root was found.
    bool removeSimplexAndCofaces(std::vector<Label> simplex) {
        auto roots = locateCofaceRoots(std::move(simplex));
        if (roots.empty()) return false;

        for (Node* n : roots) {
            if (n != nullptr) eraseSubtree(n);
        }
        return true;
    }

    // Debug / inspection helper.
    void print() const {
        std::vector<Label> cur;
        printRec(&root_, cur);
    }

private:
    Node root_;
    std::size_t face_count_;
    std::unordered_map<Label, std::vector<Node*>> index_;

    static bool isSubsequence(const std::vector<Label>& path,
                              const std::vector<Label>& target) {
        std::size_t i = 0, j = 0;
        while (i < path.size() && j < target.size()) {
            if (path[i] == target[j]) ++j;
            ++i;
        }
        return j == target.size();
    }

    static std::vector<Label> pathLabelsFromRoot(Node* n) {
        std::vector<Label> path;
        while (n != nullptr && n->parent != nullptr) {
            path.push_back(n->label);
            n = n->parent;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    bool pathContainsSubsequence(Node* n, const std::vector<Label>& simplex) const {
        std::vector<Label> path = pathLabelsFromRoot(n);
        return isSubsequence(path, simplex);
    }

    void eraseFromIndex(Node* n) {
        auto it = index_.find(n->label);
        if (it == index_.end()) return;

        auto& vec = it->second;
        vec.erase(std::remove(vec.begin(), vec.end(), n), vec.end());
        if (vec.empty()) index_.erase(it);
    }

    void eraseSubtree(Node* n) {
        if (n == nullptr || n == &root_) return;

        // Copy children first because recursive erasure mutates the map.
        std::vector<Node*> children;
        children.reserve(n->children.size());
        for (auto& kv : n->children) children.push_back(kv.second);

        for (Node* c : children) {
            eraseSubtree(c);
        }

        eraseFromIndex(n);

        Node* parent = n->parent;
        Label lbl = n->label;

        delete n;
        --face_count_;

        if (parent != nullptr) {
            parent->children.erase(lbl);
        }
    }

    void insertClosedSimplexRec(const std::vector<Label>& simplex, double filt,
                                std::size_t idx,
                                std::vector<Label>& current) {
        if (idx == simplex.size()) {
            if (!current.empty()) insert_simplex(current, filt);
            return;
        }

        // Include simplex[idx]
        current.push_back(simplex[idx]);
        insertClosedSimplexRec(simplex, filt, idx + 1, current);
        current.pop_back();

        // Exclude simplex[idx]
        insertClosedSimplexRec(simplex, filt, idx + 1, current);
    }

    void printRec(const Node* n, std::vector<Label>& cur) const {
        if (n != nullptr && n != &root_) {
            cur.push_back(n->label);
            std::cout << "{";
            for (std::size_t i = 0; i < cur.size(); ++i) {
                if (i) std::cout << ",";
                std::cout << cur[i];
            }
            std::cout << "}";
            std::cout << " " << n->filtration <<"\n";
        }

        for (auto& kv : n->children) {
            printRec(kv.second, cur);
        }

        if (n != &root_ && !cur.empty()) cur.pop_back();
    }
    double make_filtration_non_decreasing_rec(Node* node,bool& modified) {
        double maximum = node->filtration;

        for (auto& [_, child] : node->children) {
            maximum = std::max(
                maximum,
                make_filtration_non_decreasing_rec(
                    child,
                    modified));
        }

        if (node->filtration < maximum)
        {
            node->filtration = maximum;
            modified = true;
        }

        return node->filtration;
    }

    bool propagate_filtration_up(Node* node) {
        bool modified = false;

        while (node && node->parent) {

            double new_value = node->filtration;

            for (auto& [_, child] : node->children)
                new_value = std::max(new_value, child->filtration);

            if (new_value <= node->filtration)
                break;

            node->filtration = new_value;
            modified = true;

            node = node->parent;
        }

        return modified;
    }
    int eulerRec(Node* node, double t) const{
        int chi = 0;
        int dim = static_cast<int>(node->depth) - 1;
        if(dim == 0 && node->filtration !=0) std::cout <<"Point with filt = " << node->filtration << std::endl;
        if (node->filtration <= t) {
            
            if (dim % 2 == 0)
                chi += 1;
            else
                chi -= 1;
        }

        for (const auto& kv : node->children)
            chi += eulerRec(kv.second, t);

        return chi;
    }
    std::size_t count_dimension(Node* node, std::size_t dimension) const {
        std::size_t count = 0;

        if (node->depth - 1 == dimension)
            ++count;

        for (const auto& [_, child] : node->children)
            count += count_dimension(child, dimension);

        return count;
    }

    public:
        std::vector<Label> simplex_from_node(Node* node) const {
            std::vector<Label> simplex;
        
            while (node != nullptr &&
                   node != &root_)
            {
                simplex.push_back(node->label);
                node = node->parent;
            }
        
            std::sort(simplex.begin(), simplex.end());
            // std::reverse(simplex.begin(),simplex.end());
            return simplex;
        }

        double propagate_filtration(Node* node,bool& modified) {
            double maximum = node->filtration;

            for (auto& [_, child] : node->children) {
                maximum = std::max(maximum,propagate_filtration(child,modified));
            }

            if (node->filtration < maximum) {
                node->filtration = maximum;
                modified = true;
            }
            return node->filtration;
        }

        void collect_by_dimension(Node* node, std::vector<std::vector<Node*>>& levels)
        {
            if (node != &root_) {
                std::size_t dim = node->depth - 1;
                if (levels.size() <= dim) levels.resize(dim + 1);
                levels[dim].push_back(node);
            }
        
            for (auto& [_, child] : node->children)
                collect_by_dimension(child, levels);
        }

    bool make_filtration_non_decreasing()
    {
        bool modified = false;
        std::vector<std::vector<Node*>> levels;

        collect_by_dimension(&root_, levels);

        for (std::size_t dim = 1; dim < levels.size(); ++dim) {
            for (Node* node : levels[dim]) {

                auto simplex = simplex_from_node(node);
                double maximum = node->filtration;

                for (std::size_t i = 0; i < simplex.size(); ++i) {
                    std::vector<Label> facet;
                    facet.reserve(simplex.size() - 1);

                    for (std::size_t j = 0; j < simplex.size(); ++j)
                        if (i != j)
                            facet.push_back(simplex[j]);

                    Node* face = findNode(facet);

                    if (face)
                        maximum = std::max(maximum, face->filtration);
                }

                if (maximum > node->filtration) {
                    node->filtration = maximum;
                    modified = true;
                }
            }
        }

        return modified;
    }

        bool make_filtration_non_decreasing(const std::vector<Node*>& modified) {
            bool changed = false;

            std::queue<Node*> q;
            std::unordered_set<Node*> visited;

            for (Node* n : modified)
            {
                q.push(n);
                visited.insert(n);
            }

            while (!q.empty())
            {
                Node* node = q.front();
                q.pop();

                Node* parent = node->parent;

                if (parent == nullptr || parent == &root_)
                    continue;

                if (parent->filtration < node->filtration)
                {
                    parent->filtration =
                        node->filtration;

                    changed = true;

                    if (!visited.count(parent))
                    {
                        visited.insert(parent);
                        q.push(parent);
                    }
                }
            }

            return changed;
    }
    bool insert_simplex_list(const std::vector<std::vector<Label>>& simplices, const std::vector<double>& filtrations) {

        bool modified = false;

        std::queue<Node*> q;
        std::unordered_set<Node*> visited;


        for (std::size_t i = 0;i < simplices.size();++i)
        {
            Node* node =
                insert_simplex(simplices[i],filtrations[i]);

            //
            // Gudhi semantics:
            // keep the smallest filtration.
            //
            if (filtrations[i] < node->filtration)
                node->filtration = filtrations[i];

            if (!visited.count(node))
            {
                visited.insert(node);
                q.push(node);
            }
        }

        //
        // Propagate toward faces.
        //
        while (!q.empty())
        {
            Node* node = q.front();
            q.pop();

            Node* parent = node->parent;

            if (parent == nullptr ||
                parent == &root_)
            {
                continue;
            }

            if (parent->filtration <
                node->filtration)
            {
                parent->filtration =
                    node->filtration;

                modified = true;

                if (!visited.count(parent))
                {
                    visited.insert(parent);
                    q.push(parent);
                }
            }
        }

        return modified;
    }

        double ECC(double t) const {
            int chi = 0;
            for (const auto& kv : root_.children)
                chi += eulerRec(kv.second, t);
        
            return chi;
        }
        bool assign_filtration(std::vector<Label> simplex, double value) {
            Node* node = findNode(std::move(simplex));
        
            if (node == nullptr)
                return false;
        
            node->filtration = value;
        
            return true;
        }
        std::size_t num_simplices(std::size_t dimension) const {
            std::size_t count = 0;

            for (const auto& [_, child] : root_.children) {
                count += count_dimension(child, dimension);
            }

            return count;
        }
        bool has_children(Node* node) const {
            return node != nullptr &&
                   !node->children.empty();
        }

        bool has_children(const std::vector<Label>& simplex) const {
            Node* node = findNode(simplex);
        
            return node != nullptr &&
                   !node->children.empty();
        }

        bool is_maximal(Node* node) const {
            return node != nullptr &&
                   node->children.empty();
        }

        bool is_maximal(const std::vector<Label>& simplex) const {
            Node* node = findNode(simplex);
        
            return node != nullptr &&
                   node->children.empty();
        }
        bool remove_maximal_simplex(const std::vector<Label>& simplex) {
            Node* node = findNode(simplex);
            if (node == nullptr)
                return false;
        
            if (!node->children.empty())
                return false;
        
            eraseFromIndex(node);
        
            Node* parent = node->parent;
        
            Label lbl = node->label;
        
            delete node;
        
            --face_count_;
        
            if (parent != nullptr)
                parent->children.erase(lbl);
        
            return true;
        }
        void print_dimension_rec(Node* node,std::size_t dimension,std::vector<Label>& simplex) const  {
            if (node != &root_)
                simplex.push_back(node->label);

            if (node != &root_ && node->depth - 1 == dimension) {
                std::cout << "{";
                for (std::size_t i = 0; i < simplex.size(); ++i) {
                    if (i) std::cout << ",";
                    std::cout << simplex[i];
                }
                std::cout << "}  f = " << node->filtration << "\n";
            }
        
            for (auto& [_, child] : node->children)
                print_dimension_rec(child, dimension, simplex);
        
            if (node != &root_)
                simplex.pop_back();
        }
        void print_simplices_of_dimension(std::size_t dimension) const {
            std::vector<Label> simplex;        
            for (auto& [_, child] : root_.children)
                print_dimension_rec(child, dimension, simplex);
        }
        void collect_nodes(Node* node, std::vector<Node*>& nodes) {
            if (node != &root_)
                nodes.push_back(node);

            for (auto& [_, child] : node->children)
                collect_nodes(child, nodes);
        }
        void simplex_tree_to_simplex_list_rec(const Node* node,
    std::vector<std::pair<std::vector<Label>, double>>& simplex_list) const {
        if (node != &root_) {
            simplex_list.emplace_back(
                simplex_from_node(const_cast<Node*>(node)),
                node->filtration);
        }

        for (const auto& [_, child] : node->children)
            simplex_tree_to_simplex_list_rec(child, simplex_list);
    }
        std::vector<std::vector<Label>> cofaces(std::vector<Label> simplex) {
            auto s = normalize(simplex);
        
            std::vector<std::vector<Label>> result;
            std::vector<Node*> nodes;
        
            collect_nodes(&root_, nodes);
        
            for (Node* node : nodes) {
                if (node->depth < s.size())
                    continue;
            
                auto path = simplex_from_node(node);
            
                if (isSubsequence(path, s))
                    result.push_back(path);
            }
        
            return result;
        }
        std::vector<std::pair<std::vector<Label>, double>> 
        simplex_tree_to_simplex_list() const {
            std::vector<std::pair<std::vector<Label>, double>> simplex_list;
            simplex_tree_to_simplex_list_rec(&root_, simplex_list);
            return simplex_list;
        }
        double filtration(const std::vector<Label>& simplex) const {
            Node* node = findNode(simplex);    
            if (node == nullptr)
                throw std::runtime_error("Simplex not found in simplex tree.");

            return node->filtration;
        }
        void from_gudhi(const Simplex_tree & gudhi_st) {
            clear();

            for (auto sh : gudhi_st.complex_simplex_range()) {
                std::vector<Label> simplex;
            
                for (auto v : gudhi_st.simplex_vertex_range(sh))
                    simplex.push_back(static_cast<Label>(v));
            
                Node* node = insert_simplex(simplex, 0.0);
                node->filtration = gudhi_st.filtration(sh);
            }
        }
};
