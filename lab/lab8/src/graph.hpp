#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

constexpr float kInf = std::numeric_limits<float>::infinity();

struct Edge {
  int u, v;
  float w;
};

inline std::tuple<int, std::vector<Edge>, int> read_graph(const std::string& filepath) {
  std::ifstream fin(filepath);
  if (!fin.is_open()) {
    std::cerr << "ERROR: cannot open graph file: " << filepath << std::endl;
    std::exit(1);
  }

  std::vector<Edge> edges;
  std::string line;
  std::getline(fin, line); // skip header

  int min_id = std::numeric_limits<int>::max();
  int max_id = -1;

  while (std::getline(fin, line)) {
    if (line.empty()) continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream iss(line);
    int u, v;
    float w;
    if (!(iss >> u >> v >> w)) {
	    std::cerr << "WARNING: malformed line skipped: " << line << std::endl;
	    continue;
	  }
    edges.push_back({u, v, w});
    min_id = std::min(min_id, std::min(u, v));
    max_id = std::max(max_id, std::max(u, v));
  }

  // Determine 0-based or 1-based indexing
  int offset = (min_id == 1) ? 1 : 0;
  int num_nodes = max_id - offset + 1;

  // Apply offset to make all node IDs 0-based
  for (auto& e : edges) {
    e.u -= offset;
    e.v -= offset;
  }

  return std::make_tuple(num_nodes, std::move(edges), offset);
}

inline std::vector<std::vector<std::pair<int, float>>> build_adjacency(
    int num_nodes, const std::vector<Edge>& edges) {
  std::vector<std::vector<std::pair<int, float>>> adj(num_nodes);
  for (const auto& e : edges) {
    adj[e.u].emplace_back(e.v, e.w);
    adj[e.v].emplace_back(e.u, e.w); // undirected
  }
  // Sort for cache-friendly access
  for (auto& neighbors : adj) {
    std::sort(neighbors.begin(), neighbors.end());
  }
  return adj;
}

inline std::vector<float> dijkstra(
    int source,
    const std::vector<std::vector<std::pair<int, float>>>& adj) {
  int n = static_cast<int>(adj.size());
  std::vector<float> dist(n, kInf);
  dist[source] = 0.0f;

  using State = std::pair<float, int>;
  std::priority_queue<State, std::vector<State>, std::greater<State>> pq;
  pq.emplace(0.0f, source);

  while (!pq.empty()) {
    auto [d, u] = pq.top();
    pq.pop();
    if (d > dist[u]) continue;
    for (const auto& [v, w] : adj[u]) {
      float nd = d + w;
      if (nd < dist[v]) {
        dist[v] = nd;
        pq.emplace(nd, v);
      }
    }
  }
  return dist;
}

inline double compute_checksum(const std::vector<std::vector<float>>& dist_matrix) {
  double sum = 0.0;
  for (const auto& row : dist_matrix) {
    for (float d : row) {
      if (std::isfinite(d)) sum += static_cast<double>(d);
    }
  }
  return sum;
}

inline std::vector<std::pair<int, int>> read_queries(const std::string& filepath,
                                                      int offset) {
  std::ifstream fin(filepath);
  if (!fin.is_open()) {
    std::cerr << "ERROR: cannot open queries file: " << filepath << std::endl;
    std::exit(1);
  }

  std::vector<std::pair<int, int>> queries;
  std::string line;
  while (std::getline(fin, line)) {
    if (line.empty()) continue;
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream iss(line);
    int u, v;
    if (iss >> u >> v) {
      queries.emplace_back(u - offset, v - offset);
    }
  }
  return queries;
}

inline void write_queries_result(
    const std::string& filepath,
    const std::vector<std::vector<float>>& dist_matrix,
    const std::vector<std::pair<int, int>>& queries) {
  std::ofstream fout(filepath);
  for (const auto& [u, v] : queries) {
    float d = dist_matrix[u][v];
    fout << d << '\n';
  }
}
