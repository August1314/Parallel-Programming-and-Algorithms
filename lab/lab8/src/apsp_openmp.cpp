#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "graph.hpp"

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0]
              << " <graph.csv> <queries.csv> <num_threads> [output.txt]"
              << std::endl;
    return 1;
  }

  std::string graph_path = argv[1];
  std::string queries_path = argv[2];
  int num_threads = std::atoi(argv[3]);
  std::string output_path = (argc >= 5) ? argv[4] : "";

  if (num_threads < 1) {
    std::cerr << "ERROR: num_threads must be >= 1, got " << num_threads
              << std::endl;
    return 1;
  }

#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif

  // Read graph
  auto [num_nodes, edges, offset] = read_graph(graph_path);
  auto adj = build_adjacency(num_nodes, edges);
  int num_edges = static_cast<int>(edges.size());

  // Read queries
  auto queries = read_queries(queries_path, offset);

  // Distance matrix: don't pre-fill - each thread will write its own row
  std::vector<std::vector<float>> dist_matrix(num_nodes);

  auto t_start = std::chrono::high_resolution_clock::now();

  // Parallel Dijkstra over source vertices
#pragma omp parallel for schedule(dynamic)
  for (int s = 0; s < num_nodes; ++s) {
    std::vector<float> dist(num_nodes, kInf);
    dist[s] = 0.0f;

    using State = std::pair<float, int>;
    std::priority_queue<State, std::vector<State>, std::greater<State>> pq;
    pq.emplace(0.0f, s);

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

    dist_matrix[s] = std::move(dist);
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double time_sec = std::chrono::duration<double>(t_end - t_start).count();
  double checksum = compute_checksum(dist_matrix);

  // Output key=value metrics
  std::cout << "experiment=apsp" << std::endl;
  std::cout << "backend=openmp" << std::endl;
  std::cout << "num_nodes=" << num_nodes << std::endl;
  std::cout << "num_edges=" << num_edges << std::endl;
  std::cout << "num_queries=" << queries.size() << std::endl;
  std::cout << "num_threads=" << num_threads << std::endl;
  std::cout << "time_sec=" << time_sec << std::endl;
  std::cout << "checksum=" << checksum << std::endl;

  // Write query results
  if (!output_path.empty()) {
    write_queries_result(output_path, dist_matrix, queries);
  }

  return 0;
}
