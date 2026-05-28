#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "graph.hpp"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <graph.csv> <queries.csv> [output.txt]"
              << std::endl;
    return 1;
  }

  std::string graph_path = argv[1];
  std::string queries_path = argv[2];
  std::string output_path = (argc >= 4) ? argv[3] : "";

  // Read graph
  auto [num_nodes, edges, offset] = read_graph(graph_path);
  auto adj = build_adjacency(num_nodes, edges);
  int num_edges = static_cast<int>(edges.size());

  // Read queries
  auto queries = read_queries(queries_path, offset);

  // Run APSP: serial Dijkstra from each source
  std::vector<std::vector<float>> dist_matrix(num_nodes);
  auto t_start = std::chrono::high_resolution_clock::now();

  for (int s = 0; s < num_nodes; ++s) {
    dist_matrix[s] = dijkstra(s, adj);
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double time_sec = std::chrono::duration<double>(t_end - t_start).count();
  double checksum = compute_checksum(dist_matrix);

  // Output key=value metrics
  std::cout << "experiment=apsp" << std::endl;
  std::cout << "backend=serial" << std::endl;
  std::cout << "num_nodes=" << num_nodes << std::endl;
  std::cout << "num_edges=" << num_edges << std::endl;
  std::cout << "num_queries=" << queries.size() << std::endl;
  std::cout << "num_threads=1" << std::endl;
  std::cout << "time_sec=" << time_sec << std::endl;
  std::cout << "checksum=" << checksum << std::endl;

  // Write query results
  if (!output_path.empty()) {
    write_queries_result(output_path, dist_matrix, queries);
  }

  return 0;
}
