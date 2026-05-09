// radius_search_pcd_demo.cpp - PCD radius-search demo for likd-tree

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>

#include "likd_tree.hpp"

using PointType = pcl::PointXYZ;
using Clock = std::chrono::high_resolution_clock;

namespace {

void printUsage(const char* program) {
  std::cerr << "Usage:\n"
            << "  " << program
            << " <map.pcd> <qx> <qy> <qz> <radius> [--no-vis]\n\n"
            << "Example:\n"
            << "  " << program << " map.pcd 10.0 -5.0 2.0 3.0\n";
}

bool parseFloat(const char* text, float& value) {
  try {
    size_t parsed = 0;
    value = std::stof(text, &parsed);
    return parsed == std::string(text).size();
  } catch (const std::exception&) {
    return false;
  }
}

bool isFinite(const PointType& point) {
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.z);
}

float sqrDist(const PointType& a, const PointType& b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

void bruteForceRadiusSearch(const PointVector<PointType>& points,
                            const PointType& query, float radius,
                            PointVector<PointType>& results,
                            std::vector<float>& distances) {
  results.clear();
  distances.clear();

  if (radius < 0.0f) {
    return;
  }

  const float radius2 = radius * radius;
  for (const auto& point : points) {
    const float dist2 = sqrDist(point, query);
    if (dist2 <= radius2) {
      results.push_back(point);
      distances.push_back(std::sqrt(dist2));
    }
  }

  std::vector<size_t> indices(results.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](size_t lhs, size_t rhs) {
    if (distances[lhs] != distances[rhs]) {
      return distances[lhs] < distances[rhs];
    }
    const PointType& lhs_point = results[lhs];
    const PointType& rhs_point = results[rhs];
    if (lhs_point.x != rhs_point.x) {
      return lhs_point.x < rhs_point.x;
    }
    if (lhs_point.y != rhs_point.y) {
      return lhs_point.y < rhs_point.y;
    }
    return lhs_point.z < rhs_point.z;
  });

  PointVector<PointType> sorted_results;
  sorted_results.reserve(results.size());
  std::vector<float> sorted_distances;
  sorted_distances.reserve(distances.size());

  for (size_t index : indices) {
    sorted_results.push_back(results[index]);
    sorted_distances.push_back(distances[index]);
  }

  results.swap(sorted_results);
  distances.swap(sorted_distances);
}

bool sameRadiusResults(const PointVector<PointType>& tree_results,
                       const std::vector<float>& tree_distances,
                       const PointVector<PointType>& brute_results,
                       const std::vector<float>& brute_distances) {
  if (tree_results.size() != brute_results.size() ||
      tree_distances.size() != brute_distances.size()) {
    return false;
  }

  constexpr float kTolerance = 1e-5f;
  for (size_t i = 0; i < tree_results.size(); ++i) {
    if (std::fabs(tree_distances[i] - brute_distances[i]) > kTolerance) {
      return false;
    }
    if (sqrDist(tree_results[i], brute_results[i]) > kTolerance * kTolerance) {
      return false;
    }
  }

  return true;
}

double elapsedMs(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void colorize(const PointVector<PointType>& points,
              pcl::PointCloud<pcl::PointXYZRGB>& colored,
              unsigned char red, unsigned char green, unsigned char blue) {
  colored.clear();
  colored.reserve(points.size());

  for (const auto& point : points) {
    pcl::PointXYZRGB colored_point;
    colored_point.x = point.x;
    colored_point.y = point.y;
    colored_point.z = point.z;
    colored_point.r = red;
    colored_point.g = green;
    colored_point.b = blue;
    colored.push_back(colored_point);
  }
}

void visualizeRadiusSearch(const pcl::PointCloud<PointType>::ConstPtr& cloud,
                           const PointType& query,
                           const PointVector<PointType>& radius_results,
                           float radius) {
  pcl::visualization::PCLVisualizer viewer("likd-tree Radius Search");
  viewer.setBackgroundColor(0.03, 0.03, 0.03);
  viewer.addCoordinateSystem(1.0);

  pcl::visualization::PointCloudColorHandlerCustom<PointType> cloud_color(
      cloud, 170, 170, 170);
  viewer.addPointCloud<PointType>(cloud, cloud_color, "cloud");
  viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr radius_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  colorize(radius_results, *radius_cloud, 255, 0, 0);
  viewer.addPointCloud<pcl::PointXYZRGB>(radius_cloud, "radius_results");
  viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "radius_results");

  pcl::PointCloud<PointType>::Ptr query_cloud(new pcl::PointCloud<PointType>);
  query_cloud->push_back(query);
  pcl::visualization::PointCloudColorHandlerCustom<PointType> query_color(
      query_cloud, 0, 255, 0);
  viewer.addPointCloud<PointType>(query_cloud, query_color, "query");
  viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 12, "query");

  viewer.addSphere(query, radius, 0.0, 1.0, 0.0, "radius_sphere");
  viewer.setShapeRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
      pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
      "radius_sphere");
  viewer.setShapeRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_OPACITY, 0.35, "radius_sphere");
  viewer.resetCamera();

  while (!viewer.wasStopped()) {
    viewer.spinOnce(16);
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6 && argc != 7) {
    printUsage(argv[0]);
    return 1;
  }

  bool visualize = true;
  if (argc == 7) {
    if (std::string(argv[6]) != "--no-vis") {
      std::cerr << "Error: unknown option: " << argv[6] << '\n';
      printUsage(argv[0]);
      return 1;
    }
    visualize = false;
  }

  PointType query;
  float radius = 0.0f;
  if (!parseFloat(argv[2], query.x) || !parseFloat(argv[3], query.y) ||
      !parseFloat(argv[4], query.z) || !parseFloat(argv[5], radius)) {
    std::cerr << "Error: qx, qy, qz and radius must be valid floating-point "
                 "values.\n";
    printUsage(argv[0]);
    return 1;
  }
  if (!isFinite(query)) {
    std::cerr << "Error: query point coordinates must be finite values.\n";
    return 1;
  }
  if (!std::isfinite(radius) || radius < 0.0f) {
    std::cerr << "Error: radius must be a finite non-negative value.\n";
    return 1;
  }

  pcl::PointCloud<PointType>::Ptr input_cloud(new pcl::PointCloud<PointType>);
  if (pcl::io::loadPCDFile<PointType>(argv[1], *input_cloud) < 0) {
    std::cerr << "Error: failed to read PCD file: " << argv[1] << '\n';
    return 1;
  }

  PointVector<PointType> points;
  points.reserve(input_cloud->size());

  pcl::PointCloud<PointType>::Ptr valid_cloud(new pcl::PointCloud<PointType>);
  valid_cloud->reserve(input_cloud->size());

  for (const auto& point : input_cloud->points) {
    if (!isFinite(point)) {
      continue;
    }
    points.push_back(point);
    valid_cloud->push_back(point);
  }

  if (points.empty()) {
    std::cerr << "Error: PCD file does not contain any finite XYZ points.\n";
    return 1;
  }

  KDTree<PointType> tree;

  const auto build_start = Clock::now();
  tree.build(points);
  const auto build_end = Clock::now();

  PointVector<PointType> radius_results;
  std::vector<float> radius_distances;

  const auto query_start = Clock::now();
  tree.radiusSearch(query, radius, radius_results, radius_distances);
  const auto query_end = Clock::now();

  PointVector<PointType> brute_results;
  std::vector<float> brute_distances;
  bruteForceRadiusSearch(points, query, radius, brute_results,
                         brute_distances);
  const bool match = sameRadiusResults(radius_results, radius_distances,
                                       brute_results, brute_distances);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Loaded points: " << input_cloud->size() << '\n';
  std::cout << "Finite points used: " << points.size() << '\n';
  std::cout << "Build time: " << elapsedMs(build_start, build_end) << " ms\n";
  std::cout << "Radius search time: " << elapsedMs(query_start, query_end)
            << " ms\n";
  std::cout << "Query: (" << query.x << ", " << query.y << ", " << query.z
            << ")\n";
  std::cout << "Radius: " << radius << '\n';
  std::cout << "Radius search points: " << radius_results.size() << '\n';
  std::cout << "Brute-force points: " << brute_results.size() << '\n';
  if (!radius_distances.empty()) {
    std::cout << "Nearest radius result distance: " << radius_distances.front()
              << '\n';
    std::cout << "Farthest radius result distance: " << radius_distances.back()
              << '\n';
  }
  std::cout << "Brute-force check: " << (match ? "MATCH" : "MISMATCH")
            << '\n';

  if (visualize) {
    visualizeRadiusSearch(valid_cloud, query, radius_results, radius);
  }
  return match ? 0 : 2;
}
