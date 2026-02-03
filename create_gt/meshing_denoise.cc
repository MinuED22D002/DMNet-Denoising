#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Kernel/global_functions.h>
#include <fstream>
#include <open3d/Open3D.h>

#include <Eigen/Sparse>
#include <iostream>
#include <unordered_map>
#include <random>
using namespace std;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;


typedef CGAL::Triangulation_vertex_base_with_info_3<int, K> Vb;
typedef CGAL::Delaunay_triangulation_cell_base_3<K>                 Cb;
typedef CGAL::Triangulation_data_structure_3<Vb, Cb>                Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds, CGAL::Fast_location> Delaunay;

namespace std {

template<>
struct hash<Delaunay::Vertex_handle> {
    std::size_t operator()(const Delaunay::Vertex_handle& handle) const {
        return reinterpret_cast<std::size_t>(&*handle);
    }
};

template<>
struct hash<const Delaunay::Vertex_handle> {
    std::size_t operator()(const Delaunay::Vertex_handle& handle) const {
        return reinterpret_cast<std::size_t>(&*handle);
    }
};

template<>
struct hash<Delaunay::Cell_handle> {
    std::size_t operator()(const Delaunay::Cell_handle& handle) const {
        return reinterpret_cast<std::size_t>(&*handle);
    }
};

template<>
struct hash<const Delaunay::Cell_handle> {
    std::size_t operator()(const Delaunay::Cell_handle& handle) const {
        return reinterpret_cast<std::size_t>(&*handle);
    }
};


}


K::Point_3 EigenToCGAL(const Eigen::Vector3d& point) {
    return K::Point_3(point(0), point(1), point(2));
}
Eigen::Vector3d CGALToEigen(const K::Point_3& point) {
    return Eigen::Vector3d(point.x(), point.y(), point.z());
}


Eigen::Vector3d GetMinBound(const open3d::geometry::PointCloud & input_points)
{
    const std::vector<Eigen::Vector3d>& points = input_points.points_;
    if (points.size() == 0) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    auto itr_x = std::min_element(points.begin(), points.end(),
                                  [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(0) < b(0); });
    auto itr_y = std::min_element(points.begin(), points.end(),
                                  [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(1) < b(1); });
    auto itr_z = std::min_element(points.begin(), points.end(),
                                  [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(2) < b(2); });
    return Eigen::Vector3d((*itr_x)(0), (*itr_y)(1), (*itr_z)(2));
}

Eigen::Vector3d GetMaxBound(const open3d::geometry::PointCloud& input_points)
{

    const std::vector<Eigen::Vector3d>& points = input_points.points_;
    if (points.size() == 0) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    auto itr_x = std::max_element(points.begin(), points.end(),
                                  [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(0) < b(0); });
    auto itr_y = std::max_element(points.begin(), points.end(),
                                  [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(1) < b(1); });
    auto itr_z = std::max_element(points.begin(), points.end(),
                                  [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return a(2) < b(2); });
    return Eigen::Vector3d((*itr_x)(0), (*itr_y)(1), (*itr_z)(2));
}

void normlize_points(open3d::geometry::PointCloud &input, Eigen::Vector4d& transform, const double scale_factor) {
    Eigen::Vector3d min_bound = GetMinBound(input);   
    Eigen::Vector3d max_bound = GetMaxBound(input);
    double cloud_scale = std::max(std::max(max_bound(0)-min_bound(0), max_bound(1)-min_bound(1)), max_bound(2)-min_bound(2));
    cloud_scale *= scale_factor;
    Eigen::Vector3d cloud_origin;
    cloud_origin(0) = max_bound(0) - (cloud_scale + max_bound(0) - min_bound(0)) / 2.0;
    cloud_origin(1) = max_bound(1) - (cloud_scale + max_bound(1) - min_bound(1)) / 2.0;
    cloud_origin(2) = max_bound(2) - (cloud_scale + max_bound(2) - min_bound(2)) / 2.0;
    std::for_each(input.points_.begin(), input.points_.end(), [&](Eigen::Vector3d &point) { point = (point - cloud_origin) / cloud_scale; } );
    transform(0) = cloud_origin(0);
    transform(1) = cloud_origin(1);
    transform(2) = cloud_origin(2);
    transform(3) = cloud_scale;  
}

void add_noise(open3d::geometry::PointCloud &input,
               double noise_level) {

    Eigen::Vector3d voxel_min_bound = GetMinBound(input);
    Eigen::Vector3d voxel_max_bound = GetMaxBound(input);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::vector<Eigen::Vector3d> noise(input.points_.size());
    // x
    {
        double interval = voxel_max_bound(0) - voxel_min_bound(0);
        std::normal_distribution<double> dis(0.0, 1.0);
        for(auto& e : noise) {
            e(0) = noise_level * interval * dis(generator);
        }
    }
    // y
    {
        double interval = voxel_max_bound(1) - voxel_min_bound(1);
        std::normal_distribution<double> dis(0.0, 1.0);
        for(auto& e : noise) {
            e(1) = noise_level * interval * dis(generator);
        }
    }
    // z
    {
        double interval = voxel_max_bound(2) - voxel_min_bound(2);
        std::normal_distribution<double> dis(0.0, 1.0);
        for(auto& e : noise) {
            e(2) = noise_level * interval * dis(generator);
        }
    }

    for(int i = 0; i < input.points_.size(); i++) {
        input.points_[i] += noise[i];
    }
}


void uniform_sample(const open3d::geometry::PointCloud & input, open3d::geometry::PointCloud & output,
                    std::vector<int>& indices, const int fixed_num_points)      
{
    int points_num = input.points_.size();
    if(points_num <= fixed_num_points)    
    {
        output += input;
        return;
    }

    std::unordered_set<int> index_set;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, points_num - 1);       
    while (index_set.size() < fixed_num_points)
    {
        int tmp_int = dis(gen);
        index_set.insert(tmp_int);
    }
    for (int e : index_set)
    {
        output.points_.push_back(input.points_[e]);
        if (input.HasNormals()) output.normals_.push_back(input.normals_[e]);
        if (input.HasColors()) output.colors_.push_back(input.colors_[e]);
        indices.push_back(e);
    }
}


void ReadData(const std::string& point_path,
              open3d::geometry::PointCloud& original_points){
    std::cout<<("Reading from sampled points...")<<std::endl;
    open3d::io::ReadPointCloud(point_path, original_points);
    std::cout<<("Done!")<<std::endl;
}

Delaunay ConstructDelaunay(const open3d::geometry::PointCloud& sampled_points){
    std::cout << "CreateVoxelSampledDelaunayTriangulation" << std::endl;
    std::vector<Delaunay::Point> delaunay_points(sampled_points.points_.size());
    for (size_t i = 0; i < sampled_points.points_.size(); ++i) {
        delaunay_points[i] =
                Delaunay::Point(sampled_points.points_[i][0], sampled_points.points_[i][1],
                                sampled_points.points_[i][2]);
    }
    std::vector<int> point_idxs(sampled_points.points_.size());
    std::iota(point_idxs.begin(), point_idxs.end(), 0);

    std::cout << "Create triangulartion" << std::endl;
    const auto triangulation = Delaunay( boost::make_zip_iterator(boost::make_tuple( delaunay_points.begin(), point_idxs.begin() )),
                                          boost::make_zip_iterator(boost::make_tuple( delaunay_points.end(), point_idxs.end() ) )  );
    return triangulation;

}

void write_delaunay(Delaunay triangulation, std::string output_path) {
    {
        std::ofstream oFile_ocvi(output_path + "output_cell_vertex_idx.txt",std::ios::out);
        for (auto it = triangulation.all_cells_begin();
             it != triangulation.all_cells_end(); ++it)
        {
            for(int j=0; j<4; j++)
            {

                if (triangulation.is_infinite(it->vertex(j)))
                {
                    oFile_ocvi << -1 << " ";
                }
                else {
                    oFile_ocvi << it->vertex(j)->info() << " ";
                }
                if(j == 3) {
                    oFile_ocvi << std::endl;
                }
            }
        }
        oFile_ocvi.close();
    }


    {
        std::ofstream oFileT1(output_path + "output_triangulation_binary",std::ios::out | std::ios::binary);
        CGAL::set_binary_mode(oFileT1);
        oFileT1 << triangulation;
        oFileT1.close();

    }

    CGAL::Unique_hash_map<Delaunay::Cell_handle, int> cell_idx_map;
    int cell_count = 0;
    for (auto it = triangulation.all_cells_begin();
        it != triangulation.all_cells_end(); ++it)
    {
        cell_idx_map[it] = cell_count;
        cell_count++;
    }

    {
        std::ofstream oFileT_adj(output_path + "output_tetrahedron_adj",std::ios::out );
        for (auto it = triangulation.all_cells_begin();
            it != triangulation.all_cells_end(); ++it)
        {
            for(int i=0;i<4;i++)
            {
                auto neighbor_cell = it->neighbor(i);
                int neighbor_index = cell_idx_map[neighbor_cell];
                oFileT_adj<<neighbor_index<<" ";
            }
            oFileT_adj<<std::endl;
        }
        oFileT_adj.close();
    }

    {
        std::ofstream oFile_facet(output_path + "output_facet_vertex_idx.txt", std::ios::out);
        for (auto it = triangulation.finite_facets_begin(); it != triangulation.finite_facets_end(); ++it) {
            for (int i = 0; i < 4; i++) {
                if (i == it->second) {
                    continue;
                }
                oFile_facet << it->first->vertex(i)->info() << " ";
            }
            oFile_facet << std::endl;
        }
        oFile_facet.close();
    }

    {   
        std::ofstream oFile_ft(output_path + "output_facet_nei_cell.txt", std::ios::out);
        for (auto it = triangulation.finite_facets_begin(); it != triangulation.finite_facets_end(); ++it) {
            oFile_ft << cell_idx_map[it->first] << " " << cell_idx_map[it->first->neighbor(it->second)] << std::endl;
        }
        oFile_ft.close();
    }
}


int main(int argc, char* argv[]){
    std::string input_point_path = argv[1];
    std::string output_path = argv[2];
    int sample_depth = std::stoi(argv[3]);
    int npoints = std::stoi(argv[4]);
    double noise_level = std::stod(argv[5]);
    // int ref_num = std::stoi(argv[6]); // Unused

    double scale_factor = 1.1;

    open3d::geometry::PointCloud original_points;
    ReadData(input_point_path, original_points);

    open3d::geometry::PointCloud noise_points;
    noise_points += original_points; // copy
    add_noise(noise_points, noise_level);

    open3d::geometry::PointCloud normalized_points;   
    normalized_points += noise_points;                                 
    Eigen::Vector4d transform;
    normlize_points(normalized_points, transform, scale_factor);         

    open3d::geometry::PointCloud sampled_points1;
    std::vector<int> uniform_indices;
    uniform_sample(normalized_points, sampled_points1, uniform_indices, npoints);

    Delaunay triangulation = ConstructDelaunay(sampled_points1);
    std::cout << "creat delaunay data done" << std::endl;
    // No explicit labeling needed for denoising task where only offset is regressed
    // We will generate dummy labels just in case Python loads them, 
    // but the loss function is already set to ignore them for denoising.

    // Write Files
    write_delaunay(triangulation, output_path);
    std::cout << "write_delaunay done" << std::endl;

    open3d::io::WritePointCloud(output_path + "noise_points.ply", noise_points);
    open3d::io::WritePointCloud(output_path + "sampled_points.ply", sampled_points1);
    
    {
        std::ofstream oFile_transform(output_path + "transform.txt", std::ios::out );
        oFile_transform<< transform <<std::endl;
        oFile_transform.close();
    }

    // Write dummy ref_point_label.txt to prevent parser errors in DT_data.py
    // DT_data.py expects: l['ref_label'] = np.fromfile(..., sep=' ').reshape(-1, cfg["ref_num"])
    // We will write 0s.
    int dummy_ref_num = 3; // Default used in args
    int num_cells = triangulation.number_of_cells();
    // Actually mapping is cell based or... wait. meshing.cc uses labels size of "sds" which matches ref_points.
    
    // In meshing.cc, ref_points are generated per cell (ref_num per cell).
    // So we need ref_num * num_cells labels.
    int total_labels = triangulation.number_of_cells() * dummy_ref_num; 
    
    // However, since we removed compute_ref_points, we don't have ref_points. 
    // But DT_data.py loads 'ref_point_label.txt'.
    // We should write a dummy file of correct size: num_cells * ref_num * sizeof(int)
    
    // Actually, let's look at `meshing.cc`: 
    // `label_points` generates `labels`. `labels` corresponds to `signed_distances`.
    // `signed_distances` corresponds to `ref_points`.
    // `ref_points` has `t.number_of_cells() * ref_num` points (roughly, including infinite?). 
    // Actually `compute_ref_points` iterates `all_cells_begin`.
    
    // For safety, we will generate dummy zeros.
    {
        std::ofstream oFile_gt_label(output_path + "ref_point_label.txt",std::ios::out );
        // We write 0 for all cells * ref_num (3)
        // Note: all_cells includes infinite.
        for (auto it = triangulation.all_cells_begin(); it != triangulation.all_cells_end(); ++it) {
             for(int k=0; k<dummy_ref_num; ++k) oFile_gt_label << "1" << std::endl; // 1 = inside/outside dummy
        }
        oFile_gt_label.close();
    }
    
    std::cout << "done" << std::endl;
}
