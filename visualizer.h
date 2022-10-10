#ifndef _VISUALIZER_H_
#define _VISUALIZER_H_

#include <iostream>
#include <vector>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/visualization/cloud_viewer.h>

#include "metric_calculator.h"
#include "dataloader.h"


class Visualizer: public Dataloader {
private:

public:
    Visualizer();
    void load_pointcloud(pcl::visualization::PCLVisualizer::Ptr &viewer, int frame_idx);
    std::vector<std::vector<float>> draw_cube(pcl::visualization::PCLVisualizer::Ptr &viewer, std::string file, int type);
    void cloud_visualization();
};

#endif
