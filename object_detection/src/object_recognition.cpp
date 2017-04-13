// object recognition node to compare model and scene and publish keypoints from model and scene
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>


#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/filters/filter.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>



typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

ros::Publisher pub_model;
ros::Publisher pub_scene;

//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (false);
bool use_cloud_resolution_ (false);
bool use_hough_ (true);
float model_ss_ (0.1f);
float scene_ss_ (0.1f);
float rf_rad_ (0.15f);
float descr_rad_ (0.02f);
float cg_size_ (0.01f);
float cg_thresh_ (5.0f);


double
computeCloudResolution (const pcl::PointCloud<PointType>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<PointType> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i)
  {
    if (! pcl_isfinite ((*cloud)[i].x))
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}

void
cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input){

	pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
	pcl::PointCloud<PointType>::Ptr model1 (new pcl::PointCloud<PointType> ());
  	pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
  	pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
  	pcl::PointCloud<PointType>::Ptr scene1 (new pcl::PointCloud<PointType> ());
  	pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
  	pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
  	pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
  	pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
  	pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());

//load pcd file that holds the "model"
	std::string   model_path=ros::package::getPath("object_detection");
  	model_path+="/model_files/ck2_perpendicular.pcd";
  	if (pcl::io::loadPCDFile (model_path, *model1) < 0)
  	{
    		std::cout << "Error loading model cloud." << std::endl;
    		return;
  	}

  	model1->header.frame_id= "base_link";
	ros::Time time_st = ros::Time::now ();
	model1->header.stamp = time_st.toNSec()/1e3;


	std::vector<int> indices;
      	pcl::removeNaNFromPointCloud(*model1,*model, indices);
	ROS_INFO("Size of original and new pointcloud is %d and %d",model1->size(), model->size());

//create subscribed topic to pcl usable format
  	pcl::fromROSMsg(*input, *scene1);

	std::vector<int> indices1;
      	pcl::removeNaNFromPointCloud(*scene1,*scene, indices);

//Setup resolution invariance between scene and model
 	if (use_cloud_resolution_)
  	{
    		float resolution = static_cast<float> (computeCloudResolution (model));
    	if (resolution != 0.0f)
    	{
      		model_ss_   *= resolution;
      		scene_ss_   *= resolution;
      		rf_rad_     *= resolution;
      		descr_rad_  *= resolution;
      		cg_size_    *= resolution;
    	}

		ROS_INFO("Model resolution: %f", resolution);
		ROS_INFO("Model sampling size: %f", model_ss_);
		ROS_INFO("Scene sampling size: %f", scene_ss_);
		ROS_INFO("LRF support radius: %f", rf_rad_);
		ROS_INFO("SHOT descriptor radius: %f", descr_rad_);
		ROS_INFO("Clustering bin size: %f", cg_size_);
	}


//  Compute Normals

  	pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  	norm_est.setKSearch (10);
  	norm_est.setInputCloud (model);
  	norm_est.compute (*model_normals);
  	ROS_INFO("Model normals computed");

  	norm_est.setInputCloud (scene);
  	norm_est.compute (*scene_normals);
  	ROS_INFO("Scene normals computed");



//  Downsample Clouds to Extract keypoints

	pcl::PointCloud<int> sampled_indices;

  	pcl::UniformSampling<PointType> uniform_sampling;
  	uniform_sampling.setInputCloud (model);
  	uniform_sampling.setRadiusSearch (model_ss_);
  	uniform_sampling.compute (sampled_indices);
  	pcl::copyPointCloud (*model, sampled_indices.points, *model_keypoints);
  	ROS_INFO("Model total points: %d ; Selected Keypoints: %d ;",model->size (), model_keypoints->size ());

  	uniform_sampling.setInputCloud (scene);
  	uniform_sampling.setRadiusSearch (scene_ss_);
  	uniform_sampling.compute (sampled_indices);
  	pcl::copyPointCloud (*scene, sampled_indices.points, *scene_keypoints);
  	ROS_INFO("Scene total points: %d ; Selected Keypoints: %d ;",scene->size (), scene_keypoints->size ());


// generate descriptors for keypoints
  	pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
  	descr_est.setRadiusSearch (descr_rad_);
	ROS_INFO("1");

  	descr_est.setInputCloud (model_keypoints);
  	descr_est.setInputNormals (model_normals);
  	descr_est.setSearchSurface (model);
  	descr_est.compute (*model_descriptors);
	ROS_INFO("2");
	

  	descr_est.setInputCloud (scene_keypoints);
  	descr_est.setInputNormals (scene_normals);
  	descr_est.setSearchSurface (scene);
  	descr_est.compute (*scene_descriptors);
	ROS_INFO("3");

//  Find Model-Scene Correspondences with KdTree
  	pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

  	pcl::KdTreeFLANN<DescriptorType> match_search;
  	match_search.setInputCloud (model_descriptors);

  	//  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
  	for (size_t i = 0; i < scene_descriptors->size (); ++i)
  	{
    		std::vector<int> neigh_indices (1);
    		std::vector<float> neigh_sqr_dists (1);
    		if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
    		{
      			continue;
    		}
    	
		int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
    		if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
    		{
      			pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      			model_scene_corrs->push_back (corr);
    		}
  	}
  	ROS_INFO("Correspondences found : %d ;",model_scene_corrs->size());


// find correspondence grouping
//  Actual Clustering
  
  	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
  	std::vector<pcl::Correspondences> clustered_corrs;

  	//  Using Hough3D
  	if (use_hough_)
  	{

    	//  Compute (Keypoints) Reference Frames only for Hough

    		pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
    		pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

    		pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
    		rf_est.setFindHoles (true);
    		rf_est.setRadiusSearch (rf_rad_);
		ROS_INFO("4");

    		rf_est.setInputCloud (model_keypoints);
    		rf_est.setInputNormals (model_normals);
    		rf_est.setSearchSurface (model);
//		for (int i=0;i< model_keypoints->size();i++){
//			ROS_INFO("keypoint value at i= %d is %d",i,model_keypoints->points(i));
//		}

		ROS_INFO("model size : %d ", model->size());
		ROS_INFO("model_keypoint size : %d ", model_keypoints->size());
    		rf_est.compute (*model_rf);
		ROS_INFO("5");

    		rf_est.setInputCloud (scene_keypoints);
    		rf_est.setInputNormals (scene_normals);
   		rf_est.setSearchSurface (scene);
    		rf_est.compute (*scene_rf);
		ROS_INFO("6");

    	//  Clustering
    		pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
    		clusterer.setHoughBinSize (cg_size_);
    		clusterer.setHoughThreshold (cg_thresh_);
    		clusterer.setUseInterpolation (true);
    		clusterer.setUseDistanceWeight (false);

    		clusterer.setInputCloud (model_keypoints);
    		clusterer.setInputRf (model_rf);
    		clusterer.setSceneCloud (scene_keypoints);
    		clusterer.setSceneRf (scene_rf);
    		clusterer.setModelSceneCorrespondences (model_scene_corrs);

    	//clusterer.cluster (clustered_corrs);
    		clusterer.recognize (rototranslations, clustered_corrs);
  	}
  	else // Using GeometricConsistency
  	{
    		pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
    		gc_clusterer.setGCSize (cg_size_);
    		gc_clusterer.setGCThreshold (cg_thresh_);
		ROS_INFO("7");

    		gc_clusterer.setInputCloud (model_keypoints);
		ROS_INFO("8");
    		gc_clusterer.setSceneCloud (scene_keypoints);
		ROS_INFO("9");
    		gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);
		ROS_INFO("10");

    	//gc_clusterer.cluster (clustered_corrs);
    		gc_clusterer.recognize(rototranslations, clustered_corrs);
		ROS_INFO("11");

  	}

// rotation and translation matrices for the model instances found
//  Output results

  	ROS_INFO("Model instances found : %d ;",rototranslations.size());
  	for (size_t i = 0; i < rototranslations.size (); ++i)
  	{
  		ROS_INFO("Instances found : %d ;",i+1);
  		ROS_INFO("Correspondences belonging to this instance : %d ;",clustered_corrs[i].size());

		// Print the rotation matrix and translation vector
    		Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
    		Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

  		ROS_INFO("Rotation matrix: %0.2f,0.2f,0.2f ; 0.2f,0.2f,0.2f; 0.2f,0.2f,0.2f;",rotation(0,0),rotation(0,1),rotation(0,2),rotation(1,0),rotation(1,1),rotation(1,2),rotation(2,0),rotation(2,1),rotation(2,2));
  		ROS_INFO("Translation vector: %0.2f,0.2f,0.2f ;",translation(0),translation(1),translation(2));

  	}

// publish keypoints from scene and model
        sensor_msgs::PointCloud2 model_keypoints_ros;
        sensor_msgs::PointCloud2 scene_keypoints_ros;
	pcl::toROSMsg(*model_keypoints,model_keypoints_ros);
	pcl::toROSMsg(*scene_keypoints,scene_keypoints_ros);

	pub_model.publish(model_keypoints_ros);
	pub_scene.publish(scene_keypoints_ros);

}


//main
int
main(int argc, char** argv){
	ros::init(argc,argv, "object_recognition");
	ros::NodeHandle nh;

	//subscribe to scene and initiate call back
	ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_pcd",1,cloud_cb);

	//publish keypoints
	pub_model = nh.advertise<sensor_msgs::PointCloud2>("/model_keypoint_output",1);
	pub_scene = nh.advertise<sensor_msgs::PointCloud2>("/scene_keypoint_output",1);

	ros::spin ();

}

