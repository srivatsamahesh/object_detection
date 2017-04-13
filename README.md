# object_detection

Step 1: Download Ros Indigo by following below instructions
Note: download ros-indigo-desktop-full 

sudo apt-get install ros-indigo-desktop-full

http://wiki.ros.org/indigo/Installation/Ubuntu

Step 2: Install catkin_tools using apt-get

http://catkin-tools.readthedocs.io/en/latest/installing.html


Step 3: Create catkin_ws in your root folder
mkdir ~/catkin_ws/src
cd ~/catkin_ws
catkin init
catkin config --merge-devel
catkin config --extend /opt/ros/indigo
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release

Step 4: Clone this repository in catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/srivatsamahesh/object_detection.git
cd ~/catkin_ws
source ~/catkin_ws/devel/setup.bash
catkin build

Step 5: In new terminal
roscore

step 6: In new terminal 
cd ~/catkin_ws
source ~/catkin_ws/devel/setup.bash
roscd object_detection/model_files
rosrun pcl_ros pcd_to_pointcloud ck2_perpendicular.pcd 0.1 

Step 7: in new terminal 
cd ~/catkin_ws
source ~/catkin_ws/devel/setup.bash
rosrun object_detection object_recognition
