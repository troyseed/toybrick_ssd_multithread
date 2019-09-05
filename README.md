# C_rknn_ssd_multithread
toybrick rknn multithread c demo frame work.
notice: This test is only can build on target devices(toybrick prod).

# pakages dependency
sudo dnf install -y cmake gcc gcc-c++ opencv opencv-devel rknn-api

# compile step
1. sudo dnf install -y cmake gcc gcc-c++ opencv opencv-devel rknn-api
2. mkdir build
3. cd build
4. cmake ..
5. make

# run demo
1. plug usb camera to toybrick prod board
2. ./ssd_demo