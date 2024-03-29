# Installation Guide

## CentOS 8 Guide

Based on this CentOS 7 guide: <https://www.vultr.com/docs/how-to-install-opencv-on-centos-7>

### Dependencies

#### System-level dependencies

```bash
dnf install python2-devel python3-devel gcc gcc-c++
```

#### OpenCV (from Source)

##### Step 1: Install dependencies for OpenCV

```bash
# Add Okay repo for libav-devel
dnf install http://repo.okay.com.mx/centos/8/x86_64/release/okay-release-1-3.el8.noarch.rpm
dnf install gtk3-devel gstreamer1-devel gstreamer1-plugins-base-devel libdc1394-devel libgphoto2-devel libav-devel cmake
```

##### Step 2: Download the OpenCV 3.3.0 archive

```bash
wget https://github.com/opencv/opencv/archive/3.2.0.zip
unzip 3.2.0.zip
```

##### Step 3: Compile and install OpenCV 3.3.0

```bash
cd opencv-3.2.0
mkdir -v build && cd build
cmake -D CMAKE_BUILD_TYPE=DEBUG -D CMAKE_INSTALL_PREFIX=/usr/local ..
make
make install
```

##### Step 4: Configure required variables

```bash
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig/
echo '/usr/local/lib/' >> /etc/ld.so.conf.d/opencv.conf
ldconfig
```

##### Step 5 (optional): Run tests

```bash
cd
git clone https://github.com/opencv/opencv_extra.git
export OPENCV_TEST_DATA_PATH=/root/opencv_extra/testdata

# You can find several test executables named with a name similar to `opencv_test_*`
# Try one just to test that the installation was successful
cd /root/opencv-3.3.0/build/bin
ls
./opencv_test_photo
```

## Core files

```bash
# Install rawtools
sudo python3 -m pip install git+https://github.com/Topp-Roots-Lab/python-rawtools
# Clone repo
sudo git clone https://github.com/Topp-Roots-Lab/3d-root-crown-analysis-pipeline.git /opt/3d-root-crown-analysis-pipeline/
# Install with Makefile (requires pip)
sudo make install
```

To wrap up the installation, separately install the two other components to run the pipeline:

1. [New3DTraitsForRPF](https://github.com/Topp-Roots-Lab/New3DTraitsForRPF/blob/standalone-kde-traits/INSTALL.md) (standalone edition)
1. [Gia3D](https://github.com/Topp-Roots-Lab/Gia3D#build)

# Ubuntu 18.04 Guide

## Dependencies

### System-level dependencies

```bash
apt install libopencv-core3.2 libopencv-imgcodecs3.2 python2.7 python2.7-dev python3 python3-dev gcc g++
```

## Core files

```bash
# Install rawtools
sudo python3 -m pip install git+https://github.com/Topp-Roots-Lab/python-rawtools
# Clone repo
sudo git clone https://github.com/Topp-Roots-Lab/3d-root-crown-analysis-pipeline.git /opt/3d-root-crown-analysis-pipeline/
# Install with Makefile (requires pip)
sudo make install
```

## Compile C++ Binaries (Optional)

```bash
g++ -o xrcap/lib/rootCrownSegmentation xrcap/rootCrownSegmentation.cpp -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lboost_system -lboost_filesystem -lboost_program_options -ltbb
```

To wrap up the installation, separately install the two other components to run the pipeline:

1. [New3DTraitsForRPF](https://github.com/Topp-Roots-Lab/New3DTraitsForRPF/blob/standalone-kde-traits/INSTALL.md) (standalone edition)
1. [Gia3D](https://github.com/Topp-Roots-Lab/Gia3D#build)

## Optional dependencies

The following software packages are required for quality control and converting 3-D model into compression versions (.CTM), assuming a headless environment.

- xvfb
- meshlab
- meshlabserver

```bash
dnf install xorg-x11-server-Xvfb
```
