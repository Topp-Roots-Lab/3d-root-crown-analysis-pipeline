# Installation Guide for 3D-RCAP

## CentOS 8 Guide

Base on this CentOS 7 guide: https://www.vultr.com/docs/how-to-install-opencv-on-centos-7

### Dependencies

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