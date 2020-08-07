
Installation Guide for 3D-RCAP
------------------------------

CentOS 8 Guide
^^^^^^^^^^^^^^

Base on this CentOS 7 guide: https://www.vultr.com/docs/how-to-install-opencv-on-centos-7

Dependencies
~~~~~~~~~~~~

System-level dependencies
"""""""""""""""""""""""""

.. code-block:: bash

   dnf install python2-devel python3-devel gcc gcc-c++

OpenCV (from Source)
""""""""""""""""""""

Step 1: Install dependencies for OpenCV
#######################################

.. code-block:: bash

   # Add Okay repo for libav-devel
   dnf install http://repo.okay.com.mx/centos/8/x86_64/release/okay-release-1-3.el8.noarch.rpm
   dnf install gtk3-devel gstreamer1-devel gstreamer1-plugins-base-devel libdc1394-devel libgphoto2-devel libav-devel cmake

Step 2: Download the OpenCV 3.3.0 archive
#########################################

.. code-block:: bash

   wget https://github.com/opencv/opencv/archive/3.2.0.zip
   unzip 3.2.0.zip

Step 3: Compile and install OpenCV 3.3.0
########################################

.. code-block:: bash

   cd opencv-3.2.0
   mkdir -v build && cd build
   cmake -D CMAKE_BUILD_TYPE=DEBUG -D CMAKE_INSTALL_PREFIX=/usr/local ..
   make
   make install

Step 4: Configure required variables
####################################

.. code-block:: bash

   export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig/
   echo '/usr/local/lib/' >> /etc/ld.so.conf.d/opencv.conf
   ldconfig

Step 5 (optional): Run tests
############################

.. code-block:: bash

   cd
   git clone https://github.com/opencv/opencv_extra.git
   export OPENCV_TEST_DATA_PATH=/root/opencv_extra/testdata

   # You can find several test executables named with a name similar to `opencv_test_*`
   # Try one just to test that the installation was successful
   cd /root/opencv-3.3.0/build/bin
   ls
   ./opencv_test_photo

Core files
^^^^^^^^^^

.. code-block:: bash

   # Clone repo
   git clone https://github.com/Topp-Roots-Lab/3d-root-crown-analysis-pipeline.git /opt/3drcap/
   # Create symlinks in /usr/local/bin
   find /opt/3drcap/rcap -type f | while read f; do ln -sv "$f" "/usr/local/bin/$(basename "${f%.*}")"; done
   # Install Python modules for versions 2 and 3
   pip install -r /opt/3drcap/requirements.txt
   pip2 install -r /opt/3drcap/requirements.txt

Ubuntu 18.04 Guide
------------------

Dependencies
^^^^^^^^^^^^

System-level dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   apt install libopencv-core3.2 libopencv-imgcodecs3.2 python2.7 python2.7-dev python3 python3-dev gcc g++

Core files
^^^^^^^^^^

.. code-block:: bash

   # Clone repo
   git clone https://github.com/Topp-Roots-Lab/3d-root-crown-analysis-pipeline.git /opt/3drcap/
   # Create symlinks in /usr/local/bin
   find /opt/3drcap/src -type f | while read f; do ln -sv "$f" "/usr/local/bin/$(basename "${f%.*}")"; done
   # Install Python modules for versions 2 and 3
   pip install -r /opt/3drcap/requirements.txt
   pip2 install -r /opt/3drcap/requirements.txt

Compile C++ Binaries
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   g++ -o xrcap/lib/rootCrownSegmentation xrcap/rootCrownSegmentation.cpp -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lboost_system -lboost_filesystem -lboost_program_options
