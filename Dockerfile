# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

FROM ros:humble

SHELL ["/bin/bash", "-c"]

# dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y git \
                       nano \
                       vim \
                       python3-pip \
                       python3-tk \
                       python3-numpy \
                       python3-opencv \
                       python3-scipy \
                       python3-skimage \
                       python3-matplotlib \
                       libeigen3-dev \
                       tmux \
                       ros-humble-rviz2 \
                       ros-humble-robot-state-publisher \

                       ros-humble-tf-transformations \
                       ros-humble-nav2-map-server \
                       ros-humble-nav2-lifecycle-manager \
                       ros-humble-nav2-msgs
                       
RUN apt-get -y dist-upgrade
RUN pip3 install transforms3d casadi pandas do_mpc "cython<3"

# f1tenth gym
RUN git clone https://github.com/f1tenth/f1tenth_gym
RUN cd f1tenth_gym && \
    pip3 install -e .


# ...............................................................
# Make ROS 2 Humble available in interactive Bash shells by default.
RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo 'source /opt/ros/humble/setup.bash' >> /etc/bash.bashrc
# .................................................................




# ros2 gym bridge and additional packages
RUN mkdir -p sim_ws/src

# particle_filter (MIT/F1TENTH, humble-devel branch)
RUN git clone --branch humble-devel https://github.com/f1tenth/particle_filter /sim_ws/src/particle_filter

# range_libc — f1tenth maintained fork, humble-devel branch (Python 3 compatible)
RUN apt-get install -y libboost-all-dev && \
    git clone --branch humble-devel https://github.com/f1tenth/range_libc /tmp/range_libc && \
    cd /tmp/range_libc/pywrapper && \
    python3 setup.py install

COPY . /sim_ws/src/f1tenth_gym_ros
COPY mpc_f1tenth /sim_ws/src/mpc_f1tenth
RUN source /opt/ros/humble/setup.bash && \
    cd sim_ws/ && \
    apt-get update --fix-missing && \
    rosdep install -i --from-path src --rosdistro humble -y && \
    colcon build

WORKDIR '/sim_ws'
ENTRYPOINT ["/bin/bash"]

