FROM nvcr.io/nvidia/tensorflow:21.10-tf1-py3
ARG PYTHON_VERSION=3.8
RUN python -m pip uninstall tensorflow -y
RUN python -m pip uninstall scipy -y
RUN python -m pip uninstall numpy -y
RUN python -m pip install numpy==1.16.4
RUN python -m pip install scipy==1.3.1
RUN python -m pip install scikit-image==0.15.0
RUN python -m pip install tensorflow>=1.14.0
RUN python -m pip install absl-py==0.7.1 plyfile==0.7.1 trimesh==3.2.12
RUN apt-get update
RUN apt-get install -y build-essential zsh tmux wget
RUN apt-get install -y libgl1-mesa-glx libusb-1.0-0-dev
RUN wget https://raw.githubusercontent.com/fwilliams/configs/master/zsh/zshrc
RUN sed -i 's/francis/${USER}/' zshrc
RUN sed -i 's/required/sufficient/' /etc/pam.d/chsh
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN rm ${HOME}/.zshrc
RUN mv zshrc ${HOME}/.zshrc
RUN source ${HOME}/.zshrc
RUN chsh -s $(which zsh)
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install emacs
RUN wget https://raw.githubusercontent.com/fwilliams/configs/master/emacs/config.el
RUN mv config.el ~/.emacs
RUN export PATH=/usr/local/cuda-11.1/bin:$PATH
RUN export CPATH=/usr/local/cuda-11.1/include:$CPATH
RUN export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
RUN export DYLD_LIBRARY_PATH=/usr/local/cuda-11.1/lib:$DYLD_LIBRARY_PATHl
RUN python -m pip install git+git://github.com/fwilliams/point-cloud-utils