FROM carlasim/carla:0.9.14

USER root

RUN apt-get update -y
RUN apt-get install net-tools -y

RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install python3.8 -y

RUN apt-get install python3-pip -y
RUN python3.8 -m pip install --upgrade pip
RUN pip install carla==0.9.14

# We omit this because we need to expose ports manually
# expose ports
# ARG PORT_0=33333
# ARG PORT_1=33334
# ARG PORT_2=33335
# EXPOSE $PORT_0
# EXPOSE $PORT_1
# EXPOSE $PORT_2

