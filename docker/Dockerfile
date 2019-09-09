FROM quay.io/pypa/manylinux2010_x86_64 as intermediate

ARG COMPILE_THREADS=8
ARG POCO_VERSION=1.9.3
ARG LIBSSH_VERSION=0.6

WORKDIR /root

RUN yum install -y openssl-devel

# Install cmake
ADD https://github.com/Kitware/CMake/releases/download/v3.15.3/cmake-3.15.3-Linux-x86_64.sh  /root
RUN bash cmake-3.15.3-Linux-x86_64.sh --skip-license --prefix=/usr/local

# Install libssh
ADD https://git.libssh.org/projects/libssh.git/snapshot/v0-6.tar.gz /root/ssh-${LIBSSH_VERSION}.tar.gz
COPY build-ssh.sh /root
RUN bash build-ssh.sh

# Install poco
ADD https://github.com/pocoproject/poco/archive/poco-${POCO_VERSION}-release.tar.gz /root
COPY build-poco.sh /root
RUN bash build-poco.sh

FROM quay.io/pypa/manylinux2010_x86_64
COPY --from=intermediate /usr/local /usr/local
RUN yum install -y openssl-devel && yum clean all && rm -rf /var/cache/yum
WORKDIR /root
