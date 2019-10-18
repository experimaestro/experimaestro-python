# Intermediate image

FROM quay.io/pypa/manylinux2010_x86_64 as intermediate

ARG COMPILE_THREADS=8
ARG POCO_VERSION=1.9.3
ARG LIBSSH_VERSION=0.9.0

WORKDIR /root

# Necessary packages: xz to uncompress, openssl-devel for libssh
RUN yum install -y openssl-devel xz

# Install cmake
ADD https://github.com/Kitware/CMake/releases/download/v3.15.3/cmake-3.15.3-Linux-x86_64.sh  /root
RUN bash cmake-3.15.3-Linux-x86_64.sh --skip-license --prefix=/usr/local

# Install libssh
ADD https://www.libssh.org/files/0.9/libssh-${LIBSSH_VERSION}.tar.xz /root/
COPY build-ssh.sh /root
RUN bash build-ssh.sh

# Install poco
ADD https://github.com/pocoproject/poco/archive/poco-${POCO_VERSION}-release.tar.gz /root
COPY build-poco.sh /root
RUN bash build-poco.sh

# Install a patched version of patchelf
# To remove when upstream bug is fixed: 
# See https://github.com/pypa/auditwheel/issues/159
COPY build-patchelf.sh /root
RUN bash build-patchelf.sh

# Final image - just copy /usr/local

FROM quay.io/pypa/manylinux2010_x86_64
COPY --from=intermediate /usr/local /usr/local
RUN yum install -y openssl-devel && yum clean all && rm -rf /var/cache/yum
RUN mkdir /root/workdir
VOLUME [ "/root/workdir" ]
WORKDIR /root/workdir
