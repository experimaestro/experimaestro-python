cd /root
tar xJf libssh-${LIBSSH_VERSION}.tar.xz 
cd  libssh-${LIBSSH_VERSION}
mkdir build
cd build
cmake -DWITH_STATIC_LIB=OFF -DCMAKE_BUILD_TYPE=Release ..
make -j${COMPILE_THREADS}
make install
