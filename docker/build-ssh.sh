cd /root
tar xzf ssh-${LIBSSH_VERSION}.tar.gz
cd v${LIBSSH_VERSION/./-}/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j${COMPILE_THREADS}
make install
