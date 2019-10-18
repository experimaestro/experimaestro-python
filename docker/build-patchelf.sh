if test "$(patchelf  --version)" == "patchelf 0.10"; then
    git clone https://github.com/experimaestro/patchelf.git
    cd patchelf
    ./bootstrap.sh
    ./configure --prefix=/usr/local
    make && make install
else
    echo "WARNING: not building patched version of patchelf (version not 0.10)"
fi

