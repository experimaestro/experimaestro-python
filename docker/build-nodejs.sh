cd /root
tar xzf node-v${NODEJS_VERSION}.tar.gz
cd node-v${NODEJS_VERSION}

# Dirty patch for https://github.com/nodejs/node/pull/30141

(cat << EOF
--- ../2/node-v12.13.1/common.gypi	2019-11-19 08:29:05.000000000 +0000
+++ ../common.gypi	2019-11-22 15:07:01.571392403 +0000
@@ -361,7 +361,8 @@
       [ 'OS in "linux freebsd openbsd solaris android aix cloudabi"', {
         'cflags': [ '-Wall', '-Wextra', '-Wno-unused-parameter', ],
         'cflags_cc': [ '-fno-rtti', '-fno-exceptions', '-std=gnu++1y' ],
-        'ldflags': [ '-rdynamic' ],
+	     'defines': ['__STDC_FORMAT_MACROS' ],
+        'ldflags': [ '-rdynamic', '-lrt' ],
         'target_conditions': [
           # The 1990s toolchain on SmartOS can't handle thin archives.
           ['_type=="static_library" and OS=="solaris"', {
EOF
) | patch common.gypi	

export PATH=/opt/python/cp27-cp27m/bin:$PATH 
./configure --prefix=/usr/local --without-node-snapshot --without-inspector
make -j${COMPILE_THREADS} && make install