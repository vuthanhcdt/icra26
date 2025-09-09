## Build

~~~
mkdir build
cd build
cmake ..
make
~~~

This will generate the `libControlActor.so` library under `build`.

## Run

Add the library to the path:

~~~
cd examples/plugin/command_actor
export GZ_SIM_SYSTEM_PLUGIN_PATH=`pwd`/build
~~~

Then run the demo world:
gz sim -v 4 control_actor.sdf -r
