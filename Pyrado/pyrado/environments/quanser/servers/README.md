## Quanser Servers
For each Quanser platform (Qube a.k.a. Servo2, 2DoF Ball Balancer, Linear Inverted Pendulum) there is a subfolder containing a Simulink model, a Simulink Bus Object, and the binaries generated from this model running at 100Hz, 250Hz, or 500Hz.  
At the core of each Simulink model there is a [server block from Quanser](http://quanser-update.azurewebsites.net/quarc/documentation/stream_server_block.html), which manages the communication with the client. While the block itself is the same for all platforms, the required `BusObject` is specific for sensor measurements (e.g., the number of measured states).


### How to set up a fresh Windows PC to control Quanser plants from a remote client
1. Download and install:
   1. Matlab
   2. Visual Studio Pro 2015 (manually select `Programming Languages -> Visual C++` to install the compiler)
   3. Visual Studio SDK 2015
   4. QUARC 2.6 (might work with later version, but this is not tested). The documentation for QUARC can be found [here](http://quanser-update.azurewebsites.net/quarc/documentation/).
2. Create an inbound firewall rule on port 9095 that allows TCP-IP connections.
3. Check if the signature of your driver is not considered safe. 
   Go to `Device Manager` and check if there is a yellow exclamation mark next 
   to a driver under `USB devices`. If so, 
    1. [Disable secure boot](https://www.youtube.com/watch?v=S0sY0DWtRNw), and most importantly
    2. [Disable driver signature check](https://winaero.com/blog/disable-driver-signature-enforcement-permanently-in-windows-10/).
    3. Check if the yellow exclamation mark is gone. 
4. Adjust power settings by following instructions in Quanser's `quarc_installation_guide` (see the very last appendix called `Power Management`).
5. To start the server, execute, e.g., `quarc_run -r qube.rt-win64` (on the first run, a window might appear where you have to say OK); to stop, run `quarc_run -q qube.rt-win64`.
8. Now proceed to the clients in the repository (see `Pyrado/environments/quanser`) to run or create a Python client.

### How to run the servers
There are two ways:
1. Open the Simulink model, click on `Build all`, then click on `Connect to target`, and finally click on `Run` (these are all symbols in the top row of your Simulink window).
2. Run from the binary files (see point 5 above).
Now your server runs and you can start the Python client.
