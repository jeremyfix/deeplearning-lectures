# Linux Cluster scripts

These cluster scripts are used for accessing our GPU clusters if you have access to it. These clusters are scheduled with [OAR](http://oar.imag.fr/). 

**book.sh** : for booking a node

	zaza:$ book.sh -h
	Usage :
	Books a node on the GPU cluster of CentraleSupelec Metz

	   -u, --user <login>          login to connect to CS Metz
	   -c, --cluster <cluster>     uSkynet, cameron, tx (default: uSkynet), optional
	   -w, --walltime <walltime>   in hours (default: 48), optional
	   -l, --local                 if on the local network of CS Metz. A shorter network path to the cluster is issued, optional
	   -h, --help                  prints this help message

**log.sh** : for logging to a booked node
					
	zaza:$ log.sh -h	
	Usage: 
	Logs to an already booked node on the GPU cluster of CentraleSupelec Metz

	   -u, --user <login>          login to connect to CS Metz
	   -l, --local                 if on the local network of CS Metz. A shorter network path to the cluster is issued, optional
	   -j, --jobid <JOB_ID>        The JOB_ID to which to connect. If not provided a list of your booked JOB_ID will be displayed
	   -h, --help                  prints this help message

**port_forward.sh**: for forwarding a port from a booked node to your localhost

	zaza:$ port_forward.sh -h
	Usage :
	Forward a port from a machine you booked to your local computer

	   -u, --user <login>          login to connect to CS Metz
	   -l, --local                 if on the local network of CS Metz. A shorter network path to the cluster is issued, optional
	   -j, --jobid <JOB_ID>        The JOB_ID to which to connect. If not provided a list of your booked JOB_ID will be displayed
	   -m, --machine <MACHINE>     The booked hostname, optional
	   -k, --key <PATH_TO_KEY>     Use the provided ssh key for connection, optional
	   -p, --port <PORT>           The distant port <PORT> will be binded to localhost:PORT
	   -h, --help                  prints this help message 

**kill_reservation.sh** : for releasing a booked node

	zaza:$ kill_reservation.sh -h
	Usage :
	Deletes a reservation on the GPU cluster of CentraleSupelec Metz

	   -u, --user <login>          Login to connect to CS Metz
	   -l, --local                 If on the local network of CS Metz. A shorter network path to the cluster is issued, optional
	   -j, --jobid <JOB_ID>        The JOB_ID to delete. If not provided a list of your booked JOB_ID will be displayed
	   -h, --help                  Prints this help message


# Windows cluster scripts

Under windows, you need to download plink.exe on the [Putty page](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html) and place it in the same directory of port_forward_windows.bat. You also need a ssh key.

If you use a ssh key generated with ssh-keygen, you need to convert it using [puttygen.exe](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)


