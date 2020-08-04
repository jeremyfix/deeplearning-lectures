# Linux Cluster scripts

These cluster scripts are used for accessing our GPU clusters if you have access to it. These clusters are scheduled with [OAR](http://oar.imag.fr/). 

For **booking** a node

	zaza:$ cscluster book -h

	Usage :  cscluster book
	Books a node on the CS Metz clusters

		-u, --user <login>          login to connect to CentraleSupelec Metz clusters 
		-m, --machine <machine>     OPTIONAL, a specific machine
		-c, --cluster <cluster>     the cluster (e.g: uSkynet, cam, tx, kyle, sarah, john)
		-w, --walltime <walltime>   in hours (default: 24)
		-h, --help                  prints this help message

		Options specific to clusters handled with SLURM (Kyle):
		-p, --partition <partition> on which partition to book a node

For **logging** to a booked node
					
	zaza:$ cscluster log -h	

    Usage : cscluster log
	Logs to an already booked node on the CentraleSupelec Metz cluster 

		-u, --user <login>          login to connect to CentraleSupelec Metz
		-f, --frontal <frontal>     OPTIONAL, the frontal (e.g. term2.grid, slurm1, ..)
		-c, --cluster <cluster>     OPTIONAL, the cluster (e.g: uSkynet, cam, tx, kyle, sarah, john)
		-j, --jobid <JOB_ID>        The JOB_ID to which to connect. If not provided
		a list of your booked JOB_ID will be displayed
		-h, --help                  prints this help message

		You must specify either the cluster or the frontal but not both.

For **forwarding a port** from a booked node to your localhost

	zaza:$ cscluster port_forward -h

    usage :  cscluster port_forward
	Forward a port from a machine you booked to your local computer

		-u, --user <login>          login to connect to CentraleSupelec Metz
		-f, --frontal <frontal>     OPTIONAL, the frontal (e.g. term2.grid, slurm1, ..)
		-c, --cluster <cluster>     OPTIONAL, the cluster (e.g: uSkynet, cam, tx, kyle, sarah, john)
		-j, --jobid <JOB_ID>        The JOB_ID to which to connect. If not provided
		a list of your booked JOB_ID will be displayed
		-m, --machine <MACHINE>     The booked hostname.
		-p, --port <PORT>           The distant port <PORT> will be binded to 127.0.0.1:PORT
		-k, --key <PATH_TO_KEY>     Use the provided ssh key for connection
		-h, --help                  prints this help message

		You must specify either the cluster or the frontal but not both.

For **releasing** a booked node

	zaza:$ cscluster kill -h

    usage :  cscluster kill 
	Deletes a reservation on the CentraleSupelec Metz cluster

		-u, --user <login>          Login to connect to CentraleSupelec Metz
		-f, --frontal <frontal>     OPTIONAL, the frontal (e.g. term2.grid, slurm1, ..)
		-c, --cluster <cluster>     OPTIONAL, the cluster (e.g: uSkynet, cam, tx, kyle, sarah, john)
		-j, --jobid <JOB_ID>        OPTIONAL The JOB_ID to delete. If not provided
		a list of your booked JOB_ID will be displayed
		-j, --jobid all             Will kill all the jobs booked by <login>
		-h, --help                  Prints this help message

		You must specify either the cluster or the frontal but not both.

# Windows cluster scripts

## Using the bash scripts

You may be able to use the above bash scripts using :

- [PowerShell](https://docs.microsoft.com/en-us/powershell/)
- adding SSH support which can be done by using this command "Add-WindowsCapability -Online -Name OpenSSH.Client*" which installs OpenSSH
- adding BASH support which is part of the UNIX tools, and included with the installation of [GIT for Windows](https://gitforwindows.org/)

## Using putty plink

Under windows, you need to download [plink.exe](https://the.earth.li/~sgtatham/putty/latest/w64/plink.exe) on the [Putty page](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html) and place it in the same directory of port_forward_windows.bat. You also need a ssh key.

If you use a ssh key generated with ssh-keygen, you need to convert it using [puttygen.exe](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)


