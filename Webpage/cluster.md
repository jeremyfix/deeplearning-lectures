---
title:  'Using the GPU cluster of CentraleSupelec'
author:
- Jeremy Fix
keywords: [CentraleSupelec, GPUs]
...


## For the life-long training sessions SM20

Allocation of the GPU machines are handled by a resource manager called
OAR ... but this is not very important for you because I should have
already booked the machines for you ! However, you need the following
script to easily use your GPU machine :

-   [port_forward.sh](../ClusterScripts/port_forward.sh) : to activate
    port forwarding for Tensorboard, jupyter lab, ...

You should add the execution permission on this file :

``` console
mymachine:~:mylogin$ chmod u+x port_forward.sh
```

You also need the SSH key **id_rsa_SM20** I provide you during the session. You need to adapt its permissions
``` console
mymachine:~:mylogin$ chmod 600 id_rsa_SM20
```

**Important** For the following, you need to know which login has been
assigned to you, and which hostname has been assigned to you. In the
following, I will denote **dummyLog** your login and **dummyGPU** your
hostname.

### Accessing jupyter lab

I started your reservations with a jupyter lab session running. To
access it. To access it locally, just execute the port_forward
script specifying the port 8888 :

``` console
mymachine:~:mylogin$ ./port_forward.sh -u dummyLog -m dummyGPU -p 8888 -k id_rsa_SM20
```

You can now open **locally** a browser and open the page :
localhost:8888 ; You should reach your jupyter lab session.

### Accessing tensorboard

I now suppose that you already started tensorboard from within a
terminal in jupyter lab. To view locally the tensorboard interface, just
run :

``` console
mymachine:~:mylogin$ ./port_forward.sh -u dummyLog -m dummyGPU -p 6006 -k id_rsa_SM20
```

You can now open **locally** a browser and open the page :
localhost:6006 ; You should reach your tensorboard session.


## For the Master (AVR, PSA) students

Allocation of the GPU machines are handled by a resource manager called
OAR ... but this is not very important for you because I should have
already booked the machines for you ! However, you need the following
script to easily use your GPU machine :

-   [port_forward.sh](../ClusterScripts/port_forward.sh) : to activate
    port forwarding for Tensorboard, jupyter lab, ...

You should add the execution permission on this file :

``` console
mymachine:~:mylogin$ chmod u+x port_forward.sh
```

**Important** For the following, you need to know which login has been
assigned to you, and which hostname has been assigned to you. In the
following, I will denote **dummyLog** your login and **dummyGPU** your
hostname. You also need to have the private SSH key, ask your teacher with its path denoted below **path/to/id_rsa**

### Setting up the key

You have to modify the permissions on the file **path/to/id_rsa**

``` console
mymachine:~:mylogin$ chmod 600 path/to/id_rsa
```


### Accessing jupyter lab

I started your reservations with a jupyter lab session running. To
access it. To access it locally, just execute the port_forward
script specifying the port 8888 :

``` console
mymachine:~:mylogin$ ./port_forward.sh -u dummyLog -m dummyGPU -p 8888 -k path/to/id_rsa
```

You can now open **locally** a browser and open the page :
localhost:8888 ; You should reach your jupyter lab session.

### Accessing tensorboard

I now suppose that you already started tensorboard from within a
terminal in jupyter lab. To view locally the tensorboard interface, just
run :

``` console
mymachine:~:mylogin$ ./port_forward.sh -u dummyLog -m dummyGPU -p 6006 -k path/to/id_rsa
```

You can now open **locally** a browser and open the page :
localhost:6006 ; You should reach your tensorboard session.




## For the CentraleSupelec students

### The scripts

Allocation of the GPU machines are handled by a resource manager called
OAR. It can be annoying to remember the command lines to reserve a
machine and log to it. We therefore provide the scripts :

-   [book.sh](../ClusterScripts/book.sh) : to book a GPU
-   [kill\_reservation.sh](../ClusterScripts/kill_reservation.sh) : to free the
    reservation
-   [log.sh](../ClusterScripts/log.sh) : to log to the booked GPU
-   [port\_forward.sh](../ClusterScripts/port_forward.sh) : to activate port
    forwarding for Tensorboard, jupyter lab, ..

After getting these scripts, please make them executables :

```console
mymachine:~:mylogin$ chmod u+x book.sh kill_reservation.sh log.sh port_forward.sh
```

These scripts help you to make a reservation and log to the reserved
machine. These scripts must be in the **same** directory. The book.sh
script handles only one reservation, i.e. running it two times will
simply kill the first reservation.

### The how to

Get the scripts and run book.sh and log.sh as below. We also show a
typical output from the execution of the script.

``` console
mymachine:~:mylogin$ ./book.sh -u mylogin
Booking a node
Reservation successfull
Booking requested : OAR_JOB_ID =  99785
Waiting for the reservation to be running, might last few seconds
   The reservation is not yet running 
   The reservation is not yet running 
   The reservation is not yet running 
   The reservation is not yet running 
   [...]
   The reservation is not yet running 
   The reservation is not yet running 
   The reservation is running
The reservation is running
mymachine:~:mylogin$
```

If the reservation is successfull, you can then log to the booked GPU. If you do not know or remember your jobid, proceed the following way

```console
mymachine:~:mylogin$ ./log.sh -u mylogin
```

As you get your job id, you can proceed

``` console
mymachine:~:mylogin$ ./log.sh -u mylogin -j 99785
The file job_id exists. I am checking the reservation is still valid 
   The reservation is still running 
Logging to the booked node 
Connect to OAR job 99785 via the node sh11
sh11:~:mylogin$ 
```

You end up with a terminal logged on a the GPU machine where you can
execute your code. Your reservation will run for 24 hours. If you need
more time, you may need to tweak the bash script a little bit.

You can log any terminal you wish to the booked machine.

To get access to tensorboard, you need to log to the GPU, start
tensorboard and activate port forwarding :

``` console
[ In a first terminal ]
mymachine:~:mylogin$ ./log.sh -u mylogin -j 99785
...
sh11:~:mylogin$ tensorboard --logdir path_to_the_logs

[ In a second terminal ]
mymachine:~:mylogin$ ./port_forward.sh -u mylogin -j 99785 -p 6006
...
```

You can now open a browser on your machine on the port 6006 and you
should get access to tensorboard.

Once your work is finished, just unlog from the machine and run kill_reservation.sh. Please kill your reservation as soon as your work is finished in order to allow other users to book it. 

``` console
sh11:~:mylogin$ logout
Connection to sh11 closed.
Disconnected from OAR job 99785
Connection to term2.grid closed.
  Unlogged 
mymachine:/home/mylogin:mylogin$ ./kill_reservation.sh -u mylogin -j 99785
 The file job_id exists. I will kill the previous reservation in case it is running
Deleting the job = 99785 ...REGISTERED.
The job(s) [ 99785 ] will be deleted in a near future.
Waiting for the previous job to be killed
Done
mymachine:~:mylogin$
```


