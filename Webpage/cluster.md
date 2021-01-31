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

-   [cscluster](./data/ClusterScripts/cscluster) : to activate
    port forwarding for Tensorboard, jupyter lab, ...

You should add the execution permission on this file :

``` console
mymachine:~:mylogin$ chmod u+x cscluster
```

You also need the SSH key **id_rsa_SM20** I provide you during the session. You need to adapt its permissions
``` console
mymachine:~:mylogin$ chmod 600 id_rsa_SM20
```

**Important** For the following, you need to know which login has been
assigned to you. In the
following, I will denote **dummyLog** your login.

### Accessing jupyter lab

I started your reservations with a jupyter lab session running. To
access it. To access it locally, just execute the port_forward
script specifying the port 8888 :

``` console
mymachine:~:mylogin$ ./cscluster port_forward -u dummyLog -c gpu -p 8888 -k id_rsa_SM20
```

You can now open **locally** a browser and open the page :
[localhost:8888](http://localhost:8888) ; You should reach your jupyter lab session.

### Accessing tensorboard

I now suppose that you already started tensorboard from within a
terminal in jupyter lab. To view locally the tensorboard interface, just
run :

``` console
mymachine:~:mylogin$ ./cscluster port_forward -u dummyLog -c gpu -p 6006 -k id_rsa_SM20
```

You can now open **locally** a browser and open the page :
[localhost:6006](http://localhost:6006) ; You should reach your tensorboard session.


## For the Master (AVR, PSA) students

Allocation of the GPU machines are handled by a resource manager called
OAR ... but this is not very important for you because I should have
already booked the machines for you ! However, you need the following
script to easily use your GPU machine :

-   [cscluster](./data/ClusterScripts/cscluster) : to activate
    port forwarding for Tensorboard, jupyter lab, ...

You should add the execution permission on this file :

``` console
mymachine:~:mylogin$ chmod u+x cscluster
```

**Important** For the following, you need to know which login has been
assigned to you. In the
following, I will denote **dummyLog** your login. You also need to have the private SSH key, ask your teacher with its path denoted below **path/to/id_rsa**

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
mymachine:~:mylogin$ ./cscluster port_forward -u dummyLog -c gpu -p 8888 -k path/to/id_rsa
```

You can now open **locally** a browser and open the page :
[localhost:8888](http://localhost:8888) ; You should reach your jupyter lab session.

### Accessing tensorboard

I now suppose that you already started tensorboard from within a
terminal in jupyter lab. To view locally the tensorboard interface, just
run :

``` console
mymachine:~:mylogin$ ./cscluster port_forward -u dummyLog -c gpu -p 6006 -k path/to/id_rsa
```

You can now open **locally** a browser and open the page :
[localhost:6006](http://localhost:6006) ; You should reach your tensorboard session.




## For the CentraleSupelec students

Go on the [dedicated website](http://tutos.metz.centralesupelec.fr/TPs/Clusters/index.html). There it is explained how to book a node, log on it, start a VNC server and so on.
