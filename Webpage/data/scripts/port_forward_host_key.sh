#!/bin/bash

if [ $# -ne 4 ]
then
    echo "Usage : "
    echo `basename $0` " login gpu_hostname port id_rsa_file"
    exit
fi

login=$1
host=$2
port=$3
id_rsa_filepath=$4

ssh_options_node="-o ProxyCommand=ssh -i $id_rsa_filepath -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -W %h:%p \"-o ProxyCommand=ssh -i $id_rsa_filepath -W %h:%p $login@ghome.metz.supelec.fr\" $login@term2.grid"
ssh -i $id_rsa_filepath "$ssh_options_node" -v -N -L $port:localhost:$port $login@$host
