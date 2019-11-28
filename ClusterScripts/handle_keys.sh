#!/bin/bash

set -e

usage_m="Usage :
Handles SSH keys of a group of users

   -p, --prefix <login prefix> login prefix of the group of users
   -r, --range <min> <max>     range to consider
   -h, --help                  prints this help message
"

GREEN="\\e[1;32m"
NORMAL="\\e[0;39m"
RED="\\e[1;31m"
BLUE="\\e[1;34m"
MAGENTA="\\e[1;35m"

display_info() {
    echo -e "$BLUE $1 $NORMAL"
}
display_wait() {
    echo -e "$MAGENTA $1 $NORMAL"
}
display_success() {
    echo -e "$GREEN $1 $NORMAL"
}
display_error() {
    echo -e "$RED $1 $NORMAL"
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -p|--prefix)
            PREFIXLOGIN="$2"
            shift # pass argument
            shift # pass value
            ;;
		-r|--range)
			RANGEMIN=$2
			RANGEMAX=$3
			shift  # pass argument
			shift  # pass value min
			shift  # pass value max
			;;
        -h|--help)
			display_info "$usage_m"
			exit -1
			;;
		*)
			display_error "Unrecognized option $key"
			exit -1
			;;
	esac
done

# Check if the required fields have been specified
if [ -z $PREFIXLOGIN ]
then
	display_error "A login prefix is required, see --help"
	exit -1
fi
if [ -z $RANGEMIN ] || [ -z $RANGEMAX ]
then
	display_error "A range is required, see --help"
	exit -1
fi

quit=0
menu="
What should we do ?
   l : list the SSH keys
   r : remove a key
   g : generate new keys
   q : quit
" 

list_ssh_keys()
{
	for i in $(seq $RANGEMIN $RANGEMAX)
	do
		display_info "==== User $PREFIXLOGIN$i ===="
		ssh $PREFIXLOGIN$i@term2 'cat $HOME/.ssh/authorized_keys'
	done
}
remove_ssh_key()
{
	dryrun=$1
	keyword=$2
	for i in $(seq $RANGEMIN $RANGEMAX)
	do
		if [ "$dryrun" -eq "1" ] 
		then
			display_info "==== USER $PREFIXLOGIN$i ===="
			display_info "Will keep : "
			ssh $PREFIXLOGIN$i@term2 'if test -f $HOME/.ssh/authorized_keys; then if grep -v "'$keyword'" $HOME/.ssh/authorized_keys > $HOME/.ssh/keytmp_keep; then cat $HOME/.ssh/keytmp_keep; fi; fi'
			display_info "Will drop : "
			ssh $PREFIXLOGIN$i@term2 'if test -f $HOME/.ssh/authorized_keys; then if grep "'$keyword'" $HOME/.ssh/authorized_keys > $HOME/.ssh/keytmp_drop; then cat $HOME/.ssh/keytmp_drop; fi; fi'
		else
			display_info "==== USER $PREFIXLOGIN$i ===="
			ssh $PREFIXLOGIN$i@term2 'mv $HOME/.ssh/keytmp_keep $HOME/.ssh/authorized_keys; rm -f $HOME/.ssh/keytmp_drop'
			ssh $PREFIXLOGIN$i@term2 'cat $HOME/.ssh/authorized_keys'
			
		fi
	done
}
generate_new_key()
{
	keyword=$1
	display_info "Generating the keys"
	ssh-keygen -t rsa -b 2048 -N "" -C "$keyword" -f ./id_rsa_$PREFIXLOGIN
	for i in $(seq $RANGEMIN $RANGEMAX)
	do
		display_info "Removing previous rsa keys on the home of $PREFIXLOGIN$i"
		ssh $PREFIXLOGIN$i@term2 "rm -f $HOME/id_rsa*"
		display_info "Copying the private/public keys on the home of $PREFIXLOGIN$i"
		scp id_rsa_$PREFIXLOGIN $PREFIXLOGIN$i@term2:~/.ssh/id_rsa	
		scp id_rsa_$PREFIXLOGIN.pub $PREFIXLOGIN$i@term2:~/.ssh/id_rsa.pub	
		display_info "Copy ID"
		ssh-copy-id -i id_rsa_$PREFIXLOGIN $PREFIXLOGIN$i@term2 
	done
}

while [ $quit -ne 1 ]
do
	display_info "$menu"
	read choice
	case $choice in
		l)
			list_ssh_keys
			;;
		r)
			display_info "Which keyword should I consider ?"
			read unique_keyword
			# Dry run
			remove_ssh_key 1 $unique_keyword
			display_info "Should I apply it ? [y/n]"
			read doit
			if [ "$doit" == "y" ]; then
				display_success "Doing it"
				remove_ssh_key 0 $unique_keyword
			elif [ "$doit" == "n" ]; then
				display_success "Not doing it"
			else
				display_error "Unrecognized option"
			fi
			;;
		g)
			generate_new_key ""$PREFIXLOGIN"_GPU"
			;;
		q)
			quit=1
			;;
		*)
			display_error "Unknown choice"
			;;	
	esac
done

