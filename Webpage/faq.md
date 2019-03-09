---
title:  'Frequently asked questions'
author:
- Jeremy Fix
keywords: [CentraleSupelec, GPUs, FAQ]
...

All the stuff I do not know where to put otherwise :)

I always get timeout when trying to download something (bash and python) on the CentraleSupelec cluster ??
----------------------------------------------------------------------------------------------------------

Accessing the web from a GPU node goes through a proxy. If you get this
issue, you probably do not have the right environment variables set.

From a python script :

``` {.sourceCode .python}
import sys,os,os.path
os.environ['HTTP_PROXY']="http://cache.metz.supelec.fr:3128"
os.environ['HTTPS_PROXY']="https://cache.metz.supelec.fr:3128"
```

From a terminal :

``` console
sh11:~:mylogin$ source /etc/profile.d/proxy.sh
sh11:~:mylogin$ echo $http_proxy
http://cache.metz.supelec.fr:3128
```
