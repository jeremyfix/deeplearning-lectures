# With a local conda environment 

(to prevent a globally installed virtualenv), conda expected to be installed in `/opt/conda` , otherwise needs to be
adapted

```
source "/opt/conda/etc/profile.d/conda.sh"
export PATH=$PATH:/opt/conda/bin
conda create --prefix $TMPDIR/venv python==3.9
conda activate $TMPDIR/venv
```

then we can `conda install`
