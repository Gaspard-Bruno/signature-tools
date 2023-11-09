find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

eval "$(conda shell.bash hook)"
if find_in_conda_env ".*signature.*" ; then
    conda activate signature;
else
    conda create --name "signature" python=3.10 -y;
    conda activate signature;
fi

pip3 install -e . --no-cache-dir