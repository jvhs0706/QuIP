deactivate
module purge
module load gcc/9.3.0 python/3.10 arrow/11.0.0 nodejs cuda/11.7 cmake protobuf cudnn scipy-stack rust
source $HOME/Quantization_ENV/bin/activate
echo "The Python used is $(which python)"

# salloc --account=rrg-hongyanz --gpus-per-node=p100:1 --cpus-per-task=6 --mem=32000M --time=0-02:59 srun $VIRTUAL_ENV/bin/jupyterlab.sh
# salloc --account=rrg-hongyanz --gpus-per-node=p100l:1 --cpus-per-task=6 --mem=64250M --time=0-02:59 srun $VIRTUAL_ENV/bin/jupyterlab.sh
salloc --account=rrg-hongyanz --gpus-per-node=v100l:1 --cpus-per-task=8 --mem=48000M --time=0-02:59 srun $VIRTUAL_ENV/bin/jupyterlab.sh
# salloc --account=rrg-hongyanz --gpus-per-node=a100:1 --cpus-per-task=12 --mem=127500M --time=0-02:59 srun $VIRTUAL_ENV/bin/jupyterlab.sh

# salloc --account=rrg-hongyanz --nodes=1 --gpus-per-node=p100l:4 --ntasks-per-node=1 --cpus-per-task=24 --exclusive --mem=257000M --time=0-02:59 srun $VIRTUAL_ENV/bin/jupyterlab.sh
# salloc --account=rrg-hongyanz --nodes=1 --gpus-per-node=v100l:4 --ntasks-per-node=1 --cpus-per-task=32 --exclusive --mem=192000M --time=0-02:59 srun $VIRTUAL_ENV/bin/jupyterlab.sh
# salloc --account=rrg-hongyanz --nodes=1 --gpus-per-node=a100:4 --ntasks-per-node=1 --cpus-per-task=48 --exclusive --mem=510000M --time=0-02:59 srun $VIRTUAL_ENV/bin/jupyterlab.sh