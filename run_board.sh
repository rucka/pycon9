ROOT=$(cd "$(dirname "$0")"; pwd)
if [ ! "$(docker ps -q -f name=pycdeepboard)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=pycdeepboard)" ]; then
        docker rm pycdeepboard
    fi
    docker run --rm --name pycdeepboard -it -v $ROOT/tmp:/tmp -v $ROOT/data:/data -v $ROOT/logs:/logs -v $ROOT/code:/code -p 6006:6006 -w /code rucka/deeplearning:0.1 sh -c "tensorboard --logdir=$1"
fi