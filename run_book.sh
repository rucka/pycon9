ROOT=$(cd "$(dirname "$0")"; pwd)
if [ ! "$(docker ps -q -f name=pycdeepbook)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=pycdeepbook)" ]; then
        docker rm pycdeepbook
    fi
    docker run --rm --name pycdeepbook -it -v $ROOT/data:/data -v $ROOT/tmp:/tmp -v $ROOT/logs:/logs -v $ROOT/models:/models -v $ROOT/code:/code -p 8888:8888 -w /code rucka/deeplearning:0.1 sh -c "/run_jupyter.sh --allow-root"
fi