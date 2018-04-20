ROOT=$(cd "$(dirname "$0")"; pwd)
if [ ! "$(docker ps -q -f name=pycdeepcmd)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=pycdeepcmd)" ]; then
        docker rm pycdeepcmd
    fi
    docker run --rm --name pycdeepcmd -it -v $ROOT/tmp:/tmp -v $ROOT/data:/data -v $ROOT/code:/code -v $ROOT/models:/models -v $ROOT/logs:/logs -w /code rucka/deeplearning:0.1 /bin/sh -c $1
fi