ROOT=$(cd "$(dirname "$0")"; pwd)
if [ ! "$(docker ps -q -f name=pycdeepshell)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=pycdeepshell)" ]; then
        docker rm pycdeepshell
    fi
    docker run --rm --name pycdeepshell -it -v $ROOT/tmp:/tmp -v $ROOT/data:/data -v $ROOT/code:/code -v $ROOT/models:/models -v $ROOT/logs:/logs -w /code rucka/deeplearning:0.1 /bin/bash
fi