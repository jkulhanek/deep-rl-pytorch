Bootstrap: docker
From: kulhanek/deep-rl-pytorch:latest
%post
    mkdir /deep-rl-pytorch
    cd /deep-rl-pytorch
    git init
    git remote add origin https://github.com/jkulhanek/deep-rl-pytorch.git
    git pull origin master

%runscript
    echo "Container is ready!"
    echo "Launching experiment with arguments [$@]"
    exec python3 "/deep-rl-pytorch/$@"