# base

https://github.com/anibali/docker-pytorch/blob/1184c3ed80e5360c697e27788b25e9f92bddd06f/cuda-8.0/Dockerfile

## build and run

```
(cd docker/base && docker build -t mmg-base .)
nvidia-docker run -it --rm -v $(pwd):/home/user/code mmg-base /bin/bash
```

