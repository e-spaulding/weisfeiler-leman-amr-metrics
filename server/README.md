## Requirements

Create and activate the conda environment:

```
conda env create -f environment.yml
```

## Running the server

Instructions for running the server on Ego using `screen`.

Existing screen:

```
screen -r wwlk_server
./run_server.sh
```

Create a new screen:

```
screen -S wwlk_server
cd weisfeiler-leman-amr-metrics
conda deactivate
source activate wl-kernel
./run_server.sh
```

(ctrl+a+d to detach from the screen.)

Requests will be of the form

```
curl localhost:5000/ -H 'Content-Type: application/json' -d '{"kernel": "wwlk", "amr1": "(vv1 / bake :ARG0 (vv2 / man :mod (vv3 / big)))", "amr2": "(vv1 / bake :ARG0 (vv2 / woman))", "config_filepath": "/home/elizabeth/weisfeiler-leman-amr-metrics/server/example/embedding_config.yaml"}'
```

Using the example, you should get this exact response:

```
{"score":-0.142}
```

