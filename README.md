# [`allenai/tango`](https://github.com/allenai/tango) version of [`textual inversion`](https://arxiv.org/abs/2208.01618)

## Install dependencies

```shell
poetry install
```

## Login through `huggingface-cli`

```shell
poetry run huggingface-cli login
```

## Run a tango experiment

```shell
poetry run tango run configs/textual_inversion.jsonnet -w workspace
```

## Examples of generated image

- Prompt: `A <cat-toy> backpack`

![](./cat-backpack.png)


## Watch the workspace with `tango server`

```shell
poetry run tango server -w workspace/
```

![](./tango_server.png)
