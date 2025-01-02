## View Documentation

The documentation of this repo is built with Github Action, and is available at `<insert_link_when_deployed>`

## Build Documentation Locally

To build documentation locally, install [sphinx_build](https://www.sphinx-doc.org/en/master/man/sphinx-build.html) with 

```
pip install -U sphinx
```

Then run the following command in the root of the repo, install any
missing dependencies if needed.

```
PYTHONPATH=$PYTHONPATH:<path to src/nki_samples> sphinx-build doc <dst_folder>
```

The HTML file of the doc will be available at `<dst_folder>/index.html`