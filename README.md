<div align="center">
<img src="/docs/assets/logo.png" width="50%" title="Armory Logo">
</div>

[![CI][ci-badge]][ci-url]
[![PyPI Status Badge][pypi-badge]][pypi-url]
[![PyPI - Python Version][python-badge]][python-url]
[![License: MIT][license-badge]][license-url]
[![Docs][docs-badge]][docs-url]
[![Code style: black][style-badge]][style-url]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7561756.svg)](https://doi.org/10.5281/zenodo.7561756)


# Overview

Armory is a testbed for running scalable evaluations of adversarial defenses.
Configuration files are used to launch local or cloud instances of the Armory docker
containers. Models, datasets, and evaluation scripts can be pulled from external
repositories or from the baselines within this project.

Our evaluations are created so that attacks and defenses may be
interchanged. To do this we standardize all attacks and defenses as subclasses of
their respective implementations in the [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) hosted by the LF AI & Data Foundation (LFAI).


# Installation & Configuration

TLDR: Try Armory [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/twosixlabs/armory/blob/master/notebooks/running_armory_scenarios_interactively.ipynb) or follow the instructions below to install locally.


```bash
pip install armory-testbed
```

Upon installing armory, a directory will be created at `~/.armory`. This user
specific folder is the default directory for downloaded datasets, model weights, and
evaluation outputs.

To change these default directories simply run `armory configure` after installation.

If installing from the git repo in editable mode, ensure that your pip version is 22+.


# Usage

To interact with Armory's container system do the following:

## armory run
* `armory run <path/to/config.json>`
This will run a [configuration file](//github.com/twosixlabs/armory/raw/master/docs/configuration_files.md) end to end. Stdout
and stderror logs will be displayed to the user, and the container will be removed
gracefully upon completion. Results from the evaluation can be found in your output
directory.

* `armory run <path/to/config.json> --interactive`
This will launch the framework-specific container specified in the
configuration file, copy the configuration file into the container, and provide
the commands to attach to the container in a separate terminal and run the
configuration file end to end while attached to the container. A notable use
case for this would be to debug using pdb. Similar to non-interactive mode, results
from the evaluation can be found in the output directory. To later close the
interactive container simply run CTRL+C from the terminal where this command was ran.

### Example usage:
```bash
pip install armory-testbed
armory configure

git clone https://github.com/twosixlabs/armory-example.git
cd armory-example
armory run official_scenario_configs/cifar10_baseline.json
```

# Scenarios
Armory provides several baseline threat-model scenarios for various data modalities.
When running an armory configuration file, the robustness of a defense will be
evaluated against that given scenario. For more information please see our
[Scenario Documentation](//github.com/twosixlabs/armory/raw/master/docs/scenarios.md).

# FAQs
Please see the [frequently asked questions](//github.com/twosixlabs/armory/raw/master/docs/faqs.md) documentation for more information on:
* Dataset format and preprocessing
* Access to underlying models from wrapped classifiers.

# Contributing
Armory is an open source project and as such we welcome contributions! Please refer to
our [contribution docs](//github.com/twosixlabs/armory/raw/master/.github/CONTRIBUTING.md) for how to get started.

# Acknowledgment
This material is based upon work supported by the Defense Advanced Research Projects
Agency (DARPA) under Contract No. HR001120C0114. Any opinions, findings and
conclusions or recommendations expressed in this material are those of the author(s)
and do not necessarily reflect the views of the Defense Advanced Research Projects
Agency (DARPA).


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[ci-badge]: https://github.com/twosixlabs/armory/workflows/GitHub%20CI/badge.svg
[ci-url]: https://github.com/twosixlabs/armory/actions/
[pypi-badge]: https://badge.fury.io/py/armory-testbed.svg
[pypi-url]: https://pypi.org/project/armory-testbed
[python-badge]: https://img.shields.io/pypi/pyversions/armory-testbed
[python-url]: https://pypi.org/project/armory-testbed
[license-badge]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://opensource.org/licenses/MIT
[docs-badge]: https://readthedocs.org/projects/armory/badge/
[docs-url]: https://readthedocs.org/projects/armory/
[style-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[style-url]: https://github.com/ambv/black
