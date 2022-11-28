# COCOpen
UIUC COCOpen API for generated COCO formatted datasets automatically

## Installation

Using the RMDLO COCOpen library requires desktop [configuration of a GitHub SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

To install dependencies using `conda`, perform the below command in a terminal.
```bash
# Clone the repository
$ git clone git@github.com:RMDLO/COCOpen.git
# Install dependencies
$ cd COCOpen
$ In the environment.yml file, change `name` to the name you would like for the conda environment
$ Run conda env create -f environment.yml command
$ Activate the environment: conda activate <env name>
```