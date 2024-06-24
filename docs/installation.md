## Installation

<details>
<summary>#### Installation of GIN Client</summary>

``` bash
install-gin-client() {
    if ! command -v git-annex &> /dev/null; then
        install-git-annex-using-cable
    fi

    ORIG_DIR=$PWD
    cd /tmp
    wget https://gin.g-node.org/G-Node/gin-cli-releases/raw/master/gin-cli-latest-linux.tar.gz
    tar xvf gin-cli-latest-linux.tar.gz
    sudo mv gin /usr/local/bin
    gin --version
    gin login # GIN Login ID, password
    cd $ORIG_DIR
}

install-git-annex-using-cabal() {
    # Install Development Tools group
    sudo dnf groupinstall -y "Development Tools"

    # Install EPEL repository if not already installed
    sudo dnf install -y epel-release
    sudo dnf install -y powertools
    sudo dnf config-manager --set-enabled powertools # For Rocky Linux 8
    sudo dnf config-manager --set-enabled crb # For Rocky Linux 9

    # Install required dependencies
    sudo dnf install -y \
        alex \
        bzip2-devel \
        cabal-install \
        file \
        file-devel \
        file-libs \
        gcc \
        ghc \
        gnutls-devel \
        happy \
        libgsasl-devel \
        libidn-devel \
        libxml2-devel \
        make \
        openssl-devel \
        xz-devel \
        zlib-devel
    
    cabal update
    BINDIR=$HOME/.cabal/bin
    cabal install --bindir=$BINDIR c2hs
    cabal configure --extra-include-dirs=/usr/include --extra-lib-dirs=/usr/lib64
    cabal install --bindir=$BINDIR git-annex
    export PATH=$BINDIR:$PATH
    grep -qxF 'export PATH=$HOME/.cabal/bin:$PATH' $HOME/.bashrc || echo 'export PATH=$HOME/.cabal/bin:$PATH' >> $HOME/.bashrc
    git-annex --version
}
```

</details>

<details>
<summary>#### Installation of Python packages</summary>

``` bash
git clone git@github.com:ywatanabe1989/siEEG_ripple.git && cd siEEG_ripple
python -m venv env && source ./env/bin/activate
pip install -U pip && pip install -r requirements.txt

# External scripts
mkdir -p ./scripts/externals/ && cd ./scripts/externals/
git clone git@github.com:ywatanabe1989/mngs.git && pip install -e ./mngs # for development

# Adds current directory in PYTHONPATH
echo "export PYTHONPATH=.:$PYTHONPATH" >> ~/.bashrc
```

</details>


<details>
<summary>#### Downloading of the dataset by Boran et al., 2020</summary>

## Downloads the original .h5 files using gin

``` bash
cd ./scripts/externals/
gin get USZ_NCH/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM
cd Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM

download_h5_files() {
    for f in ./data_nix/*.h5; do
        gin unlock $f
        gin get-content $f
    done
}

screen -dmS download_Boran_et_al bash -c "$(declare -f download_h5_files); download_h5_files"

```
</details>


<details>
<summary>#### Softlink to the h5 data</summary>

```bash
cd data
ln -s ../scripts/externals/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/data_nix \
    ./data_nix
ls ./data_nix
```

</details>
