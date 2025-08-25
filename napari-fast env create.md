conda create -y -n napari-fast -c conda-forge python=3.12 
conda activate napari-fast
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 
conda install conda-forge::ipykernel jupyter matplotlib
conda install conda-forge::numpy pandas scikit-image scipy loguru
python -m pip install cellpose --upgrade
conda install conda-forge::napari matplotlib-scalebar
pip install bioio bioio-ome-tiff bioio-ome-zarr bioio-czi