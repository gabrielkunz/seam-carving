# Seam Carving

description


## Context


The program was developed and executed in a MacBook Pro 13-inch running macOS 10.15.6. The same bash commands should work on a machine running a unix-based OS. The same program was tested afterwards in a Dell notebook running Ubuntu 16.0.4.


## Installation

It is recommended to use the program in a virtual environment. To create a virtual environment, the virtualenv package has to be installed:

```bash
python3 -m pip install virtualenv
```

After that, navigate to the folder where the 'main' folder for the scripts is located. Then we can create the virtual environment using the following command:

```bash
python3 -m venv env
```

You'll notice that the 'env' folder was created in your workspace. Now we need to activate the environment:

```bash
source env/bin/activate
```

After activating the environment, any packages installed will be restricted to that environment. To retrieve the list of packages installed in your environment, you can use the following command:

```bash
python3 -m pip list
```

These are the packages installed in the virtual enviroment when this guide was written:

```bash
Package         Version
--------------- --------
cycler          0.10.0
kiwisolver      1.2.0
llvmlite        0.33.0
matplotlib      3.3.0
numba           0.50.1
numpy           1.19.1
opencv-python   4.3.0.36
Pillow          7.2.0
pip             20.2
pyparsing       2.4.7
python-dateutil 2.8.1
scipy           1.5.2
setuptools      40.8.0
six             1.15.0
tqdm            4.48.0
```

You can now install the required packages using the following command:

```bash
python3 -m pip install <package name>
python3 -m pip install -r requirements.txt
```

During the installation of each package, replace the <package name> with the name of the package displayed in the list provided earlier.

To leave the virtual environment, just run the following command:

```bash
deactivate
```

## Usage

First, navigate to your workspace folder where the 'main', 'images' and 'results' folders are located. Then, activate your virtual environment created in the 'Installation' section:

```bash
source env/bin/activate
```

Once in the virtual environment, open the 'main' folder:

```bash
cd main
```

Now you can run the program using the following command:
```bash
python3 scEnergy.py -in <image filename (in /images/ folder)> -out <output filename> -scale <downsizing scale> -seam <seam orientation, v for vertical h for horizontal> -energy <energy algorithm (e.g. s for sobel)>
```

As for example:

```bash
python3 scEnergy.py -in image.jpg -out result.jpg -scale 0.5 -seam h -energy s
```

In case you need help with the parameters, just run the command below:

```bash
python3 scEnergy.py -h
```

```bash
usage: scEnergy.py [-h] -in IN -out OUT -scale SCALE -seam SEAM
                   [-energy ENERGY]

optional arguments:
  -h, --help      show this help message and exit
  -in IN          Path to input image
  -out OUT        Output image file name
  -scale SCALE    Downsizing scale. e.g. 0.5
  -seam SEAM      Seam orientation (h = horizontal seam, v = vertical seam
  -energy ENERGY  Energy algorithm (s = Sobel, p = Prewitt)
```

## License
[MIT](https://choosealicense.com/licenses/mit/)