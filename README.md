# Seam Carving

Content-Aware Scaling using Seam Carving method with different algorithms for energy mapping.

Quick reference: https://en.wikipedia.org/wiki/Seam_carving

Demo:https://www.aryan.app/seam-carving/

## Context

The program was developed and tested on a MacBook Pro 13-inch running macOS 10.15.6. The same bash commands should work on a machine running a unix-based OS.

## Installation

It is recommended to use the program in a virtual environment. To create a virtual environment, the virtualenv package has to be installed:

```bash
python3 -m pip install virtualenv
```

After that, navigate to the folder where the 'src' folder for the scripts is located. Then we can create the virtual environment using the following command:

```bash
python3 -m venv env
```

You'll notice that the 'env' folder was created in your workspace. Now we need to activate the environment:

```bash
source env/bin/activate
```

To install the required packages, use the following command:

```bash
python3 -m pip install -r requirements.txt
```

To leave the virtual environment, just run the following command:

```bash
deactivate
```

## Usage

First, navigate to your workspace folder where the 'src', 'images' folders are located. Then, activate your virtual environment created in the 'Installation' section:

```bash
source env/bin/activate
```

Once in the virtual environment, open the 'src' folder:

```bash
cd src
```

Now you can run the program using the following command:
```bash
python3 sc.py -in <image filename (in /images/ folder)> -scale <downsizing scale> -seam <seam orientation, v for vertical h for horizontal> -energy <energy algorithm (e.g. s for sobel)>
```

As for example:

```bash
python3 sc.py -in image.jpg -scale 0.5 -seam h -energy s
```

In case you need help with the parameters, just run the command below:

```bash
python3 sc.py -h
```

```bash
usage: sc.py [-h] -in IN -scale SCALE -seam SEAM [-energy ENERGY]

optional arguments:
  -h, --help      show this help message and exit
  -in IN          Path to input image
  -scale SCALE    Downsizing scale. e.g. 0.5
  -seam SEAM      Seam orientation (h = horizontal seam, v = vertical seam
  -energy ENERGY  Energy algorithm (all = All existing algorithms, s = Sobel, p = Prewitt, l = Laplacian, r = Roberts)
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
