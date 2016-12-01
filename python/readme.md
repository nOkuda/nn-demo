# Setting up Tensorflow

## Summary

Go to [Tensorflow's website](https://www.tensorflow.org), click on "Get
Started", then follow the directions under "Download and Setup".  The rest of
this document contains tips that will probably save you time.  I'm assuming
you're working on a lab computer.

## Lab Architecture

The lab is currently set up with multiple machines.  There are four categories
of machines:  the ones you sit at, the potatoes, the yams, and the others.  The
GPUs are set up on the yams, so if you want to use a GPU with tensorflow,
you'll want to be logged into one of the yams.

### Logging into a yam

You'll first need a lab account.  Log into one of the desktop machines and then
open terminal.

There are three yams.  They are names `0yam`, `1yam`, and `2yam`.  To log into
`2yam`, type
```
ssh 2yam
```
into the terminal and then press the `Enter` key.  Logging into the other yams
takes a similar form.

If you don't have ssh keys set up for the machines in the lab, you'll be
prompted to enter your password.  Once you're logged in, you should be in your
home directory (which is stored on guru, the lab file server).

## Setting up Your Environment

Tensorflow requires your environment to be set up in a certain way.  The
following is how I've set mine up in accordance with tensorflow's demands.

### Bash

To use the GPU, tensorflow needs CUDA to be installed.  The yams all have CUDA
installed, but your environment needs to be configured so that tensorflow
recognizes that CUDA is installed.  In my `~/.bash_profile`, I have the
following lines:
```
if [ -d "/usr/local/cuda" ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    PATH=$PATH:/usr/local/cuda/bin
fi

export PATH
```
Once you've got these lines in your `~/.bash_profile`, save the file and log
back into a yam.

#### Explanation

The first line checks whether CUDA is installed on the machine.  If so, the
following two lines are executed; otherwise, they aren't.  This allows me to
have one `~/.bash_profile` for my lab account instead of having to manually
load configuration files every time I log into a yam.

The second, third, and last lines set up the environment correctly for
tensorflow to recognize that CUDA has been installed.

##### Nitty Gritty Details

In case you were curious, `LD_LIBRARY_PATH` is the set of locations that the
compiler will look at to find library files.  The CUDA compiler needs to know
where the CUDA library files are stored, so the second line adds the CUDA
library files location to `LD_LIBRARY_PATH`.

Also for the curious, `PATH` is the set of locations bash searches for
executable binaries when a command is entered into the bash prompt.  The third
line adds the CUDA exectuables location to `PATH`, thus allowing for the CUDA
compiler to be called.

##### `~/.bash_profile` vs. `~/.bashrc`

You might have wondered why I put the above lines into my `~/.bash_profile`
instead of my `~/.bashrc`.  It is simply because I define my environment
variables in `~/.bash_profile`.  There are some funny rules about how bash
loads `~/.bash_profile` and `~/.bashrc`, so me some moons ago decided that it
would be best to stuff environment variables into `~/.bash_profile` instead of
`~/.bashrc`.  It also happens that my `~/.bash_profile` loads my `~/.bashrc`,
which means that regardless of funny rules, I could have placed my environment
variables in either file.  If you decide to put environment variables into
`~/.bashrc` tensorflow doesn't work, you may want to try putting your
environment variables into `~/.bash_profile`.

### Python

Now we're ready for installing tensorflow for Python.

#### Making a Virtual Environment

Choose a convenient place to set up a virtual environment.  I've chosen the
local directory on the yam I'm using (`/local/okuda/tf`).  Make that directory.
Now run
```
python3 -m venv <path to chosen directory>
```
This should create a virtual environment for your chosen directory.  Now
activate your virtual environment by running
```
source <path to chosen directory>/bin/activate
```
The terminal prompt will probably change, indicating that you are now in a
virtual environment.

#### Installing Tensorflow

While your virtual environment is active, you can install tensorflow via pip.
To do so, run
```
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0rc0-cp35-cp35m-linux_x86_64.whl
pip3 install --upgrade $TF_BINARY_URL
```
The first line sets the version of tensorflow you'll be downloading.  It is
version 0.12 with the GPU features enabled.

#### Installing Other Python Packages

You'll probably want a few more Python packages installed in your virtual
envrionment.  While your virtual environment is active, run
```
pip3 install <package name>
```
to install a package.  I like to have `numpy`, `scipy`, and `matplotlib`
installed.

#### Getting Out of a Virtual Environment

Assuming your virtual environment is still active and that you want your
virtual environment to no longer be active, you can deactivate your virtual
environment by running
```
deactivate
```

## Making Sure Tensorflow Works

To make sure that your tensorflow installation is working properly, start by
activating your virtual environment.  Then run
```
python3 <path to virtual environment directory>/lib64/python3.5/site-packages/tensorflow/models/image/mnist/convolutional.py
```
If the GPU is running properly, you should see some messages that say so.  I saw
```
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcurand.so locally
```
There was also some data download necessary
```
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting data/train-images-idx3-ubyte.gz
Extracting data/train-labels-idx1-ubyte.gz
Extracting data/t10k-images-idx3-ubyte.gz
Extracting data/t10k-labels-idx1-ubyte.gz
```
Then the code started running.
```
I tensorflow/core/common_runtime/gpu/gpu_device.cc:951] Found device 0 with properties: 
name: TITAN X (Pascal)
major: 6 minor: 1 memoryClockRate (GHz) 1.531
pciBusID 0000:01:00.0
Total memory: 11.90GiB
Free memory: 11.76GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:972] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Creating TensorFlow device (/gpu:0) -> (device: 0, name: TITAN X (Pascal), pci bus id: 0000:01:00.0)
```
The final lines I saw were
```
Step 8400 (epoch 9.77), 3.8 ms
Minibatch loss: 1.596, learning rate: 0.006302
Minibatch error: 0.0%
Validation error: 0.7%
Step 8500 (epoch 9.89), 3.8 ms
Minibatch loss: 1.620, learning rate: 0.006302
Minibatch error: 1.6%
Validation error: 0.9%
Test error: 0.8%
```
