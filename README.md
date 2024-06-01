# gpunix

Welcome to gpunix, an fully-fledged Unix operating system that runs on the GPU. (Yes, you read that correctly)

> I'd just like to interject for a moment. What you're referring to as Unix, is in fact, GPU/Unix, or as I've recently taken to calling it, GPU plus Unix. Unix is not an operating system unto itself, but rather another component of a fully functioning GPU system made useful by the GPU kernels, memory management, and parallel processing capabilities comprising a full computational framework.

## Project Overview

gpunix is an esoteric project that aims to port the beloved Unix operating system (xv6) to run exclusively on CUDA-enabled devices. By harnessing the parallel processing capabilities of CUDA, we're taking Unix to a whole new level! Tested on a GTX 1060, CUDA compute capability 6.1.

> Many computer users run a modified version of the Unix system every day, without realizing it. Through a peculiar turn of events, the version of Unix which is widely used today is often called Unix, and many of its users are not aware that it is basically the GPU system, developed with the integration of GPU computing.

## Features

- Absolute exploitation of CUDA's supposed turing completeness
- Lightning-fast parallel processing for all your Unix tasks.
- Embrace the power of CUDA with familiar Unix commands.
- Experience the joy of debugging Unix on a GPU.

> There really is a Unix, and these people are using it, but it is just a part of the system they use. Unix is the software layer: the suite of tools and libraries that provide the interface and environment for running applications. This layer is essential for managing hardware resources, but it can only fully leverage the computational power of modern hardware when combined with GPU processing. Unix is normally used in combination with the GPU computational framework: the whole system is basically Unix with GPU added, or GPU/Unix. All the so-called Unix-based distributions are really distributions of GPU/Unix!

## Getting Started

To get started with gpunix, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/cuda-unix.git`
2. Install the necessary CUDA toolkit and drivers.
3. Build the gpunix kernel: `make cuda-kernel`
4. Launch gpunix: `./cuda-unix`

## License

This project is licensed under the [MIT License](LICENSE).

## Disclaimer

Please note that gpunix is an experimental project and should not be used in production environments. I don't even know how that would be possible, to be honest. Use at your own risk!
