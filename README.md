

# Image Denoising Techniques using CUDA C++

> The image denoising project involves optimizing a CUDA-based K-Nearest and Wiener filter implementation for image denoising, focusing on enhancing performance through efficient shared memory utilization and validating effectiveness using quantitative measures such as PSNR. Detailed performance analysis are conducted using NVIDIA Nsight Systems to ensure the filter not only reduces noise effectively but also operates at optimal speed and efficiency.


## Built With

- CUDA C++
- C++
- Python

## Tools used

- NVIDIA Nsight Compute
- VSCODE

## GPU Architecture Used

- Hopper H100


## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites
The set-up for this project is on a PBS(Portable Batch System) for job scheduling. Make sure you are not on a non-supported system like SLURM. 
The project can however work on other systems like slurm based system. You have to make sure that your job submission is supported. In that case, use the main.cu file in your setup.

## Configuration and set-up

- Kindly ensure that your environments supports the following:
  + CUDA C++
  + C++
  + Python

- Cloning the repository
  - ```
    https://github.com/richardoppiyo/imageDenoiding_cpe613Project.git
    ```

- Navigate to the project's root folder to see the different version:
  - ```
    cd imageDenoiding_cpe613Project
    ```

- To run a particular version, navigate into its folder. Example below:
  - ```
    cd modifiedKNN/
    ```
    Once you are in this folder, run the job as follows:

    ```
        ./runjob.sh
    ```

    output file will be created where you can view the performance information and outputImage.txt will also be generated, which is processed further to get teh denoised image

### Run tests
- To run a test version, navigate into nsight folder and to the version folder. Example below:
  - ```
    cd nsight/modifiedKNN/

    ```
    Once you are in this folder, run the job as follows:

    ```
        ./runjob.sh
    ```

    In addition to the output files, profile.ncu-rep and report1.nsys-rep will be generated which can be used in the Nsight compute system for performnce visualization


## Author

👤 **Richard Opiyo Omenyo**

- GitHub: [@richaroppiyo](https://github.com/richardoppiyo)
- Twitter: [@blessed_ricky](https://twitter.com/blessed_ricky)
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/richardoppiyo/)


## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

Feel free to check the [issues page](../../issues/).

## Show your support

Give a ⭐️ if you like this project!

## Acknowledgments

- Dr. Wise, my Instructor for CPE613 at the University of Alabama.
- Hat tip to anyone whose code was used
- Inspiration
- etc

## 📝 License

This project is [MIT](./MIT.md) licensed.
