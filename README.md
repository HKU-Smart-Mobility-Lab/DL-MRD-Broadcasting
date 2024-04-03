### Dynamic Matching Radius Decision Model for On-Demand Ride Services: A Deep Multi-Task Learning Approach

![效果图](https://img.shields.io/static/v1?label=build&message=passing&color=green) ![](https://img.shields.io/static/v1?label=python&message=3&color=blue) ![](https://img.shields.io/static/v1?label=release&message=2.0&color=green) ![](https://img.shields.io/static/v1?label=license&message=MIT&color=blue)

### Background

As ride-hailing services have experienced significant growth, most research has concentrated on the dispatching mode, where drivers must adhere to the platform's assigned routes. However, the broadcasting mode, in which drivers can freely choose their preferred orders from those broadcast by the platform, has received less attention. One important but challenging task in such a system is the determination of the optimal matching radius, which usually varies across space, time, and real-time supply/demand characteristics. This study develops a **D**eep **L**earning-based **M**atching **R**adius **D**ecision (DL-MRD) model that predicts key system performance metrics for a range of matching radii, which enables the ride-hailing platform to select an optimal matching radius that maximizes overall system performance according to real-time supply and demand information. To simultaneously maximize multiple system performance metrics for matching radius determination, we devise a novel multi-task learning algorithm named **W**eighted **E**xponential **S**moothing **M**ulti-task (WESM) learning strategy that enhances convergence speed of each task (corresponding to the optimization of one metric) and delivers more accurate overall predictions. We evaluate our methods in a simulation environment designed for broadcasting-mode-based ride-hailing service. Our findings reveal that dynamically adjusting matching radii based on our proposed approach significantly improves system performance.



### Contributions

• We make the first attempt to develop a deep learning-based approach to dynamically adjust the matching radius for system performance optimization in ride-hailing systems with the broadcasting mode.

• We develop a novel multi-task learning strategy to balance the tradeoff among the optimization for various system performance metrics in ride-hailing systems, including order fulfillment rate, driver utilization rate, average pickup distance, and platform revenue. The multi-task learning training strategy is shown to be effective in attaining much faster convergence speed and converging to a lower loss. 

• We have conducted extensive experiments based on a tailored simulation platform for broadcasting mode operations, which validate the effectiveness of our proposed Deep Learning-based Matching Radius Decision (DL-MRD) approach and multi-task training strategies.



<div id="pdfContainer"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/pdfobject/2.2.6/pdfobject.min.js"></script>
<script>
    PDFObject.embed("https://github.com/HKU-Smart-Mobility-Lab/DL-MRD-Broadcasting/blob/main/order-matching.pdf", "#pdfContainer");
</script>


### Install Simulator

1. Download the code

  `git clone git@github.com:HKU-Smart-Mobility-Lab/Transportation_Simulator.git`

2. Pull the docker image

  `docker pull jingyunliu663/simulator`

- after running the code, you can use `docker images` to check whether the image is available
- the docker image comes with the conda environment `new_simulator` and the mongoDB service running in background within the container

3. Run the docker image & get a docker container
```bash
docker run -d -e CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1 -v /path/to/the/Transportation_Simulator:/simulator/scripts --name simulator jingyunliu663/simulator
```
- Arguments:
  - `-d`: detach, run in background
  - `-v path/to/your/local/file/system:/path/to/your/target/directory/inside/container`: This will mount the local directory into the container. Any changes made to the files in this directory within the container will also be reflected on the local host system, and vice versa.
  - `--name`: name the container as *simulator*
  - the last argument is the image name (after all, container is a running instance of an image)

- you can use `docker ps` to check the running containers

4. Enter the interactive shell of the conatiner `simulator`
```bash
docker exec -it simultor /bin/bash
```

- After enter the interactive shell , you will be in the working directory `/simulator`, you can navigate yourself to  `/simulator/scripts` directory (the directory you choose to mount to) to run the main function
- You have to activate the conda environment: `conda activate new_simulator` 






###  Citing

If you use any part of this repo, you are highly encouraged to cite our paper:

Chen, T., Shen, Z., Feng, S., Yang, L., & Ke, J. (2023). Dynamic Adjustment of Matching Radii under the Broadcasting Mode: A Novel Multitask Learning Strategy and Temporal Modeling Approach. arXiv preprint arXiv:2312.05576.





##### 
