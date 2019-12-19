# Affect-based Intrinsic Rewards for Learning General Representations  
Official code repository for [https://arxiv.org/abs/1912.00403](https://arxiv.org/abs/1912.00403).  
The work on this repository is still in progress and will be done by the end of 12/2019.  
    
### System Requirements  
  
* Operating system - Windows 10 or Ubuntu 18.04.  
* GPU - Nvidia GTX1080 or higher is recommended (although some of the research was done using GTX1060).  
  
### Installation  
  
The project is based on Python 3.6 and TensorFlow 2.0. All the necessary packages are in requirements.txt. We recommend creating virual environment using Anaconda as follows:  
  
1) Download and install Anaconda Python from here:  
[https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)  
  
2) Enter the following commands to create a virtual environment:  
```
conda create -n tf36 python=3.6 anaconda
activate tf36
pip install -r requirements.txt
```
  
For more information on how to manage conda environments, please refer to:  
[https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)  
  
### Simulation  
  
For the simulation environment, we used [AirSim](https://github.com/Microsoft/AirSim).  
Since the compiled simulation is too heavy to upload, we encourage you to compile AirSim and play with it.  
To assemble the exact environment we had been working on, we will soon upload our map and JSON files here.  
  
### Training  
  
**Imitation learning**  
  
To record the data for imitation learning, the following script will drive and record data simultaneously. All you need to do is to run the following script while the simulation is running. Here is an execution example:  
```  
python utils/record_cam_driving.py --camera rgb --dest_dir C:\\Users\\user\\Documents  
```  
The script let the car drive and expect you to instruct the desired action using the keys a, w, s, e and d. Additional options for ground truth can be found in the script.  
  
Next is preparing the data for training. All the data preprocessing functions are under the same file.  
  
### Citing    
  
If this repository helped you in your research, please consider citing:  
```  
@article{zadok2019affect,
  title={Affect-based Intrinsic Rewards for Learning General Representations},
  author={Zadok, Dean and McDuff, Daniel and Kapoor, Ashish},
  journal={arXiv preprint arXiv:1912.00403},
  year={2019}
}
```  
  
### Acknowledgments  
  
We thank the subjects that helped us creating the dataset for the affect-based reward model, and the rest of the team for the help with the simulation and the experiments.  
