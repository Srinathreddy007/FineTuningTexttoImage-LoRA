# Fine-tuning Text-to-Image with LoRA
## Abstract
This project presents an advanced application of the Stable Diffusion Model for image generation from text prompts. Involving a three-part architecture comprising a CLIP Text Encoder, Variational Autoencoder (VAE), and UNet2D, our model demonstrates enhanced adaptability and efficiency. The CLIP Text Encoder optimizes general text understanding through a Transformer architecture. The VAE encodes image pixel values into a latent space, facilitating the generation of noise to condition image synthesis. Finally, UNet predicts noise residuals to refine image outputs. To improve efficiency further, we incorporate LoRA layers within the UNet, allowing fine-tuning to specific text prompts while retaining pre-trained weights, reducing computational demands, and enhancing image fidelity. This approach combines foundational image generation methodology with advanced tactics found in more recent studies, showcasing the efficacy of language models and newer architectural innovations in text-to-image generation.


**The Slurm Output file contains the Model Architecture and Parameters**
</br>
<h2>Running the Code on HPC</h2> 
Follow these steps to run the code on a High-Performance Computing (HPC) cluster.

### 1. Navigate to the Scratch Folder
Create a directory in the `/scratch/your_net_id/` folder. Replace `your_net_id` with your actual net ID.

```bash
cd /scratch/your_net_id/

```
### 2. Clone the Git Repo
Clone the repository using the following command: 
``` bash
git clone https://github.com/Srinathreddy007/FineTuningTexttoImage-LoRA.git
```

### 3. Create a Conda Environment
Create a conda environment in the scratch folder and initialize it:
```bash
conda create --name /scratch/your_net_id/ENV_NAME python=3.10
conda init
```

Navigate to the project folder:
```bash
cd /scratch/your_net_id/FineTuningTexttoImage-LoRA/
```
### 4. Modify the submit.sbatch File to train the Model
There is an `submit.sbatch` file in the repository that needs to be modified to train the model on the HPC cluster. Open it using a text editor like vim or a command-line editor.

Change the line `--mail-user ` to receive updates about the model status:
```bash
#SBATCH --mail-user=netid@nyu.edu
```
In the `submit.sbatch` file, change the path to navigate to the folder:
```bash
cd /scratch/your_net_id/FineTuningTexttoImage-LoRA/
```
Replace the line that activates the environment:
```bash
source activate /scratch/your_net_id/ENV_NAME
```

Save the changes to the  `submit.sbatch` file.

### 5. Submit the Job
Before submitting the `submit.sbatch` file, download the `Diffusers library` directly from git repo, as the pip command downloads a previous version, which is not compatible with the code </br>
To do this, run the command:
```bash
git clone https://github.com/huggingface/diffusers.git
```
The the remaining downloads would be taken care by the requirements file </br>

Submit the `submit.sbatch` file for running:
```bash
sbatch submit.sbatch
```
Ensure that you have appropriate permissions and resources allocated on the HPC cluster before submitting the job.

The required libraries will be downloaded when you submit the `submit.sbatch` file. It contains a line `pip insall -r requirements.txt` that takes care of the necessary downloads. 

`Note: Replace your_net_id and ENV_NAME with your actual Net ID and environment name respectively.`

## 6. Generating Images from Prompts
There is a file `generate.py` that contains code to load the trained model weights and generate images from imput prompts. </br>
The file is run using `test.sbatch` file </br>

Navigate to the project folder:
```bash
cd /scratch/your_net_id/FineTuningTexttoImage-LoRA/
```
### Modify the test.sbatch File to train the Model
There is an `test.sbatch` file in the repository that needs to be modified to generate images. Open it using a text editor like vim or a command-line editor.

Change the line `--mail-user ` to receive updates about the model status:
```bash
#SBATCH --mail-user=netid@nyu.edu
```
In the `test.sbatch` file, change the path to navigate to the folder:
```bash
cd /scratch/your_net_id/FineTuningTexttoImage-LoRA/
```
Replace the line that activates the environment:
```bash
source activate /scratch/your_net_id/ENV_NAME
```
You can change the prompt by changing the `--prompt` arguement </br>
Save the output directory folder to your directory

Save the changes to the  `test.sbatch` file. </br>
Run the file:
```bash
sbatch test.sbatch
```

Note: Checkpoint files have not been uploaded to the Git Repo as the files are very large. </br>

## Caution: The GPU on local systems/ jupyter notebook would not be sufficient to load the model, hence it is adviced to run on HPC Cluster </br>

## Running the Code on the Local System
### 1. Install the anaconda 2024.02version to run without compatibility issues. 

### 2. Clone the Git Repo
Clone the repository using the following command: 
``` bash
git clone https://github.com/Srinathreddy007/FineTuningTexttoImage-LoRA.git
```
### 3. Create a Conda Environment
Navigate to the destination folder. Create the conda environment using the `.yml` file provided in the repo and activate the environment.
```bash
conda create --name /path/to/your/folder/ENV_NAME --file gpu_env.yml
conda activate /path/to/your/folder/ENV_NAME 
```
Then navigate to the destination folder. Create the conda environment with `python=3.10`
```bash
conda create --name /path/to/your/folder/ENV_NAME python=3.10
conda activate /path/to/your/folder/ENV_NAME 
```

### 4. Install Necessary Libraries
Before running the requirements file, download the `Diffusers Library` from the git repo
```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
cd ..

```
Run the command. The `requirements.txt` file is also provided in the repo. 
```bash
pip install -r requirements.txt
```

### 5. Run the Code:
Run the `main.py` file using the command:
```bash
python main.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --dataset_name="nlphuji/flickr30k" --output_dir="/scratch/ap8638/Train_DL/Train-Results/Train-3" --cache_dir="/scratch/ap8638/Train_DL/output/cache/" --logging_dir="/scratch/ap8638/Train_DL/output/logs"  --learning_rate=1e-4 --num_train_epochs=1   
```


 ### 6. Generate the Images from Prompts
 Run the `generate.py` file using the command:
 ```bash
python generate.py --prompt='a futuristic cityscape at sunset' --model_path='/scratch/ap8638/Train_DL/Train-Results/Train-2/pytorch_lora_weights.safetensors' --steps=25
```

`Don't change the command-line arguments, as they are the parameters set for replicating the best model. But feel free to change the paths as necessary`

 ## Contributors:
 <ul>
  <li> Sai SrinathReddy Sivareddy(ss18364)</li>
  <li>Akash Peddaputha (ap8638)</li>
  <li>Asad Anjum(aa12277)</li>
 </ul>
 
## Acknowledgements
 We used online resources like StackOverflow, GitHub repos, HuggingFace Tutorials, Kaggle Competition entries, ChatGPT and official PyTorch, numpy, matplotlib documentation and recitations by Karthik Garimella. 

## System Specifications:
Used NYU HPC Cluster to train the model
<ul>
    <li>
        <strong>NYU Greene HPC VM Specifications:</strong>
        <ul>
            <li><strong>CPU:</strong> 8 Virtualized Cores of Intel Xeon-Platinum 8286</li>
            <li><strong>GPU:</strong> Nvidia Quadro RTX 8000 and Nvidia V100</li>
            <li><strong>System Memory:</strong> 96 GB</li>
            <li><strong>Python Version:</strong> 3.10.14</li>
            <li><strong>CUDA Version:</strong> v12.1</li>
            <li><strong>Torch Version:</strong> 2.3.0</li>
        </ul>
    </li>
</ul>






