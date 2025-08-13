# Autocut V2

## Instructions for Creating and Using `config.yml`

To set up your configuration file for Autocut V2, follow these steps:

1. **Locate the `config_template.yml` in the root directory.**  
   This file serves as the starting point for your configuration.

2. **Copy and rename it to `config.yml`.**  
   You can do this using the command line or your file explorer:
   ```bash
   cp config_template.yml config.yml
   ```

3. **Customize fields based on your needs.**  
   - **Input/Output Settings:**  
     - `input_video`: Path to your input video file.
     - `output_dir`: Directory where you want the output files to be saved.
   
   - **Execution Settings:**  
     - `device`: Specify the device to run the tool (e.g., CPU, GPU).
     - `num_workers`: Set the number of workers for processing (adjust based on your system).
   
   - **Workflow Settings:**  
     Customize the workflow options to suit your project requirements.
   
   - **Normalization & Scene Detection Settings:**  
     Configure these settings depending on your video content and processing needs.
   
   - **Criteria Settings:**  
     - `nsfw`: Filter for not-safe-for-work content.
     - `face`: Enable/disable face detection.
     - Additional criteria can be added as needed.

4. **Save it in the root directory.**  
   Ensure that your `config.yml` file is saved in the root of your project directory.

5. **Examples of Running the Tool:**  
   - **Using CLI:**  
     Run the following command in your terminal:
     ```bash
     python autocut.py --config config.yml
     ```  
   - **Using Python API:**  
     You can also run the tool programmatically:
     ```python
     from autocut import Autocut
     autocut = Autocut(config='config.yml')
     autocut.run()
     ```

By following these instructions, you will be able to configure and run Autocut V2 effectively!