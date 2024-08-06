import os

# Algorithm list
ALGORITHM_LIST = ['naive_rag', 'query_rewrite_rag',  'iterative_rag', 'self_ask', 'active_rag','selfrag_reproduction']

# Configuration files directory
config_dir_base = "./config"

# Run scripts directory
run_dir = "./run/rag_inference"

# Create the run scripts directory (if it doesn't exist)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

# List to store generated script paths
script_paths = []

# Iterate over the algorithm list
for algorithm_name in ALGORITHM_LIST:
    config_dir = os.path.join(config_dir_base, algorithm_name)
    
    # Create the algorithm directory inside run_dir (if it doesn't exist)
    algorithm_run_dir = os.path.join(run_dir, algorithm_name)
    if not os.path.exists(algorithm_run_dir):
        os.makedirs(algorithm_run_dir)

    # Iterate over the configuration files directory
    for filename in os.listdir(config_dir):
        if filename.endswith(".yaml"):
            # Get the filename (without extension)
            basename = os.path.splitext(filename)[0]

            # Generate the Shell script filename
            script_filename = os.path.join(algorithm_run_dir, f"{basename}.sh")

            # Determine the Python file to execute
            if "interact" in filename:
                python_file = "main-interact.py"
            else:
                python_file = "main-evaluation.py"

            # Generate the Shell script content
            script_content = f"# export CUDA_VISIBLE_DEVICES=1\n"
            script_content += f"python ./{python_file}\\\n"
            script_content += f" --config ./config/{algorithm_name}/{filename}"

            # Write the Shell script file
            with open(script_filename, "w") as script_file:
                script_file.write(script_content)

            print(f"Generated {script_filename}")

            # Append the script path to the list
            # "interact" not in filename
            # if  '70B' in filename and 'pregiven_passages' not in filename and 'interact' not in filename:
            # if '4bit' in filename and all(substring not in filename for substring in ['pregiven_passages']):
            #     script_paths.append(f"sh {script_filename}")
            script_paths.append(f"sh {script_filename}")

# Write the script paths to a text file
with open("./auto_gpu_scheduling_scripts/auto_run-scripts.py", "w") as txt_file:
    txt_file.write("\n".join(script_paths))

print("Generated success!!!")


