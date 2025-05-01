import gradio as gr
import os
import subprocess
import shutil

def scale_images(workspace, scaling_option):
    """
    If scaling_option is not "No Scaling", this function renames the existing 
    "images" folder to "images_original" and creates a new "images" folder where 
    each image is resized according to the scaling option.
    
    Scaling options:
      - Half: 50% size
      - Quarter: 25% size
      - Eighth: 12.5% size
      - 1600k: longest dimension set to 1600 pixels (only downscale if larger)
    """
    workspace = os.path.abspath(workspace)
    images_folder = os.path.join(workspace, "images")
    original_folder = os.path.join(workspace, "images_original")

    if not os.path.exists(images_folder):
        return f"Error: The images folder was not found at {images_folder}"

    if scaling_option == "No Scaling":
        return "No scaling selected. Using original images."

    # Prevent accidental overwrite if images_original already exists.
    if os.path.exists(original_folder):
        return f"Error: {original_folder} already exists. Please remove or rename it before scaling."

    # Rename original images folder and create a new one.
    os.rename(images_folder, original_folder)
    os.makedirs(images_folder, exist_ok=True)

    supported_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"]
    files_processed = 0

    for filename in os.listdir(original_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext in supported_extensions:
            input_path = os.path.join(original_folder, filename)
            output_path = os.path.join(images_folder, filename)
            if scaling_option in ["Half", "Quarter", "Eighth"]:
                scale_map = {"Half": "50%", "Quarter": "25%", "Eighth": "12.5%"}
                resize_value = scale_map[scaling_option]
                # Using ImageMagick's convert command to resize.
                cmd = f'convert "{input_path}" -resize {resize_value} "{output_path}"'
            elif scaling_option == "1600k":
                # The '1600x1600>' forces the longest dimension to 1600 pixels (if larger).
                cmd = f'convert "{input_path}" -resize 1600x1600\> "{output_path}"'
            else:
                continue

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                return f"Error processing {filename}: {result.stderr}"
            files_processed += 1

    return f"Processed {files_processed} images.\nNew images in: {images_folder}\nOriginal images in: {original_folder}"

def run_colmap(workspace, matching_type):
    """
    Runs COLMAP steps using the provided workspace directory and feature matching type.
    Assumes the workspace has an 'images' folder.
    """
    workspace = os.path.abspath(workspace)
    images_folder = os.path.join(workspace, "images")
    db_path = os.path.join(workspace, "database.db")
    sparse_path = os.path.join(workspace, "sparse")
    log = []

    def add_log(msg):
        print(msg)  # Print to console for debugging.
        log.append(msg)

    add_log(f"Workspace: {workspace}")

    if not os.path.exists(images_folder):
        add_log(f"Error: The images folder was not found at {images_folder}")
        return "\n".join(log)

    os.makedirs(sparse_path, exist_ok=True)

    def run_command(cmd, description):
        add_log(f"=== {description} ===")
        add_log(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            add_log("STDOUT:")
            add_log(result.stdout)
        if result.stderr:
            add_log("STDERR:")
            add_log(result.stderr)
        return result.returncode

    # COLMAP step 1: Create the database.
    ret = run_command(f"colmap database_creator --database_path {db_path}", "Creating database")
    if ret != 0:
        add_log("Error during database creation.")
        return "\n".join(log)

    # COLMAP step 2: Feature extraction.
    ret = run_command(f"colmap feature_extractor --database_path {db_path} --image_path {images_folder}", "Extracting features")
    if ret != 0:
        add_log("Error during feature extraction.")
        return "\n".join(log)

    # COLMAP step 3: Feature matching.
    matcher_cmd = {
        "Exhaustive": f"colmap exhaustive_matcher --database_path {db_path}",
        "Sequential": f"colmap sequential_matcher --database_path {db_path}",
        "Spatial": f"colmap spatial_matcher --database_path {db_path}"
    }
    ret = run_command(matcher_cmd[matching_type], f"Running {matching_type} matching")
    if ret != 0:
        add_log("Error during feature matching.")
        return "\n".join(log)

    # COLMAP step 4: Sparse reconstruction.
    ret = run_command(f"colmap mapper --database_path {db_path} --image_path {images_folder} --output_path {sparse_path}", "Running sparse reconstruction")
    if ret != 0:
        add_log("Error during sparse reconstruction.")
        return "\n".join(log)

    # Verify the expected output.
    result_dir = os.path.join(sparse_path, "0")
    expected_files = [os.path.join(result_dir, fname) for fname in ["cameras.bin", "images.bin", "points3D.bin"]]

    if all(os.path.exists(f) for f in expected_files):
        add_log("COLMAP reconstruction completed successfully!")
        add_log(f"Results are in: {result_dir}")
    else:
        add_log("Error: COLMAP did not generate the expected output files.")
        for f in expected_files:
            add_log(f"{f}: {'Found' if os.path.exists(f) else 'Not found'}")

    return "\n".join(log)

def run_gaussian_splat(workspace, iterations):
    """
    Runs the opensplat command to train Gaussian Splats using COLMAP output and moves the generated .ply file to the workspace directory.
    """
    workspace = os.path.abspath(workspace)
    log = []

    def add_log(msg):
        print(msg)  # Print to console for debugging.
        log.append(msg)

    # Check if COLMAP output exists.
    sparse_dir = os.path.join(workspace, "sparse", "0")
    expected_files = [os.path.join(sparse_dir, fname) for fname in ["cameras.bin", "images.bin", "points3D.bin"]]
    if not all(os.path.exists(f) for f in expected_files):
        add_log(f"Error: COLMAP output files not found in {sparse_dir}.")
        return "\n".join(log)

    # Run opensplat command from the specific build directory.
    build_dir = "/Users/shubham/Documents/GitHub/gsplat/OpenSplat/build"
    opensplat_cmd = f"./opensplat {workspace} -n {iterations}"
    add_log("=== Training Gaussian Splat ===")
    add_log(f"Running: {opensplat_cmd} in directory {build_dir}")

    result = subprocess.run(opensplat_cmd, shell=True, capture_output=True, text=True, cwd=build_dir)
    if result.stdout:
        add_log("STDOUT:")
        add_log(result.stdout)
    if result.stderr:
        add_log("STDERR:")
        add_log(result.stderr)

    if result.returncode != 0:
        add_log("Error during Gaussian Splat training.")
        return "\n".join(log)

    # Find and move the .ply file from build_dir to workspace.
    ply_files = [f for f in os.listdir(build_dir) if f.endswith(".ply")]
    if ply_files:
        # Assume the most recently created .ply file is the one we want.
        ply_file = max(
            [os.path.join(build_dir, f) for f in ply_files],
            key=os.path.getctime
        )
        output_ply = os.path.join(workspace, os.path.basename(ply_file))
        shutil.move(ply_file, output_ply)
        add_log("Gaussian Splat training completed successfully!")
        add_log(f"Output .ply file: {os.path.basename(output_ply)}")
        add_log(f"Location: {workspace}")
    else:
        add_log("Error: No .ply file found in the build directory after training.")
        return "\n".join(log)

    return "\n".join(log)

def process_workflow(workspace, scaling_option, matching_type, iterations):
    """
    This function first scales the images (unless 'No Scaling' is selected),
    runs COLMAP on the workspace, and then trains Gaussian Splats to produce a .ply file.
    """
    logs = []
    logs.append("=== Image Preparation ===")
    # Run image scaling if needed.
    scale_result = scale_images(workspace, scaling_option)
    logs.append(scale_result)
    
    logs.append("\n=== Running COLMAP Reconstruction ===")
    colmap_result = run_colmap(workspace, matching_type)
    logs.append(colmap_result)
    
    logs.append("\n=== Running Gaussian Splat Training ===")
    gaussian_result = run_gaussian_splat(workspace, iterations)
    logs.append(gaussian_result)
    
    return "\n".join(logs)

# Build a single-page Gradio interface.
with gr.Blocks() as demo:
    gr.Markdown("# COLMAP Workflow with Image Scaling and Gaussian Splat Training")
    gr.Markdown(
        "Provide the workspace directory (which must contain an `images` folder). "
        "Choose a scaling option and a COLMAP feature matching method. "
        "If a scaling option other than 'No Scaling' is selected, the current "
        "`images` folder will be renamed to `images_original` and resized images "
        "will be saved in a new `images` folder. COLMAP is then run on these images. "
        "Finally, Gaussian Splat training is performed to generate a .ply file."
    )
    
    workspace_input = gr.Textbox(label="Workspace Directory", placeholder="/path/to/workspace")
    scaling_input = gr.Radio(choices=["No Scaling", "Half", "Quarter", "Eighth", "1600k"],
                              label="Image Scaling Option",
                              value="No Scaling")
    matching_input = gr.Radio(choices=["Exhaustive", "Sequential", "Spatial"],
                               label="COLMAP Feature Matching Type",
                               value="Exhaustive")
    iterations_input = gr.Slider(minimum=100, maximum=10000, step=100, value=2000,
                                 label="Gaussian Splat Training Iterations")
    
    run_button = gr.Button("Run Workflow")
    output_log = gr.Textbox(label="Processing Log", lines=25)

    run_button.click(fn=process_workflow,
                     inputs=[workspace_input, scaling_input, matching_input, iterations_input],
                     outputs=output_log)

demo.launch()