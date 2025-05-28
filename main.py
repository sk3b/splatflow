import gradio as gr
from model3d import Model3D
import os
import subprocess
import shutil
import tempfile
import atexit
import signal
import json

# Configuration
CONFIG = {
    "build_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "splatflow", "build"),
    "supported_video_extensions": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
    "supported_image_extensions": [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"]
}

# Keep track of temporary directories to clean up
temp_dirs = set()

def cleanup_temp_dirs():
    """Clean up all temporary directories."""
    for temp_dir in temp_dirs:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
    temp_dirs.clear()

# Register cleanup function to run on normal exit
atexit.register(cleanup_temp_dirs)

# Register cleanup function to run on SIGTERM
signal.signal(signal.SIGTERM, lambda signum, frame: cleanup_temp_dirs())

def validate_video_file(video_path):
    """Validate if the video file exists and has a supported extension."""
    if not os.path.exists(video_path):
        return False, "Video file does not exist."
    
    ext = os.path.splitext(video_path)[1].lower()
    if ext not in CONFIG["supported_video_extensions"]:
        return False, f"Unsupported video format. Supported formats: {', '.join(CONFIG['supported_video_extensions'])}"
    
    return True, "Video file is valid."

def extract_frames_from_video(video_path, output_dir, num_frames=50, min_buffer=1):
    """
    Extract frames from a video using sharp-frames tool.
    Returns the path to the directory containing extracted frames.
    """
    try:
        # Validate video file
        is_valid, message = validate_video_file(video_path)
        if not is_valid:
            return None, message
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Run sharp-frames command
        cmd = f"sharp-frames {video_path} {output_dir} --num-frames {num_frames} --min-buffer {min_buffer}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            return None, f"Error extracting frames: {result.stderr}"
            
        # Verify that frames were extracted
        extracted_frames = [f for f in os.listdir(output_dir) if f.lower().endswith(tuple(CONFIG["supported_image_extensions"]))]
        if not extracted_frames:
            return None, "No frames were extracted from the video."
            
        return output_dir, f"Successfully extracted {len(extracted_frames)} frames."
    except Exception as e:
        return None, f"Error during frame extraction: {str(e)}"

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

    # Verify build directory exists
    if not os.path.exists(CONFIG["build_dir"]):
        add_log(f"Error: Build directory not found at {CONFIG['build_dir']}")
        return "\n".join(log)

    # Verify opensplat executable exists
    opensplat_path = os.path.join(CONFIG["build_dir"], "opensplat")
    if not os.path.exists(opensplat_path):
        add_log(f"Error: opensplat executable not found at {opensplat_path}")
        return "\n".join(log)

    # Run opensplat command
    opensplat_cmd = f"./opensplat {workspace} -n {iterations}"
    add_log("=== Training Gaussian Splat ===")
    add_log(f"Running: {opensplat_cmd} in directory {CONFIG['build_dir']}")

    result = subprocess.run(opensplat_cmd, shell=True, capture_output=True, text=True, cwd=CONFIG["build_dir"])
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
    ply_files = [f for f in os.listdir(CONFIG["build_dir"]) if f.endswith(".ply")]
    if ply_files:
        # Assume the most recently created .ply file is the one we want.
        ply_file = max(
            [os.path.join(CONFIG["build_dir"], f) for f in ply_files],
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

def create_workspace_from_images(images):
    """
    Creates a temporary workspace with the uploaded images.
    Returns the path to the temporary workspace.
    """
    # Create a temporary directory for the workspace
    workspace = tempfile.mkdtemp()
    temp_dirs.add(workspace)  # Add to set of directories to clean up
    images_folder = os.path.join(workspace, "images")
    os.makedirs(images_folder, exist_ok=True)
    
    # Save uploaded images to the workspace
    for img in images:
        if img is not None:
            # Get the original filename or generate one if not available
            filename = getattr(img, 'name', None)
            if filename is None:
                filename = f"image_{len(os.listdir(images_folder))}.jpg"
            else:
                filename = os.path.basename(filename)
            
            # Save the image
            img_path = os.path.join(images_folder, filename)
            shutil.copy2(img.name, img_path)
    
    return workspace

def create_workspace_from_video(video_file, num_frames=50, min_buffer=1):
    """
    Creates a temporary workspace with frames extracted from the uploaded video.
    Returns the path to the temporary workspace.
    """
    # Create a temporary directory for the workspace
    workspace = tempfile.mkdtemp()
    temp_dirs.add(workspace)  # Add to set of directories to clean up
    images_folder = os.path.join(workspace, "images")
    os.makedirs(images_folder, exist_ok=True)
    
    # Extract frames from video
    frames_dir, status = extract_frames_from_video(video_file.name, images_folder, num_frames, min_buffer)
    if frames_dir is None:
        raise Exception(status)
    
    return workspace

def process_workflow(input_type, input_files, scaling_option, matching_type, iterations, num_frames=50, min_buffer=1):
    """
    This function processes either uploaded images or a video through the entire workflow:
    1. Creates a temporary workspace with the input
    2. Scales the images if needed
    3. Runs COLMAP
    4. Trains Gaussian Splats
    5. Returns the generated .ply file
    """
    if input_type == "Video" and not input_files:
        return None, "No video was uploaded. Please upload a video first."
    elif input_type == "Images" and (not input_files or len(input_files) == 0):
        return None, "No images were uploaded. Please upload some images first."
    
    try:
        # Create workspace from uploaded files
        if input_type == "Video":
            workspace = create_workspace_from_video(input_files[0], num_frames, min_buffer)
        else:
            workspace = create_workspace_from_images(input_files)
        
        # Run image scaling if needed
        scale_result = scale_images(workspace, scaling_option)
        if "Error" in scale_result:
            return None, scale_result
        
        # Run COLMAP
        colmap_result = run_colmap(workspace, matching_type)
        if "Error" in colmap_result:
            return None, colmap_result
        
        # Run Gaussian Splat training
        splat_result = run_gaussian_splat(workspace, iterations)
        if "Error" in splat_result:
            return None, splat_result
        
        # Find the generated .ply file
        ply_file = None
        for file in os.listdir(workspace):
            if file.endswith('.ply'):
                ply_file = os.path.join(workspace, file)
                break
        
        if ply_file is None:
            return None, "No .ply file was generated during processing."
        
        # Create a copy of the .ply file in a new temporary directory that won't be cleaned up immediately
        preview_dir = tempfile.mkdtemp()
        temp_dirs.add(preview_dir)
        preview_ply = os.path.join(preview_dir, os.path.basename(ply_file))
        shutil.copy2(ply_file, preview_ply)
        
        return preview_ply, "Processing completed successfully!"
        
    except Exception as e:
        return None, f"An error occurred: {str(e)}"
    finally:
        # Clean up temporary workspace
        if 'workspace' in locals():
            try:
                shutil.rmtree(workspace, ignore_errors=True)
                temp_dirs.discard(workspace)  # Remove from set if cleanup successful
            except Exception:
                pass  # If cleanup fails, the atexit handler will try again

# Build a single-page Gradio interface.
with gr.Blocks() as demo:
    gr.Markdown("# Splatflow")
    gr.Markdown(
        "Upload your images or video to create a 3D Gaussian Splat model."
    )
    
    with gr.Row():
        with gr.Column():
            input_type = gr.Radio(
                choices=["Images", "Video"],
                label="Input Type",
                value="Images"
            )
            
            image_input = gr.File(
                label="Upload Images",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
                visible=True
            )
            
            video_input = gr.File(
                label="Upload Video",
                file_types=["video"],
                type="filepath",
                visible=False
            )
            
            num_frames = gr.Slider(
                minimum=10,
                maximum=200,
                step=10,
                value=50,
                label="Number of Frames to Extract",
                visible=False
            )
            
            min_buffer = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=1,
                label="Minimum Buffer Between Frames",
                visible=False
            )
            
            scaling_input = gr.Radio(
                choices=["No Scaling", "Half", "Quarter", "Eighth", "1600k"],
                label="Image Scaling Option",
                value="No Scaling"
            )
            matching_input = gr.Radio(
                choices=["Exhaustive", "Sequential", "Spatial"],
                label="COLMAP Feature Matching Type",
                value="Exhaustive"
            )
            iterations_input = gr.Slider(
                minimum=100,
                maximum=10000,
                step=100,
                value=2000,
                label="Gaussian Splat Training Iterations"
            )
            
            run_button = gr.Button("Create Model")
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=3
            )
        
        with gr.Column():
            model_preview = Model3D(  # Use the custom Model3D component
                label="3D Model Preview (Output Only)",
                height=500,
                interactive=False,
                camera_position=(0, 0, 2)
            )
    
    def update_input_visibility(input_type):
        return {
            image_input: gr.update(visible=input_type == "Images"),
            video_input: gr.update(visible=input_type == "Video"),
            num_frames: gr.update(visible=input_type == "Video"),
            min_buffer: gr.update(visible=input_type == "Video")
        }
    
    input_type.change(
        fn=update_input_visibility,
        inputs=[input_type],
        outputs=[image_input, video_input, num_frames, min_buffer]
    )
    
    def process_input(input_type, image_files, video_file, scaling, matching, iterations, frames, buffer):
        if input_type == "Video":
            if not video_file:
                return None, "Please upload a video file first."
            return process_workflow(input_type, [video_file], scaling, matching, iterations, frames, buffer)
        else:
            if not image_files:
                return None, "Please upload some images first."
            return process_workflow(input_type, image_files, scaling, matching, iterations)
    
    run_button.click(
        fn=process_input,
        inputs=[
            input_type,
            image_input,
            video_input,
            scaling_input,
            matching_input,
            iterations_input,
            num_frames,
            min_buffer
        ],
        outputs=[model_preview, status_output]
    )

demo.launch(share=False)
