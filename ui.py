# ui.py
import os
import sys
import time
import json
import subprocess
import streamlit as st
from pathlib import Path

# Use an absolute path to the project's root directory.
ROOT_DIR = Path(__file__).resolve().parent

# --- THE FIX: Check for both 'Output' and 'output' to be robust ---
def get_output_dir():
    """Finds the correct output directory, handling case differences."""
    dir_upper = ROOT_DIR / "Output"
    dir_lower = ROOT_DIR / "output"
    if dir_upper.exists():
        return dir_upper
    return dir_lower

# --- Helper Functions (Updated to use the new function) ---

def get_projects():
    """Scans the Output directory for project folders."""
    output_dir = get_output_dir()
    if not output_dir.exists():
        return []
    return [p.name for p in output_dir.iterdir() if p.is_dir() and (p / "source_assets").exists()]

def get_runs(project_name):
    """Scans a project's runs directory."""
    if not project_name: return []
    output_dir = get_output_dir()
    runs_dir = output_dir / project_name / "runs"
    if not runs_dir.exists():
        return []
    return sorted([r.name for r in runs_dir.iterdir() if r.is_dir()], reverse=True)

def run_command_and_display_output(command, log_placeholder):
    """
    Runs a command in a sanitized subprocess and displays its live output.
    """
    log_lines = []
    log_placeholder.code("Executing command: " + " ".join(command), language="log")
    
    env = os.environ.copy()
    for key in list(env.keys()):
        if key.startswith("STREAMLIT_"):
            del env[key]
    env['PYTHONIOENCODING'] = 'utf-8'

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1,
        env=env
    )

    for line in iter(process.stdout.readline, ''):
        log_lines.append(line.strip())
        log_placeholder.code('\n'.join(log_lines), language="log")

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command, output='\n'.join(log_lines))

# --- Callback Function for Production ---
def trigger_production_run():
    """
    This function is called ONLY when the 'Start Production!' button is clicked.
    """
    project = st.session_state.selected_project
    run_name = st.session_state.run_name_input
    use_gpu = st.session_state.use_gpu_input
    no_crop = st.session_state.no_crop_input
    brief_method = st.session_state.brief_method_input
    
    log_placeholder = st.session_state.log_placeholder
    log_placeholder.info(f"Starting production for run '{run_name}'...")

    command = [sys.executable, "main.py", "--project-name", project, "--run-name", run_name]
    if use_gpu: command.append("--use-gpu")
    if no_crop: command.append("--no-crop")
    
    if brief_method == "Use Custom Clips File":
        use_custom_clips = st.session_state.custom_clips_input
        command.extend(["--use-custom-clips", use_custom_clips])
    else: # Generate with LLM
        llm_template = st.session_state.llm_template_input
        custom_instructs = st.session_state.get("custom_instructs_input")
        command.extend(["--use-template", llm_template])
        if custom_instructs: command.extend(["--custom-instructs", custom_instructs])

    try:
        run_command_and_display_output(command, log_placeholder)
        st.success(f"Production for run '{run_name}' completed successfully!")
        st.balloons()
        time.sleep(2)
        st.rerun()
    except subprocess.CalledProcessError:
        st.error("Production failed. The log above contains the error details.")
    except Exception as e:
        st.error(f"A fatal error occurred: {e}")

# --- UI Configuration ---
st.set_page_config(page_title="HookGen Control Panel", page_icon="🎬", layout="wide")
st.title("🎬 HookGen Control Panel")
st.markdown("An interactive UI to manage projects and generate short-form video hooks.")

if 'selected_project' not in st.session_state:
    st.session_state['selected_project'] = None

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Dashboard", "Production Hub"])

# ==============================================================================
# PAGE 1: PROJECT DASHBOARD
# ==============================================================================
if page == "Project Dashboard":
    st.header("Project Dashboard")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Manage Projects")
        projects = get_projects()
        try:
            current_index = projects.index(st.session_state.get('selected_project'))
        except (ValueError, TypeError):
            current_index = 0
        selected = st.selectbox("Select an Existing Project", options=projects, index=current_index)
        st.session_state['selected_project'] = selected if projects else None
        with st.expander("Create New Project"):
            with st.form("prepare_form"):
                new_project_name = st.text_input("New Project Name", help="A unique name, e.g., 'lex-fridman-interview'").strip().replace(" ", "-")
                youtube_url = st.text_input("YouTube URL", help="The full URL of the YouTube video.")
                submitted = st.form_submit_button("Prepare Project Assets")
            if submitted:
                if not new_project_name or not youtube_url:
                    st.error("Please provide both a project name and a YouTube URL.")
                else:
                    st.info(f"Starting preparation for '{new_project_name}'...")
                    log_placeholder = st.empty()
                    try:
                        # Call prepare.py which uses "Output"
                        command = [sys.executable, "prepare.py", "--project-name", new_project_name, "--url", youtube_url]
                        run_command_and_display_output(command, log_placeholder)
                        st.success(f"Project '{new_project_name}' prepared successfully!")
                        st.balloons()
                        st.session_state['selected_project'] = new_project_name
                        time.sleep(1)
                        st.rerun()
                    except subprocess.CalledProcessError:
                        st.error(f"Preparation failed. See logs above for details.")
                    except Exception as e:
                        st.error(f"A fatal error occurred: {e}")
    with col2:
        st.subheader("Project Details")
        project = st.session_state.get('selected_project')
        if not project:
            st.info("Select a project from the left (or create a new one) to see its details.")
        else:
            st.markdown(f"**Project:** `{project}`")
            output_dir = get_output_dir()
            source_assets_dir = output_dir / project / "source_assets"
            st.markdown("#### Source Assets Status")
            assets = {"Source Video": "source_video.mp4", "Source Audio": "source_audio.wav", "LLM Summary": "llm_summary.txt", "Pipeline Transcript": "transcript.json"}
            status_cols = st.columns(len(assets))
            for i, (name, path) in enumerate(assets.items()):
                if (source_assets_dir / path).exists(): status_cols[i].success(f"✅ {name}")
                else: status_cols[i].error(f"❌ {name}")
            
            video_path = source_assets_dir / "source_video.mp4"
            if video_path.exists():
                st.video(str(video_path))
            
            summary_path = source_assets_dir / "llm_summary.txt"
            if summary_path.exists():
                with st.expander("View LLM Summary"): st.text(summary_path.read_text(encoding='utf-8'))
            
            transcript_path = source_assets_dir / "transcript.json"
            if transcript_path.exists():
                with st.expander("View Pipeline-Ready Transcript (JSON)"): st.json(json.loads(transcript_path.read_text(encoding='utf-8')))

# ==============================================================================
# PAGE 2: PRODUCTION HUB
# ==============================================================================
elif page == "Production Hub":
    st.header("Production Hub")
    st.subheader("1. Select Project")
    projects = get_projects()
    if not projects:
        st.warning("No projects found. Please go to the 'Project Dashboard' to create one first.")
        st.stop()

    try:
        current_index = projects.index(st.session_state.get('selected_project'))
    except (ValueError, TypeError):
        current_index = 0

    selected_project = st.selectbox(
        "Which project do you want to work on?", options=projects, index=current_index,
        key="selected_project", on_change=st.rerun
    )
    
    project = st.session_state['selected_project']
    
    st.divider()
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("2. Configure & Launch New Run")
        
        default_run_name = f"{project}_run_{int(time.time())}"
        st.text_input("Run Name", value=st.session_state.get("run_name_input", default_run_name), key="run_name_input")
        st.checkbox("Use GPU", value=True, help="Accelerates face detection.", key="use_gpu_input")
        st.checkbox("Disable Vertical Cropping", help="Skips ASD and maintains original aspect ratio.", key="no_crop_input")
        st.markdown("---")
        st.radio("Creative Brief Method", ("Generate with LLM", "Use Custom Clips File"), horizontal=True, key="brief_method_input")
        
        if st.session_state.brief_method_input == "Generate with LLM":
            st.selectbox("LLM Template", ['rapidfire', 'general', 'emotional', 'keytakeaway', 'controversial'], key="llm_template_input")
            if st.session_state.get("llm_template_input") == 'general':
                st.text_area("Custom Instructions (for 'general' template)", placeholder="e.g., Find 3 clips about the future of AI.", key="custom_instructs_input")
        else:
            st.text_input("Custom Clips File Path", value="inputs/creative_brief.json", key="custom_clips_input")

        st.button("Start Production!", on_click=trigger_production_run, use_container_width=True)
        
        st.session_state.log_placeholder = st.empty()

    with col2:
        st.subheader("3. Past Runs & Results")
        runs = get_runs(project)
        if not runs:
            st.info("No runs found for this project yet. Launch one from the panel on the left.")
        else:
            selected_run_to_view = st.selectbox("Select a run to view", options=runs)
            if selected_run_to_view:
                output_dir = get_output_dir()
                run_dir = output_dir / project / "runs" / selected_run_to_view
                st.markdown(f"#### Results for `{selected_run_to_view}`")
                video_path = run_dir / "final_short_video.mp4"
                if video_path.exists(): st.video(str(video_path))
                else: st.warning("Final video not found. Run may have failed or is still processing.")
                log_path = run_dir / "run.log"
                if log_path.exists():
                    with st.expander("View Full Run Log"): st.code(log_path.read_text(encoding='utf-8'), language="log")
                summary_path = run_dir / "run_summary.json"
                if summary_path.exists():
                    with st.expander("View Run Configuration"): st.json(json.loads(summary_path.read_text(encoding='utf-8')))