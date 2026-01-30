#!/usr/bin/env python3
"""
Flask backend for LLM-Enhanced Quadruped Control Frontend

This provides a web interface for running the LLM pipeline instead of CLI.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, TypedDict

from flask import Flask, Response, jsonify, render_template, request, send_file
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
CORS(app)

# Suppress Flask's default request logging (reduces terminal spam)
werkzeug_log = logging.getLogger("werkzeug")
werkzeug_log.setLevel(logging.ERROR)


class LogEntry(TypedDict):
    timestamp: str
    level: str
    message: str


class PipelineResults(TypedDict, total=False):
    command: str
    total_iterations: int
    best_score: float
    pipeline_success: bool
    elapsed_time: float
    results_directory: str
    best_iteration: int | None


class PipelineStatus(TypedDict):
    running: bool
    current_iteration: int
    max_iterations: int
    command: str
    logs: list[LogEntry]
    results: PipelineResults | None
    start_time: float | None
    error: str | None
    video_dir: str | None


# Global state for tracking pipeline runs
pipeline_status: PipelineStatus = {
    "running": False,
    "current_iteration": 0,
    "max_iterations": 0,
    "command": "",
    "logs": [],
    "results": None,
    "start_time": None,
    "error": None,
    "video_dir": None,  # Current run's video directory
}

# Thread control
stop_event = threading.Event()
pipeline_thread: threading.Thread | None = None

log_queue: queue.Queue[LogEntry] = queue.Queue()


def check_api_key() -> tuple[bool, str]:
    """Check if the required API key is configured."""
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).parent.parent / ".env")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            return (
                False,
                "ANTHROPIC_API_KEY not configured. Please set it in the .env file.",
            )
        return True, "API key configured"
    except ImportError:
        return False, "python-dotenv not installed. Run: pip install python-dotenv"


def add_log(message: str, level: str = "info") -> None:
    """Add a log message to the queue."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry: LogEntry = {"timestamp": timestamp, "level": level, "message": message}
    pipeline_status["logs"].append(log_entry)
    log_queue.put(log_entry)


class FrontendLogHandler(logging.Handler):
    """Custom log handler that forwards pipeline logs to the frontend."""

    def emit(self, record: logging.LogRecord) -> None:
        # Check if stop was requested
        if stop_event.is_set():
            raise KeyboardInterrupt("Stop requested by user")

        try:
            msg = self.format(record)
            level = record.levelname.lower()
            if level == "warning":
                level = "warning"
            elif level in ("error", "critical"):
                level = "error"
            else:
                level = "info"

            # Check for iteration pattern to update progress
            iter_match = re.search(r"---\s*Iteration\s+(\d+)\s*---", msg)
            if iter_match:
                pipeline_status["current_iteration"] = int(iter_match.group(1))

            # Also check for other iteration indicators
            iter_match2 = re.search(
                r"Iteration\s+(\d+)\s+completed", msg, re.IGNORECASE
            )
            if iter_match2:
                pipeline_status["current_iteration"] = int(iter_match2.group(1))

            add_log(msg, level)
        except Exception:
            pass


def setup_log_capture() -> FrontendLogHandler:
    """Setup log capture from the pipeline's logger."""
    llm_logger = logging.getLogger("llm_integration")

    # Remove existing frontend handlers to avoid duplicates
    for handler in llm_logger.handlers[:]:
        if isinstance(handler, FrontendLogHandler):
            llm_logger.removeHandler(handler)

    # Add our custom handler
    frontend_handler = FrontendLogHandler()
    frontend_handler.setLevel(logging.INFO)
    frontend_handler.setFormatter(logging.Formatter("%(message)s"))
    llm_logger.addHandler(frontend_handler)

    return frontend_handler


def setup_video_directory() -> Path:
    """Set up a fresh video directory for the current run.

    This creates frontend/results/videos/<timestamp>/ and clears any previous run folders.
    """
    import shutil

    videos_base = Path(__file__).parent / "results" / "videos"
    videos_base.mkdir(parents=True, exist_ok=True)

    # Delete all existing subdirectories (previous runs)
    for item in videos_base.iterdir():
        if item.is_dir():
            shutil.rmtree(item)

    # Create new directory for this run with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_video_dir = videos_base / f"run_{timestamp}"
    run_video_dir.mkdir(parents=True, exist_ok=True)

    return run_video_dir


def run_pipeline_thread(
    command: str,
    max_iterations: int,
    config_mode: str,
    results_dir: str,
    solver_max_iter: int | None = None,
) -> None:
    """Run the pipeline in a background thread."""
    global pipeline_status, stop_event

    # Clear stop event at start
    stop_event.clear()

    # Set up fresh video directory for this run
    video_dir = setup_video_directory()

    pipeline_status["running"] = True
    pipeline_status["current_iteration"] = 0
    pipeline_status["max_iterations"] = max_iterations
    pipeline_status["command"] = command
    pipeline_status["logs"] = []
    pipeline_status["results"] = None
    pipeline_status["error"] = None
    pipeline_status["start_time"] = time.time()
    pipeline_status["video_dir"] = str(video_dir)

    frontend_handler = None

    try:
        add_log(f"Starting pipeline for command: '{command}'", "info")
        add_log(f"Max LLM iterations: {max_iterations}", "info")
        add_log(f"Video directory: {video_dir}", "info")
        if solver_max_iter:
            add_log(f"Solver max iterations per solve: {solver_max_iter}", "info")
        else:
            add_log("Solver max iterations: unlimited (until convergence)", "info")
        add_log(f"Config mode: {config_mode}", "info")
        add_log(f"Results directory: {results_dir}", "info")

        # Set environment variables BEFORE importing pipeline
        os.environ["MAX_LLM_ITERATIONS"] = str(max_iterations)
        os.environ["RESULTS_DIR"] = results_dir
        os.environ["VIDEO_DIR"] = str(video_dir)  # Pass video directory to pipeline

        # Import and configure
        add_log("Loading pipeline components...", "info")
        import config
        from llm_integration import FeedbackPipeline

        config.CONSTRAINT_MODE = config_mode

        # Set solver max iterations if specified
        if solver_max_iter and solver_max_iter > 0:
            config.solver_config["ipopt.max_iter"] = solver_max_iter
            add_log(f"IPOPT max_iter set to: {solver_max_iter}", "info")

        # Setup log capture to forward pipeline logs to frontend
        frontend_handler = setup_log_capture()

        # Initialize pipeline
        add_log("Initializing feedback pipeline...", "info")
        pipeline = FeedbackPipeline(config)

        # Override max_iterations directly on the pipeline instance
        pipeline.max_iterations = max_iterations
        add_log(f"Pipeline max_iterations set to: {pipeline.max_iterations}", "info")

        # Run pipeline
        add_log("Running optimization pipeline...", "info")
        results = pipeline.run_pipeline(command)

        start_time = pipeline_status["start_time"]
        elapsed_time = time.time() - start_time if start_time else 0

        # Store results
        pipeline_status["results"] = {
            "command": results.get("command", command),
            "total_iterations": results.get("total_iterations", 0),
            "best_score": results.get("best_score", 0),
            "pipeline_success": results.get("pipeline_success", False),
            "elapsed_time": round(elapsed_time, 1),
            "results_directory": results.get("results_directory", results_dir),
            "best_iteration": results.get("best_iteration"),
        }

        if results.get("pipeline_success"):
            add_log("Pipeline completed successfully!", "success")
        else:
            add_log("Pipeline completed with limited success.", "warning")

    except KeyboardInterrupt:
        add_log("Pipeline interrupted by user.", "warning")
        pipeline_status["error"] = "Pipeline interrupted by user"

    except ImportError as e:
        error_msg = f"Import Error: {e}. Make sure all dependencies are installed."
        add_log(error_msg, "error")
        pipeline_status["error"] = error_msg

    except Exception as e:
        error_msg = f"Error running pipeline: {str(e)}"
        add_log(error_msg, "error")
        pipeline_status["error"] = error_msg
        import traceback

        add_log(traceback.format_exc(), "error")

    finally:
        pipeline_status["running"] = False
        add_log("Pipeline finished.", "info")

        # Cleanup log handler
        if frontend_handler:
            try:
                llm_logger = logging.getLogger("llm_integration")
                llm_logger.removeHandler(frontend_handler)
            except Exception:
                pass


@app.route("/")
def index() -> str:
    """Serve the main frontend page."""
    return render_template("index.html")


@app.route("/api/check-config", methods=["GET"])
def check_config() -> Response:
    """Check if the system is properly configured."""
    api_ok, api_msg = check_api_key()
    return jsonify(
        {"api_key_configured": api_ok, "api_key_message": api_msg, "ready": api_ok}
    )


@app.route("/api/run", methods=["POST"])
def run_pipeline() -> Response | tuple[Response, int]:
    """Start the pipeline with given parameters."""
    global pipeline_status

    if pipeline_status["running"]:
        return jsonify({"error": "Pipeline is already running"}), 400

    data = request.json
    command = data.get("command", "").strip()

    if not command:
        return jsonify({"error": "Please provide a command"}), 400

    # Check API key first
    api_ok, api_msg = check_api_key()
    if not api_ok:
        return jsonify({"error": api_msg}), 400

    max_iterations = int(data.get("max_iterations", 5))
    config_mode = data.get("config_mode", "complementarity")
    results_dir = data.get("results_dir", "results/llm_iterations")

    # Solver max iterations (0 or empty = unlimited)
    solver_max_iter_raw = data.get("solver_max_iter", "")
    solver_max_iter = (
        int(solver_max_iter_raw)
        if solver_max_iter_raw and str(solver_max_iter_raw).strip()
        else None
    )

    # Start pipeline in background thread
    global pipeline_thread
    pipeline_thread = threading.Thread(
        target=run_pipeline_thread,
        args=(command, max_iterations, config_mode, results_dir, solver_max_iter),
        daemon=True,
    )
    pipeline_thread.start()

    return jsonify({"status": "started", "command": command})


@app.route("/api/status", methods=["GET"])
def get_status() -> Response:
    """Get the current pipeline status."""
    return jsonify(
        {
            "running": pipeline_status["running"],
            "current_iteration": pipeline_status["current_iteration"],
            "max_iterations": pipeline_status["max_iterations"],
            "command": pipeline_status["command"],
            "results": pipeline_status["results"],
            "error": pipeline_status["error"],
            "elapsed_time": round(time.time() - pipeline_status["start_time"], 1)
            if pipeline_status["start_time"] and pipeline_status["running"]
            else None,
        }
    )


@app.route("/api/logs", methods=["GET"])
def get_logs() -> Response:
    """Get all logs."""
    return jsonify({"logs": pipeline_status["logs"]})


@app.route("/api/logs/stream")
def stream_logs() -> Response:
    """Stream logs using Server-Sent Events."""

    def generate() -> Iterator[str]:
        while True:
            try:
                log_entry = log_queue.get(timeout=1)
                yield f"data: {json.dumps(log_entry)}\n\n"
            except queue.Empty:
                # Send heartbeat
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/stop", methods=["POST"])
def stop_pipeline() -> Response | tuple[Response, int]:
    """Stop the running pipeline."""
    global stop_event, pipeline_thread

    if not pipeline_status["running"]:
        return jsonify({"error": "No pipeline is running"}), 400

    add_log("Stop requested by user - terminating pipeline...", "warning")

    # Set the stop event
    stop_event.set()

    # Give the thread a moment to check the flag
    time.sleep(0.5)

    # If still running, we need to force stop
    if pipeline_status["running"] and pipeline_thread and pipeline_thread.is_alive():
        add_log("Force stopping pipeline...", "warning")
        # Mark as stopped
        pipeline_status["running"] = False
        pipeline_status["error"] = "Pipeline stopped by user"
        add_log("Pipeline stopped.", "warning")

    return jsonify({"status": "stopped"})


@app.route("/api/results", methods=["GET"])
def get_results() -> Response | tuple[Response, int]:
    """Get the results from the last pipeline run."""
    if pipeline_status["results"]:
        return jsonify(pipeline_status["results"])
    return jsonify({"error": "No results available"}), 404


@app.route("/api/list-results", methods=["GET"])
def list_results() -> Response:
    """List available result files."""
    results_dir = Path(__file__).parent / "results" / "llm_iterations"
    if not results_dir.exists():
        return jsonify({"files": []})

    files = []
    for f in sorted(results_dir.glob("*.json"), reverse=True):
        files.append(
            {
                "name": f.name,
                "path": str(f),
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            }
        )

    return jsonify({"files": files[:50]})  # Return last 50 files


@app.route("/api/videos", methods=["GET"])
def list_videos() -> Response:
    """List available videos from the current run's video directory."""
    videos = []

    # Get the current run's video directory
    video_dir = pipeline_status.get("video_dir")

    if not video_dir:
        # Try to find the most recent video run directory
        videos_base = Path(__file__).parent / "results" / "videos"
        if videos_base.exists():
            run_dirs = [d for d in videos_base.iterdir() if d.is_dir()]
            if run_dirs:
                video_dir = str(max(run_dirs, key=lambda d: d.stat().st_mtime))

    if video_dir:
        video_path = Path(video_dir)
        if video_path.exists():
            for video_file in sorted(video_path.glob("*.mp4")):
                name = video_file.name
                # Determine video type from filename
                if "simulation_iter" in name:
                    video_type = "simulation"
                    iteration = (
                        name.replace("simulation_iter_", "")
                        .replace("_DEBUG", "")
                        .replace(".mp4", "")
                    )
                elif "planned_traj_iter" in name:
                    video_type = "planned"
                    iteration = (
                        name.replace("planned_traj_iter_", "")
                        .replace("_DEBUG", "")
                        .replace(".mp4", "")
                    )
                    if "_DEBUG" in name:
                        video_type = "planned-debug"
                elif "debug_trajectory_iter" in name:
                    video_type = "debug"
                    iteration = (
                        name.replace("debug_trajectory_iter_", "")
                        .replace("_DEBUG", "")
                        .replace(".mp4", "")
                    )
                else:
                    video_type = "other"
                    iteration = "0"

                videos.append(
                    {
                        "name": name,
                        "path": str(video_file),
                        "type": video_type,
                        "iteration": iteration,
                        "size": video_file.stat().st_size,
                        "modified": datetime.fromtimestamp(
                            video_file.stat().st_mtime
                        ).isoformat(),
                        "source": "video_dir",
                    }
                )

    # Sort by iteration (descending) then by type
    type_order: dict[str, int] = {
        "simulation": 0,
        "planned": 1,
        "planned-debug": 2,
        "debug": 3,
        "other": 4,
    }
    videos.sort(
        key=lambda v: (
            -int(str(v["iteration"])) if str(v["iteration"]).isdigit() else 0,
            type_order.get(str(v["type"]), 4),
        )
    )

    return jsonify({"videos": videos, "video_directory": video_dir})


@app.route("/api/video/<path:video_path>")
def serve_video(video_path: str) -> Response | tuple[Response, int]:
    """Serve a video file."""
    # Security: ensure the path is within the results directory
    base_dir = Path(__file__).parent / "results"

    # Handle both absolute and relative paths
    if video_path.startswith("/"):
        video_file = Path(video_path)
    else:
        video_file = base_dir / video_path

    # Resolve to absolute and check it's within results
    try:
        video_file = video_file.resolve()
        base_dir = base_dir.resolve()

        if not str(video_file).startswith(str(base_dir)):
            return jsonify({"error": "Access denied"}), 403

        if not video_file.exists():
            return jsonify({"error": "Video not found"}), 404

        return send_file(video_file, mimetype="video/mp4", as_attachment=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video-by-name/<filename>")
def serve_video_by_name(filename: str) -> Response | tuple[Response, int]:
    """Serve a video file by name from the current run's video directory."""
    video_dir = pipeline_status.get("video_dir")

    if not video_dir:
        # Try to find the most recent video run directory
        videos_base = Path(__file__).parent / "results" / "videos"
        if videos_base.exists():
            run_dirs = [d for d in videos_base.iterdir() if d.is_dir()]
            if run_dirs:
                video_dir = str(max(run_dirs, key=lambda d: d.stat().st_mtime))

    if not video_dir:
        return jsonify({"error": "No video directory found"}), 404

    video_file = Path(video_dir) / filename

    if not video_file.exists():
        return jsonify({"error": "Video not found"}), 404

    return send_file(video_file, mimetype="video/mp4", as_attachment=False)


@app.route("/api/latest-iteration-video")
def get_latest_iteration_video() -> Response:
    """Get the video for the most recently completed iteration from the current run's video directory."""
    video_dir = pipeline_status.get("video_dir")

    if not video_dir:
        # Try to find the most recent video run directory
        videos_base = Path(__file__).parent / "results" / "videos"
        if videos_base.exists():
            run_dirs = [d for d in videos_base.iterdir() if d.is_dir()]
            if run_dirs:
                video_dir = str(max(run_dirs, key=lambda d: d.stat().st_mtime))

    if not video_dir:
        return jsonify(
            {
                "found": False,
                "current_iteration": pipeline_status.get("current_iteration", 0),
            }
        )

    # Find the most recent iteration video (check multiple video types)
    latest_video = None
    latest_iteration = -1

    video_path = Path(video_dir)
    if video_path.exists():
        # Check all video types: planned_traj, debug_trajectory, simulation
        video_patterns = [
            ("planned_traj_iter_*.mp4", "planned"),
            ("debug_trajectory_iter_*.mp4", "debug"),
            ("simulation_iter_*.mp4", "simulation"),
        ]

        for pattern, video_type in video_patterns:
            for video_file in video_path.glob(pattern):
                name = video_file.name
                try:
                    # Extract iteration number from filename
                    if video_type == "planned":
                        iter_part = (
                            name.replace("planned_traj_iter_", "")
                            .replace("_DEBUG", "")
                            .replace(".mp4", "")
                        )
                    elif video_type == "debug":
                        iter_part = name.replace("debug_trajectory_iter_", "").replace(
                            ".mp4", ""
                        )
                    else:  # simulation
                        iter_part = name.replace("simulation_iter_", "").replace(
                            ".mp4", ""
                        )

                    iteration = int(iter_part)

                    # Prefer planned > simulation > debug for same iteration
                    type_priority: dict[str, int] = {
                        "planned": 3,
                        "simulation": 2,
                        "debug": 1,
                    }
                    current_priority = type_priority.get(video_type, 0)
                    existing_priority = 0
                    if latest_video:
                        existing_priority = type_priority.get(
                            str(latest_video.get("type", "")), 0
                        )

                    # Update if this is a newer iteration, or same iteration with higher priority
                    if iteration > latest_iteration or (
                        iteration == latest_iteration
                        and current_priority > existing_priority
                    ):
                        latest_iteration = iteration
                        latest_video = {
                            "name": name,
                            "path": str(video_file),
                            "iteration": iteration,
                            "type": video_type,
                            "source": "video_dir",
                        }
                except ValueError:
                    continue

    if latest_video:
        return jsonify(
            {
                "found": True,
                "video": latest_video,
                "current_iteration": pipeline_status.get("current_iteration", 0),
            }
        )

    return jsonify(
        {
            "found": False,
            "current_iteration": pipeline_status.get("current_iteration", 0),
        }
    )


@app.route("/api/video-dir", methods=["GET"])
def get_video_dir() -> Response:
    """Get the current run's video directory path."""
    video_dir = pipeline_status.get("video_dir")
    return jsonify(
        {"video_dir": video_dir, "running": pipeline_status.get("running", False)}
    )


@app.route("/api/video-from-results/<filename>")
def serve_video_from_results(filename: str) -> Response | tuple[Response, int]:
    """Serve a video file by name from the frontend/results directory."""
    main_results_dir = Path(__file__).parent / "results"
    video_file = main_results_dir / filename

    if not video_file.exists():
        return jsonify({"error": "Video not found"}), 404

    # Security check
    try:
        video_file = video_file.resolve()
        main_results_dir = main_results_dir.resolve()

        if not str(video_file).startswith(str(main_results_dir)):
            return jsonify({"error": "Access denied"}), 403
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return send_file(video_file, mimetype="video/mp4", as_attachment=False)


if __name__ == "__main__":
    print("=" * 60)
    print("LLM-Enhanced Quadruped Control - Web Frontend")
    print("=" * 60)

    # Check configuration
    api_ok, api_msg = check_api_key()
    if api_ok:
        print(f"✓ {api_msg}")
    else:
        print(f"⚠ {api_msg}")

    print()
    print("Starting server at http://localhost:5001")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
