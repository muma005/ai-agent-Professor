import subprocess
import os


def submit_to_kaggle(
    submission_path: str,
    competition_name: str,
    message: str = "Professor v2 submission",
) -> dict:
    """
    Submit a CSV to Kaggle via the API.
    
    Returns:
    {
        "success": bool,
        "message": str,
        "public_score": float or None,
    }
    """
    if not os.path.exists(submission_path):
        return {"success": False, "message": f"File not found: {submission_path}"}
    
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", competition_name,
        "-f", submission_path,
        "-m", message,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    if result.returncode == 0:
        # Parse the public LB score from output if available
        public_score = _parse_score(result.stdout)
        return {
            "success": True,
            "message": result.stdout.strip(),
            "public_score": public_score,
        }
    else:
        return {
            "success": False,
            "message": result.stderr.strip(),
            "public_score": None,
        }


def _parse_score(output: str) -> float:
    """Try to extract the public LB score from kaggle CLI output."""
    import re
    match = re.search(r'score:\s*([\d.]+)', output.lower())
    if match:
        return float(match.group(1))
    return None
