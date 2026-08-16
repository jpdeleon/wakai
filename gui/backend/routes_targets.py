"""Target snapshot persistence: save/load/list a target's metadata, ref_table,
and model results to/from SQLite (wakai/database.py).
"""
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BACKEND_DIR))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from flask import Blueprint, current_app, jsonify, request  # noqa: E402

from wakai.database import TargetDatabase  # noqa: E402

targets_bp = Blueprint("routes_targets", __name__)


def _target_database():
    return TargetDatabase(current_app.config["DATABASE_PATH"])


@targets_bp.route("/api/targets/save", methods=["POST"])
def save_target():
    """Persist one target's metadata, ref_table, and model results."""
    data = request.get_json(force=True)
    target = data.get("target")
    ref_table = data.get("ref_table")
    results = data.get("results")
    if not isinstance(target, dict):
        return jsonify({"status": "error", "message": "target must be an object"}), 400
    if not isinstance(ref_table, list):
        return jsonify({"status": "error", "message": "ref_table must be a list"}), 400
    if not isinstance(results, dict):
        return jsonify({"status": "error", "message": "results must be an object"}), 400
    try:
        key = _target_database().save(target, ref_table, results)
    except (TypeError, ValueError) as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Database save failed: {exc}"}), 500
    return jsonify({"status": "ok", "key": key})


@targets_bp.route("/api/targets/load", methods=["GET"])
def load_target():
    """Load a target snapshot by Gaia ID or normalized name."""
    gaia_id = str(request.args.get("gaia_id") or "").strip() or None
    name = str(request.args.get("name") or "").strip() or None
    if not gaia_id and not name:
        return jsonify({"status": "error", "message": "Provide gaia_id or name"}), 400
    try:
        record = _target_database().load(gaia_id=gaia_id, name=name)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Database load failed: {exc}"}), 500
    if record is None:
        return jsonify({"status": "error", "message": "No saved results for target"}), 404
    return jsonify({"status": "ok", "record": record})


@targets_bp.route("/api/targets", methods=["GET"])
def list_targets():
    """List saved targets, most recently updated first."""
    try:
        targets = _target_database().list_targets()
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Database query failed: {exc}"}), 500
    return jsonify({"status": "ok", "targets": targets})
