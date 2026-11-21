import os
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from supabase import Client, create_client

# loading supabase credentials from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
# Supabase credentials

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# Fast API
app = FastAPI(title="Drone Detection Supabase API", version="2.1.0")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


#database tables & their PK's
TABLES = {
    "sensors": "sensor_id",
    "audio_recordings": "recording_id",
    "camera_frames": "frame_id",
    "acoustic_observations": "acoustic_id",
    "visual_observations": "visual_id",
    "fusion_observations": "fusion_id",
    "tracks": "track_id",
    "track_observations": "track_id",
    "track_states": "state_id",
    "alerts": "alert_id",
}
# Configuration for the unified dashboard event stream

EVENT_CONFIG = {
    "acoustic_detection": ("acoustic_observations", "observed_at", "acoustic_id"),
    "visual_detection": ("visual_observations", "observed_at", "visual_id"),
    "fusion_update": ("fusion_observations", "fused_at", "fusion_id"),
    "track_update": ("tracks", "updated_at", "track_id"),
    "alert": ("alerts", "created_at", "alert_id"),
}

# Request body used by the generic create endpoint.
class JsonBody(BaseModel):
    data: dict[str, Any] = Field(default_factory=dict)

# Validated input for associating a fused observation with a track
class TrackObservationBody(BaseModel):
    track_id: UUID
    fusion_id: UUID
    association_distance_m: float | None = Field(default=None, ge=0)
    association_score: float | None = Field(default=None, ge=0, le=1)

# Fusion layer parameters 
class SensorPayload(BaseModel):
    acoustic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    acoustic_angle: float = Field(default=0.0, ge=0.0, le=360.0)
    visual_score: float = Field(default=0.0, ge=0.0, le=1.0)
    visual_angle: float = Field(default=0.0, ge=0.0, le=360.0)
    modality: str = Field(default="visual")

# Validate a resource name and return its primary-key column
def check_table(table: str) -> str:
    if table not in TABLES:
        raise HTTPException(404, "Unknown resource")
    return TABLES[table]

# Run a Supabase query and convert database errors to HTTP 502 errors
def execute(operation):
    try:
        return operation().data
    except Exception as exc:
        raise HTTPException(502, f"Supabase request failed: {exc}") from exc

# Return an exact row count, optionally filtered by column values
def count_rows(table: str, column: str = "*") -> int:
    check_table(table)
    try:
        response = supabase.table(table).select(
            column, count="exact", head=True
        ).execute()
        return response.count or 0
    except Exception as exc:
        raise HTTPException(502, f"Supabase count failed for {table}: {exc}") from exc

# Return a paginated list of rows from an approved table
def list_rows(table: str, limit: int, offset: int) -> list[dict[str, Any]]:
    key = check_table(table)
    return execute(
        lambda: supabase.table(table)
        .select("*")
        .order(key, desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )

# Return one row by its table-specific primary key
def get_row(table: str, item_id: str) -> dict[str, Any]:
    key = check_table(table)
    rows = execute(
        lambda: supabase.table(table)
        .select("*")
        .eq(key, item_id)
        .limit(1)
        .execute()
    )
    if not rows:
        raise HTTPException(404, "Resource not found")
    return rows[0]

# Insert one row into an approved table and return the inserted row
def insert_row(table: str, data: dict[str, Any]) -> dict[str, Any]:
    check_table(table)
    if not data:
        raise HTTPException(422, "Request body cannot be empty")
    rows = execute(lambda: supabase.table(table).insert(data).execute())
    return rows[0] if rows else data

# Delete one row by primary key and report missing records as 404
def delete_row(table: str, item_id: str) -> None:
    key = check_table(table)
    rows = execute(lambda: supabase.table(table).delete().eq(key, item_id).execute())
    if not rows:
        raise HTTPException(404, "Resource not found")

# Verify that the API can reach the Supabase database
@app.get("/health")
def health():
    try:
        execute(lambda: supabase.table("sensors").select("sensor_id").limit(1).execute())
        return {"status": "ok", "database": "supabase"}
    except HTTPException as exc:
        raise HTTPException(503, detail=exc.detail) from exc


@app.get("/api/v1/dashboard/summary")
def dashboard_summary():
    return {
        "enabled_sensors": count_enabled_sensors(),
        "total_sensors": count_rows("sensors", "sensor_id"),
        "active_tracks": count_active_tracks(),
        "active_alerts": count_active_alerts(),
        "total_acoustic_observations": count_rows("acoustic_observations", "acoustic_id"),
        "total_visual_observations": count_rows("visual_observations", "visual_id"),
        "total_fusion_observations": count_rows("fusion_observations", "fusion_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def count_enabled_sensors() -> int:
    try:
        response = supabase.table("sensors").select(
            "sensor_id", count="exact", head=True
        ).eq("enabled", True).execute()
        return response.count or 0
    except Exception as exc:
        raise HTTPException(502, f"Supabase count failed for enabled sensors: {exc}") from exc

# Count tracks that are currently considered active
def count_active_tracks() -> int:
    try:
        response = supabase.table("tracks").select(
            "track_id", count="exact", head=True
        ).in_("status", ["tentative", "confirmed", "coasting"]).execute()
        return response.count or 0
    except Exception as exc:
        raise HTTPException(502, f"Supabase count failed for tracks: {exc}") from exc


def count_active_alerts() -> int:
    try:
        response = supabase.table("alerts").select(
            "alert_id", count="exact", head=True
        ).eq("status", "active").execute()
        return response.count or 0
    except Exception as exc:
        raise HTTPException(502, f"Supabase count failed for alerts: {exc}") from exc

# Combine recent observations, track updates, and alerts into one feed
@app.get("/api/v1/dashboard/events")
def dashboard_events(
    since: datetime | None = Query(default=None),
    limit: int = Query(100, ge=1, le=500),
):
    since = since or datetime.now(timezone.utc)
    events: list[dict[str, Any]] = []

    for event_type, (table, time_column, id_column) in EVENT_CONFIG.items():
        rows = execute(
            lambda table=table, time_column=time_column: supabase.table(table)
            .select("*")
            .gte(time_column, since.isoformat())
            .order(time_column, desc=True)
            .limit(min(limit, 100))
            .execute()
        )
        for row in rows:
            event_time = row.get(time_column)
            confidence = (
                row.get("fusion_confidence")
                or row.get("visual_confidence")
                or row.get("drone_probability")
                or row.get("confidence")
            )
            severity = "high" if isinstance(confidence, (int, float)) and confidence >= 0.75 else "info"
            if event_type == "alert":
                severity = row.get("severity", "warning")
            events.append({
                "event_type": event_type,
                "event_id": row.get(id_column),
                "event_time": event_time,
                "severity": severity,
                "message": {
                    "acoustic_detection": "Acoustic observation received",
                    "visual_detection": "Visual observation received",
                    "fusion_update": "Fusion-layer update received",
                    "track_update": "Track state updated",
                    "alert": row.get("message") or "Operator alert created",
                }[event_type],
                "payload": row,
            })

    events.sort(key=lambda event: event.get("event_time") or "", reverse=True)
    return events[:limit]

# Return overall API status, dashboard counts, and sensor information
@app.get("/api/v1/system/status")
def system_status():
    summary = dashboard_summary()
    sensors = execute(
        lambda: supabase.table("sensors")
        .select("sensor_id,name,sensor_type,enabled,updated_at")
        .order("name")
        .execute()
    )
    return {
        "status": "operational",
        "database": "supabase",
        "summary": summary,
        "sensors": sensors,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

# Return all configured sensors for the system-status screen
@app.get("/api/v1/system/sensors")
def system_sensors():
    return execute(
        lambda: supabase.table("sensors")
        .select("*")
        .order("name")
        .execute()
    )

# Generic CRUD routes are declared last so they cannot capture the specific dashboard, system, alert, or track routes above
@app.get("/api/v1/{resource}")
def list_resource(
    resource: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    return list_rows(resource, limit, offset)

# Insert one row into one of the approved Supabase resources
@app.post("/api/v1/{resource}", status_code=201)
def create_resource(resource: str, body: JsonBody):
    return insert_row(resource, body.data)

# Retrieve one resource row by its ID
@app.get("/api/v1/{resource}/{item_id}")
def get_resource(resource: str, item_id: str):
    return get_row(resource, item_id)

# Delete one resource row by its ID
@app.delete("/api/v1/{resource}/{item_id}", status_code=204)
def delete_resource(resource: str, item_id: str):
    delete_row(resource, item_id)


@app.post("/api/v1/track-observations", status_code=201)
def associate_observation(body: TrackObservationBody):
    return insert_row("track_observations", body.model_dump(exclude_none=True))

# Return fused observations associated with one track
@app.get("/api/v1/tracks/{track_id}/observations")
def get_track_observations(track_id: UUID):
    return execute(
        lambda: supabase.table("track_observations")
        .select("*, fusion_observations(*)")
        .eq("track_id", str(track_id))
        .order("associated_at")
        .execute()
    )

# Return current and predicted IMM states for one track
@app.get("/api/v1/tracks/{track_id}/states")
def get_track_states(track_id: UUID, limit: int = Query(100, ge=1, le=1000)):
    return execute(
        lambda: supabase.table("track_states")
        .select("*")
        .eq("track_id", str(track_id))
        .order("state_timestamp", desc=True)
        .limit(limit)
        .execute()
    )

# Return unresolved alerts, newest first
@app.get("/api/v1/alerts/active")
def active_alerts(limit: int = Query(100, ge=1, le=1000)):
    return execute(
        lambda: supabase.table("alerts")
        .select("*")
        .eq("status", "active")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

# Return fusion layer score, operation state, and action
@app.post("/api/v1/fusion/evaluate")
def evaluate_fusion(payload:SensorPayload):
    score_acoustic = payload.acoustic_score
    angle_acoustic = payload.acoustic_angle
    score_visual = payload.visual_score
    angle_visual = payload.visual_angle
    has_visual = score_visual > 0.0
    error_threshold = 15.0 

    # Weighting Rules
    if not has_visual:
        weight_acoustic = 1.0
        weight_visual = 0.0
        difference = None
    else:
        angle_difference = abs(angle_acoustic = angle_visual)
        difference = min(angle_difference, 360 - angle_difference)

        if difference <= error_threshold:
            weight_acoustic = 0.25
            weight_visual = 0.75
        else:
            weight_acoustic = 1.0
            weight_visual = 0.0

    # Fusion confidence score
    confidence = (weight_acoustic * score_acoustic) + (weight_visual * score_visual)

    # Decision Thresholds
    if confidence < 0.40:
        state = "PASSIVE_SCANNING"
        action = "HOLD_POSITION"
    elif 0.40 <= confidence < 0.75:
        state = "ACOUSTIC_DETECTION"
        action = "SLEW_CAMERA"
    else:
        state = "TARGET_ALERT"
        action = "TRIGGER_ALERT_AND_LOG"

        try:
            alert_data = {
                "status": "active",
                "severity": "high",
                "message": f"Fusion alert triggered with confidence {confidence:.2f}"
            }
            supabase.table("alerts").insert(alert_data).execute()
        except Exception:
            pass

    return{
        "fused_score": round(confidence, 3),
        "weights": {"acoustic": weight_acoustic, "visual": weight_visual},
        "state": state,
        "action": action,
        "differece": difference
    }
