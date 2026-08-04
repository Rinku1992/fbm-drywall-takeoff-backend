"""
Shared infrastructure helpers for the unit-count-resolver service.

SELF-CONTAINED BY DESIGN. Services in this repo do not import across service
folders, so the helpers this service needs are COPIED here verbatim from
`xtimator-3d/helper.py` @ origin/main (87e6696e). Do not "refactor" this into a
cross-service import — that is the repo convention, not an accident.

Lifted unchanged (byte-for-byte, via AST source extraction):
    load_pg_pool, close_pg_pool, pg_run   -- Cloud SQL pool + retrying executor
    load_vertex_ai_client                 -- Vertex client (+ context caching)
    load_nearest_region                   -- GeoLite2 region selection
    phoenix_call                          -- retrying Gemini call + Pydantic validation

Adapted (ONE deliberate change, documented in MF_REBUILD_REPORT.md):
    load_organization_slug -- the original is `async` and awaits
        run_in_threadpool(pg_run). The resolver runs its whole job synchronously
        inside a worker thread, so this copy is the plain SYNC equivalent calling
        pg_run directly. Same query, same fallback.

NOTE on phoenix_call: it is lifted from origin/main, whose retry escalates
temperature to a 0.25 ceiling. The extraction path in resolver.py deliberately
does NOT rely on that behaviour — it pins temperature and caps attempts at 2
(master plan section 9). See resolver.py:_extract_unit_counts.
"""
import json
import logging
import math
import random
random.seed(0)
import datetime
from time import sleep

import geoip2.database as geoip2_database
from google.oauth2 import service_account
from google.api_core.exceptions import (
    ResourceExhausted,
    ServiceUnavailable,
    DeadlineExceeded,
    InternalServerError,
)
from google.auth.transport.requests import Request
from google.auth.exceptions import TransportError
from google.cloud.sql.connector import Connector, IPTypes
from sqlalchemy import create_engine
from sqlalchemy.exc import (
    OperationalError,
    InterfaceError,
    TimeoutError,
    DBAPIError,
)
from pg8000.dbapi import (
    InterfaceError as InterfaceErrorPG8000,
    DatabaseError as DatabaseErrorPG8000,
)
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import Content, Part  # noqa: F401  (re-exported for resolver.py)
from vertexai.caching import CachedContent

from .prompts import FEEDBACK_GENERATOR


_pg_engine = None
_connector = None
_db_credentials = None


def load_pg_pool(credentials):
    global _pg_engine, _connector, _db_credentials

    if _pg_engine is not None:
        return _pg_engine

    if _connector is None:
        _connector = Connector(refresh_strategy="LAZY")

    pg = credentials["CloudSQL"]

    instance_connection_name = pg["connection_name"]
    db_name = pg["database_name"]
    sa_key = pg["service_account_key"]

    with open(sa_key, "r") as f:
        sa_payload = json.load(f)

    user = sa_payload["client_email"]

    _db_credentials = service_account.Credentials.from_service_account_file(
        sa_key,
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform"
        ]
    )

    request = Request()

    def get_conn():

        nonlocal request

        if (
            _db_credentials.expired
            or _db_credentials.token is None
        ):
            _db_credentials.refresh(request)

        conn = _connector.connect(
            instance_connection_name,
            pg["driver"],
            user=user,
            password=_db_credentials.token,
            db=db_name,
            enable_iam_auth=True,
            ip_type=IPTypes.PRIVATE,
        )

        conn.autocommit = True
        return conn

    _pg_engine = create_engine(
        "postgresql+pg8000://",
        creator=get_conn,
        pool_size=pg.get("min_pool_size", 3),
        max_overflow=max(
            pg.get("max_pool_size", 10)
            - pg.get("min_pool_size", 3),
            0
        ),
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
        pool_use_lifo=True,
        future=True,
    )

    return _pg_engine


def close_pg_pool():
    global _pg_engine, _connector, _db_credentials

    if _pg_engine is not None:
        try:
            _pg_engine.dispose()
        except Exception as e:
            logging.warning(f"SYSTEM: {e}")
        _pg_engine = None

    if _connector is not None:
        try:
            _connector.close()
        except Exception as e:
            logging.warning(f"SYSTEM: {e}")
        _connector = None

    _db_credentials = None


def pg_run(
    credentials,
    pg_pool,
    query,
    params=None,
    fetch=False,
    max_retries=5,
    initial_backoff=1.0,
    max_backoff=30.0,
    execute_many=False,
):
    if params is None:
        params = ()

    for attempt in range(max_retries):

        conn = None
        cursor = None

        try:

            conn = pg_pool["engine"].raw_connection()
            cursor = conn.cursor()

            if execute_many:
                cursor.executemany(query, params)
            else:
                cursor.execute(query, params)

            result = None

            if fetch:
                rows = cursor.fetchall()
                columns = [c[0] for c in cursor.description]
                result = [
                    dict(zip(columns, row))
                    for row in rows
                ]

            conn.commit()

            return result

        except (
            DBAPIError,
            DatabaseErrorPG8000,
            OperationalError,
            InterfaceError,
            InterfaceErrorPG8000,
            TimeoutError,
            TransportError,
        ) as e:

            if conn is not None:
                try:
                    conn.invalidate()
                except Exception:
                    pass

            message = str(e).lower()

            retryable = any(x in message for x in (
                "deadlock",
                "serialization",
                "lock not available",
                "too many connections",
                "connection reset",
                "server closed",
                "network error",
                "timeout",
                "broken pipe",
                "ssl syscall",
                "terminating connection",
            ))

            if retryable and attempt + 1 < max_retries:

                delay = min(
                    initial_backoff * (2 ** attempt)
                    + random.uniform(0, 1),
                    max_backoff,
                )

                logging.warning(
                    "SYSTEM: PostgreSQL retry "
                    f"{attempt+1}/{max_retries} "
                    f"after {delay:.2f}s "
                    f"({type(e).__name__}: {e})"
                )

                sleep(delay)
                continue

            if isinstance(e, TransportError) and attempt + 1 < max_retries:
                close_pg_pool()
                delay = min(
                    initial_backoff * (2 ** attempt)
                    + random.uniform(0, 1),
                    max_backoff,
                )
                sleep(delay)
                engine = load_pg_pool(credentials)
                pg_pool["engine"] = engine
                continue

            raise

        except Exception:

            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass

            raise

        finally:

            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass

            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    raise RuntimeError(
        f"SYSTEM: PostgreSQL query failed after {max_retries} retries."
    )


def load_vertex_ai_client(credentials, ip_address, prompts=None, default_region="us-central1", max_retry=5, base_delay=1.0):
    with open(credentials["VertexAI"]["service_account_key"], 'r') as f:
        project_id = json.load(f)["project_id"]
    region = load_nearest_region(
        ip_address,
        credentials["geolite_database"],
        credentials["VertexAI"]["llm"]["available_regions"],
        default_region=default_region
    )
    vertexai.init(project=project_id, location=region)
    vertex_ai_client = lambda system_instruction: GenerativeModel(
        credentials["VertexAI"]["llm"]["model_name"],
        system_instruction=system_instruction
    )
    is_cached = False
    if prompts and GenerativeModel(credentials["VertexAI"]["llm"]["model_name"]).count_tokens(prompts).total_tokens >= 1024:
        is_cached = True
        n_iterations = 0
        while n_iterations < max_retry:
            try:
                cached_content = CachedContent.create(
                    model_name=credentials["VertexAI"]["llm"]["model_name"],
                    contents=prompts,
                    ttl=datetime.timedelta(hours=5),
                    display_name="drywall_predictor_cache"
                )
                break
            except (ResourceExhausted, InternalServerError) as e:
                n_iterations += 1
                if n_iterations >= max_retry:
                    raise e
                sleep_time = base_delay * (2 ** (n_iterations - 1)) + random.uniform(0, 0.5)
                sleep(sleep_time)
                logging.warning(f"SYSTEM: Vertex AI Gemini: {e}: RETRYING ...")
        vertex_ai_client = GenerativeModel.from_cached_content(cached_content)
    generation_config = credentials["VertexAI"]["llm"]["parameters"]
    return vertex_ai_client, generation_config, is_cached


def load_nearest_region(ip_address, geolite_database, available_regions, default_region="us-central1"):
    def _compute_haversine_distance(latitude_1, longitude_1, latitude_2, longitude_2):
        R = 6371
        d_latitude = math.radians(latitude_2-latitude_1)
        d_longitude = math.radians(longitude_2-longitude_1)
        a = math.sin(d_latitude/2)**2 + math.cos(math.radians(latitude_1)) * math.cos(math.radians(latitude_2)) * math.sin(d_longitude/2)**2
        return 2*R*math.asin(math.sqrt(a))

    if not ip_address or "," not in ip_address:
        return default_region
    if ip_address and "," in ip_address:
        ip_address = ip_address.split(",")[0].strip()
    geoip2_reader = geoip2_database.Reader(geolite_database)
    try:
        response = geoip2_reader.city(ip_address)
        (response.location.latitude, response.location.longitude, response.country.iso_code)
        nearest_region = None
        minimum_distance = float("inf")
        for region, (latitude, longitude) in available_regions.items():
            distance = _compute_haversine_distance(response.location.latitude, response.location.longitude, latitude, longitude)
            if distance < minimum_distance:
                minimum_distance = distance
                nearest_region = region
        return nearest_region
    except Exception:
        return default_region


def phoenix_call(generate_content_lambda, max_retry=5, base_delay=1.0, pydantic_model=None, verify_field_counts=None):
    n_iterations = 0
    temperature = 0
    exceptions = list()
    feedback_prompt = ''
    while n_iterations < max_retry:
        try:
            response = generate_content_lambda(feedback_prompt, temperature)
            if pydantic_model:
                json_response = json.loads(response.text.strip("`json").replace("{{", '{').replace("}}", '}'))
                if verify_field_counts:
                    for field, count in verify_field_counts.items():
                        if len(json_response[field]) != count:
                            raise ValueError(f"Predicted {field} count: {len(json_response[field])} does not match with the expected number: {count}")
                response_json_pydantic = pydantic_model(**json_response)
                return response_json_pydantic, json_response
            return response.text
        except (ResourceExhausted, ServiceUnavailable, DeadlineExceeded) as e:
            n_iterations += 1
            if n_iterations >= max_retry:
                raise e
            sleep_time = base_delay * (2 ** (n_iterations - 1)) + random.uniform(0, 0.5)
            sleep(sleep_time)
            logging.warning(f"SYSTEM: Vertex AI Gemini: {e}: RETRYING ...")
        except Exception as e:
            n_iterations += 1
            if n_iterations >= max_retry:
                raise e
            exceptions.append(e)
            system_feedback = [Part.from_text(FEEDBACK_GENERATOR.format(max_retry=max_retry, exceptions=exceptions))]
            feedback_prompt = Content(role="model", parts=system_feedback)
            temperature = min(0.25 * (n_iterations + 1) / max_retry, 0.25)
            logging.warning(f"SYSTEM: Vertex AI Gemini: Response Generation/Parsing failed with ERROR: {e}: RETRYING ...")
            logging.warning(f"SYSTEM: RETRYING with TEMPERATURE: {temperature}")


def load_organization_slug(credentials, pg_pool, user_id):
    """SYNC copy of xtimator-3d/helper.py:load_organization_slug (helper.py:1464).

    Identical query and fallback; the only change is that it calls pg_run
    directly instead of `await run_in_threadpool(partial(pg_run, ...))`, because
    the resolver's whole job already runs off the event loop in a worker thread.
    """
    query = f"""SELECT COALESCE(o.organization_slug, 
            NULLIF(split_part(u.user_email,'@',2),''), 
            u.user_email) AS org_or_domain
        FROM {credentials["CloudSQL"]["table_name_users"]} u
        LEFT JOIN {credentials["CloudSQL"]["table_name_organizations"]} o ON TEXT(u.organization_id) = TEXT(o.organization_id)
        WHERE LOWER(u.user_email) = LOWER(%s);
    """
    query_output = pg_run(credentials, pg_pool, query, params=(user_id,), fetch=True)
    if query_output and query_output[0]["org_or_domain"]:
        return query_output[0]["org_or_domain"]
    return user_id.split('@')[1]
