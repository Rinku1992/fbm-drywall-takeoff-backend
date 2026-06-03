## Introduction
#### XTIMATOR is an AI-Powered automated construction estimation system designed to generate accurate drywall material quantities directly from architectural floor plans. It leverages structured wall geometry, room polygons, ceiling configurations, openings and architectural scale information to compute drywall surface areas and material requirements for both walls and ceilings while accounting for project-specific waste factors and installation constraints.

## XTIMATOR Functional Architecture on GCP
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/106274e5-d12f-44c9-9eee-5fcbdb40dfb1" />

## Installation
### Fully Managed Relational Database - Cloud SQL for PostgreSQL
<b>Database Name: </b> <b><i>drywall_takeoff</i></b><br>

<b>1. Table Names</b><br>
1. <b><i>plans</i></b>
```sql
CREATE TABLE plans (
    plan_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    user_id TEXT NOT NULL,

    status TEXT,
    plan_name TEXT,
    plan_type TEXT,
    file_type TEXT,

    pages INTEGER DEFAULT 0,
    size_in_bytes BIGINT DEFAULT 0,

    source TEXT,
    sha256 TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    multipage_elevation_map JSONB DEFAULT '{}'::jsonb
);
```
