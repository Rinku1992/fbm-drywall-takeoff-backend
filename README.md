## Introduction
#### XTIMATOR is an AI-Powered automated construction estimation system designed to generate accurate drywall material quantities directly from architectural floor plans. It leverages structured wall geometry, room polygons, ceiling configurations, openings and architectural scale information to compute drywall surface areas and material requirements for both walls and ceilings while accounting for project-specific waste factors and installation constraints.

## XTIMATOR Functional Architecture on GCP
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/106274e5-d12f-44c9-9eee-5fcbdb40dfb1" />

## Installation
### Fully Managed Relational Database - Cloud SQL for PostgreSQL
<b>Database Name: </b> <b><i>drywall_takeoff</i></b><br>

<b>Table Names,</b><br>
1. <b><i>projects</i></b>
```sql
CREATE TABLE projects (
    project_id TEXT PRIMARY KEY,

    project_name TEXT,
    project_location TEXT,
    FBM_branch TEXT,
    project_type TEXT,
    project_area TEXT,
    contractor_name TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT
);
```

2. <b><i>plans</i></b>
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

3. <b><i>pages</i></b>
```sql
CREATE TABLE pages (
    plan_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    user_id TEXT,

    page_number INTEGER NOT NULL,

    mask_factor JSONB,
    bounding_box_offsets JSONB,

    source TEXT,
    thumbnail TEXT,
    plan_type TEXT,

    extracted BOOLEAN DEFAULT FALSE,
    status TEXT,

    is_floorplan BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (project_id, plan_id, page_number)
);
```

4. <b><i>models</i></b>
```sql
CREATE TABLE models (
    plan_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    user_id TEXT,

    page_number INTEGER NOT NULL,
    page_section_number INTEGER NOT NULL DEFAULT 0,
    page_sections INTEGER DEFAULT 1,

    scale TEXT,

    model_2d JSONB DEFAULT '{}'::jsonb,
    model_3d JSONB DEFAULT '{}'::jsonb,
    takeoff JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,

    source TEXT,
    target_drywalls TEXT,

    waste_average DOUBLE PRECISION,
    drywall_negate_opening_area_threshold DOUBLE PRECISION,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (
        project_id,
        plan_id,
        page_number,
        page_section_number
    )
);
```

5. <b><i>model_revisions_2d</i></b>
```sql
CREATE TABLE model_revisions_2d (
    plan_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    user_id TEXT,

    page_number INTEGER NOT NULL,
    page_section_number INTEGER NOT NULL DEFAULT 0,

    revision_number INTEGER NOT NULL,

    scale TEXT,

    model JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (
        project_id,
        plan_id,
        page_number,
        page_section_number,
        revision_number
    )
);
```

6. <b><i>model_revisions_3d</i></b>
```sql
CREATE TABLE model_revisions_3d (
    plan_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    user_id TEXT,

    page_number INTEGER NOT NULL,
    page_section_number INTEGER NOT NULL DEFAULT 0,

    revision_number INTEGER NOT NULL,

    scale TEXT,

    model JSONB NOT NULL DEFAULT '{}'::jsonb,
    takeoff JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (
        project_id,
        plan_id,
        page_number,
        page_section_number,
        revision_number
    )
);
```
