## Introduction
#### XTIMATOR is an AI-Powered automated construction estimation system designed to generate accurate drywall material quantities directly from architectural floor plans. It leverages structured wall geometry, room polygons, ceiling configurations, openings and architectural scale information to compute drywall surface areas and material requirements for both walls and ceilings while accounting for location jurisdiction on construction practices, project-specific waste factors and installation constraints.

## XTIMATOR Functional Architecture
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/5a85bd37-36ed-4185-bb50-8e47161be420" />

## XTIMATOR User Guides
* ### Onboarding External Organizations into Xtimator Application
    * [User guide to onboard external organizations](https://fbm.atlassian.net/wiki/spaces/FBMAITEAM/pages/3429728263/User+Guide+Onboarding+External+Organizations+into+Xtimator+Application)

## Installation
### Fully Managed Relational Database - Cloud SQL for PostgreSQL
<b>Database Name: </b> <b><i>drywall_takeoff</i></b><br>
```sql
CREATE DATABASE drywall_takeoff;
```

<b>Table Names,</b><br>
1. <b><i>projects</i></b>
```sql
CREATE TABLE projects (
    project_id TEXT PRIMARY KEY,

    project_name TEXT,
    project_location TEXT,
    project_location_pincode INTEGER,
    "FBM_branch" TEXT,
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

    multipage_elevation_map JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (project_id, plan_id),
    CONSTRAINT plan_unique
        UNIQUE (project_id, plan_id)
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

    is_vector BOOLEAN,
    vector_scale JSONB DEFAULT '{}'::jsonb,
    vector_ceiling_height JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (project_id, plan_id, page_number),
    CONSTRAINT page_unique
        UNIQUE (project_id, plan_id, page_number)
);
```

4. <b><i>models</i></b>
```sql
CREATE TABLE models (
    plan_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    user_id TEXT,

    page_number INTEGER NOT NULL,
    page_section_number TEXT,
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
    ),
    CONSTRAINT model_unique
        UNIQUE (project_id, plan_id, page_number, page_section_number)
);
```

5. <b><i>model_revisions_2d</i></b>
```sql
CREATE TABLE model_revisions_2d (
    plan_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    user_id TEXT,

    page_number INTEGER NOT NULL,
    page_section_number TEXT,

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
    ),
    CONSTRAINT model_revision_2d_unique
        UNIQUE (project_id, plan_id, page_number, page_section_number, revision_number)
);
```

6. <b><i>model_revisions_3d</i></b>
```sql
CREATE TABLE model_revisions_3d (
    plan_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    user_id TEXT,

    page_number INTEGER NOT NULL,
    page_section_number TEXT,

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
    ),
    CONSTRAINT model_revision_3d_unique
        UNIQUE (project_id, plan_id, page_number, page_section_number, revision_number)
);
```

7. <b><i>sessions</i></b>
```sql
CREATE TABLE sessions (
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    plan_id TEXT,
    page_number INTEGER NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT,

    PRIMARY KEY (
        session_id
    )
);
```

8. <b><i>users</i></b>
```sql
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,

    group_ids TEXT[] DEFAULT ARRAY[]::TEXT[],

    organization_id TEXT,
    user_name TEXT,
    user_location TEXT,
    is_external BOOLEAN
);
```

9. <b><i>groups</i></b>
```sql
CREATE TABLE groups (

    id SERIAL PRIMARY KEY,
    region_id INT REFERENCES regions(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(region_id, name)
);
```

10. <b><i>organizations</i></b>
```sql
CREATE TABLE organizations (
    organization_id UUID PRIMARY KEY,
    organization_name VARCHAR(255) NOT NULL,
    organization_slug VARCHAR(100) UNIQUE NOT NULL,

    status VARCHAR(50) NOT NULL,

    billing_plan VARCHAR(50),
    subscription_status VARCHAR(50),

    timezone VARCHAR(100),
    country_code VARCHAR(10)
);
```

11. <b><i>regions</i></b>
```sql
CREATE TABLE regions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

12. <b><i>organization_regions</i></b>
```sql
CREATE TABLE organization_regions (
    organization_id UUID NOT NULL REFERENCES organizations(organization_id) ON DELETE CASCADE,
    region_id INT NOT NULL REFERENCES regions(id) ON DELETE CASCADE,

    PRIMARY KEY (organization_id, region_id),
    CONSTRAINT region_unique
        UNIQUE (organization_id, region_id)
);
```

13. <b><i>roles</i></b>
```sql
CREATE TABLE roles (

    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    description TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

13. <b><i>sku</i></b>
```sql
CREATE TABLE sku (
    sku_id TEXT PRIMARY KEY,

    sku_description TEXT NOT NULL,

    product_cat_code INTEGER,
    product_cat_description TEXT,

    thickness_inches DOUBLE PRECISION
        CHECK (thickness_inches > 0),

    fire_rating TEXT,

    is_lightweight BOOLEAN NOT NULL DEFAULT FALSE,
    is_wide_stretch BOOLEAN NOT NULL DEFAULT FALSE,

    color_code JSONB DEFAULT '{}'::jsonb,

    waste TEXT,

    sheet_size TEXT NOT NULL
);
```

14. <b><i>external_otp_tokens</i></b>
```sql
CREATE TABLE external_otp_tokens (
    email         TEXT PRIMARY KEY,
    otp_code      TEXT NOT NULL,
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at    TIMESTAMP WITH TIME ZONE NOT NULL,
    is_verified   BOOLEAN NOT NULL DEFAULT FALSE,
    attempts      INTEGER NOT NULL DEFAULT 0
);
```
 
<b>Grant CloudSQL Permissions to SA: </b>
```sql
GRANT SELECT, INSERT, UPDATE, DELETE
ON ALL TABLES IN SCHEMA public
TO "sa-drywall-api-dev@prj-fbm-drywall-dev.iam";
```

<b>Add CORS permission to GCS artifacts bucket</b>
```bash
> cd xtimator-3d
> gcloud storage buckets update gs://drywall-takeoff-artifacts-dev --cors-file=config/gcs_bucket_CORS.json
```
