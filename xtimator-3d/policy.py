from functools import partial
from fastapi.concurrency import run_in_threadpool

from helper import pg_run


__all__ = ["AccessControlService"]

class Admin:

    def __init__(self, role_name):
        self._role_name = role_name

    def __bool__(self):
        return self._role_name.lower().find("admin") != -1

    @property
    def local(self):
        return self._role_name.lower() == "admin"

    @property
    def super(self):
        return self._role_name.lower() == "super admin"

class Scope:

    def __init__(self, scopes):
        self._scopes = [scope.lower() for scope in scopes]

    @property
    def read(self):
        return "read" in self._scopes

    @property
    def write(self):
        return "write" in self._scopes

    @property
    def update(self):
        return "update" in self._scopes

    @property
    def delete(self):
        return "delete" in self._scopes

class AccessControlService:

    def __init__(self, credentials, pg_pool):
        self._credentials = credentials
        self._pg_pool = pg_pool

    async def _is_owner(self, project_id, plan_id, page_number, user_id):
        query = f"""
            SELECT user_id FROM {self._credentials["CloudSQL"]["table_name_models"]}
            WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND page_number = %s LIMIT 1;
        """
        user_owner = await run_in_threadpool(partial(pg_run, self._credentials, self._pg_pool, query, params=(project_id, plan_id, page_number,), fetch=True))
        return user_id.lower() == user_owner[0]["user_id"].lower()

    async def load_scope(self, user_id, project_id, plan_id, page_number):
        is_owner = await self._is_owner(project_id, plan_id, page_number, user_id)
        if is_owner:
            user_scopes = ["read", "write", "update", "delete"]
            return Scope(user_scopes)
        query = f"""
            SELECT
            p.permission_name AS permission_name
            FROM {self._credentials["CloudSQL"]["table_name_users"]} u
            JOIN {self._credentials["CloudSQL"]["table_name_roles"]} r
                ON u.role_id = r.role_id
            JOIN {self._credentials["CloudSQL"]["table_name_role_permissions"]} rp
                ON r.role_id = rp.role_id
            JOIN {self._credentials["CloudSQL"]["table_name_permissions"]} p
                ON p.permission_id = rp.permission_id
            WHERE LOWER(u.user_email) = LOWER(%s);
        """
        user_scopes = await run_in_threadpool(partial(pg_run, self._credentials, self._pg_pool, query, params=(user_id,), fetch=True))
        user_scopes = [user_scope["permission_name"] for user_scope in user_scopes]
        return Scope(user_scopes)

    async def load_regional_users(self, user_id):
        query = f"""
            WITH visible_organizations AS (
                SELECT organization_id
                FROM {self._credentials["CloudSQL"]["table_name_users"]}
                WHERE LOWER(user_email) = LOWER(%s)

                UNION

                SELECT up.organization_id
                FROM {self._credentials["CloudSQL"]["table_name_user_partner_organizations"]} up JOIN {self._credentials["CloudSQL"]["table_name_users"]} u ON up.user_id = u.user_id
                WHERE LOWER(u.user_email) = LOWER(%s)
            ),

            user_region_groups AS (
                SELECT DISTINCT
                    g.group_id
                FROM {self._credentials["CloudSQL"]["table_name_groups"]} g
                JOIN {self._credentials["CloudSQL"]["table_name_group_keys"]} gk
                    ON gk.group_key_id = g.group_key_id
                JOIN {self._credentials["CloudSQL"]["table_name_group_entities"]} ge
                    ON ge.group_id = g.group_id
                WHERE LOWER(gk.group_key_name) = 'region'
            ),

            visible_regions AS (
                SELECT DISTINCT
                    ge.region_id
                FROM {self._credentials["CloudSQL"]["table_name_group_entities"]} ge
                JOIN user_region_groups urg
                    ON urg.group_id = ge.group_id
                JOIN {self._credentials["CloudSQL"]["table_name_organization_regions"]} ogr
                    ON ogr.region_id = ge.region_id
                WHERE ge.region_id IS NOT NULL
                  AND ogr.organization_id IN (
                      SELECT organization_id
                      FROM visible_organizations
                  )
            )

            SELECT DISTINCT
                u.user_email AS user_email
            FROM {self._credentials["CloudSQL"]["table_name_users"]} u
            JOIN {self._credentials["CloudSQL"]["table_name_user_regions"]} ur
                ON ur.user_id = u.user_id
            JOIN visible_regions vr
                ON vr.region_id = ur.region_id
            WHERE u.organization_id IN (
                SELECT organization_id
                FROM visible_organizations
            )
            ORDER BY u.user_email;
        """
        visible_users = await run_in_threadpool(partial(pg_run, self._credentials, self._pg_pool, query, params=(user_id, user_id,), fetch=True))
        visible_users = [visible_user["user_email"].lower() for visible_user in visible_users]
        if visible_users:
            return visible_users
        return [user_id]

    async def load_regional_users_partner_organizations(self, user_id):
        query = f"""
            WITH partner_organizations AS (
                SELECT up.organization_id
                FROM {self._credentials["CloudSQL"]["table_name_user_partner_organizations"]} up JOIN {self._credentials["CloudSQL"]["table_name_users"]} u ON up.user_id = u.user_id
                WHERE LOWER(u.user_email) = LOWER(%s)
            ),

            user_region_groups AS (
                SELECT DISTINCT
                    g.group_id
                FROM {self._credentials["CloudSQL"]["table_name_groups"]} g
                JOIN {self._credentials["CloudSQL"]["table_name_group_keys"]} gk
                    ON gk.group_key_id = g.group_key_id
                JOIN {self._credentials["CloudSQL"]["table_name_group_entities"]} ge
                    ON ge.group_id = g.group_id
                WHERE LOWER(gk.group_key_name) = 'region'
            ),

            partner_regions AS (
                SELECT DISTINCT
                    ge.region_id
                FROM {self._credentials["CloudSQL"]["table_name_group_entities"]} ge
                JOIN user_region_groups urg
                    ON urg.group_id = ge.group_id
                JOIN {self._credentials["CloudSQL"]["table_name_organization_regions"]} ogr
                    ON ogr.region_id = ge.region_id
                WHERE ge.region_id IS NOT NULL
                  AND ogr.organization_id IN (
                      SELECT organization_id
                      FROM partner_organizations
                  )
            )

            SELECT DISTINCT
                u.user_email AS user_email
            FROM {self._credentials["CloudSQL"]["table_name_users"]} u
            JOIN {self._credentials["CloudSQL"]["table_name_user_regions"]} ur
                ON ur.user_id = u.user_id
            JOIN partner_regions vr
                ON vr.region_id = ur.region_id
            WHERE u.organization_id IN (
                SELECT organization_id
                FROM partner_organizations
            )
            ORDER BY u.user_email;
        """
        partner_users = await run_in_threadpool(partial(pg_run, self._credentials, self._pg_pool, query, params=(user_id,), fetch=True))
        partner_users = [partner_user["user_email"].lower() for partner_user in partner_users]
        return partner_users

    async def load_grouped_users(self, user_id):
        query = f"""
            WITH visible_organizations AS (
                SELECT organization_id
                FROM {self._credentials["CloudSQL"]["table_name_users"]}
                WHERE LOWER(user_email) = LOWER(%s)

                UNION

                SELECT up.organization_id
                FROM {self._credentials["CloudSQL"]["table_name_user_partner_organizations"]} up JOIN {self._credentials["CloudSQL"]["table_name_users"]} u ON up.user_id = u.user_id
                WHERE LOWER(u.user_email) = LOWER(%s)
            ),

            user_groups AS (
                SELECT DISTINCT
                    g.group_id
                FROM {self._credentials["CloudSQL"]["table_name_groups"]} g
                JOIN {self._credentials["CloudSQL"]["table_name_group_keys"]} gk
                    ON gk.group_key_id = g.group_key_id
                JOIN group_entities ge
                    ON ge.group_id = g.group_id
                WHERE ge.user_id = (SELECT user_id from users WHERE LOWER(user_email) = LOWER(%s))
                  AND LOWER(gk.group_key_name) = 'user_email'
            ),

            grouped_users AS (
                SELECT DISTINCT
                    ge.user_id
                FROM {self._credentials["CloudSQL"]["table_name_group_entities"]} ge
                JOIN user_groups ug
                    ON ug.group_id = ge.group_id
                WHERE ge.user_id IS NOT NULL
            )

            SELECT DISTINCT
                u.user_email AS user_email
            FROM users u
            JOIN grouped_users gu
                ON gu.user_id = u.user_id
            WHERE u.organization_id IN (
                SELECT organization_id
                FROM visible_organizations
            )
            ORDER BY u.user_email;
        """
        visible_users = await run_in_threadpool(partial(pg_run, self._credentials, self._pg_pool, query, params=(user_id, user_id, user_id,), fetch=True))
        visible_users = [visible_user["user_email"].lower() for visible_user in visible_users]
        if visible_users:
            return visible_users
        return [user_id]

    async def load_grouped_users_partner_organizations(self, user_id):
        query = f"""
            WITH partner_organizations AS (
                SELECT up.organization_id
                FROM user_partner_organizations up JOIN users u ON up.user_id = u.user_id
                WHERE LOWER(u.user_email) = LOWER(%s)
            ),

            user_groups AS (
                SELECT DISTINCT
                    g.group_id
                FROM groups g
                JOIN group_keys gk
                    ON gk.group_key_id = g.group_key_id
                JOIN group_entities ge
                    ON ge.group_id = g.group_id
                WHERE ge.user_id = (SELECT user_id from users WHERE LOWER(user_email) = LOWER(%s))
                  AND LOWER(gk.group_key_name) = 'user_email'
            ),

            grouped_users AS (
                SELECT DISTINCT
                    ge.user_id
                FROM group_entities ge
                JOIN user_groups ug
                    ON ug.group_id = ge.group_id
                WHERE ge.user_id IS NOT NULL
            )

            SELECT DISTINCT
                u.user_email AS user_email
            FROM users u
            JOIN grouped_users gu
                ON gu.user_id = u.user_id
            WHERE u.organization_id IN (
                SELECT organization_id
                FROM partner_organizations
            )
            ORDER BY u.user_email;
        """
        partner_users = await run_in_threadpool(partial(pg_run, self._credentials, self._pg_pool, query, params=(user_id, user_id,), fetch=True))
        partner_users = [partner_user["user_email"].lower() for partner_user in partner_users]
        if partner_users:
            return partner_users
        return [user_id]

    async def load_user_region_names(self, user_id):
        query = f"""
            SELECT
                DISTINCT(r.region_name) AS region_name
            FROM {self._credentials["CloudSQL"]["table_name_users"]} u
            JOIN {self._credentials["CloudSQL"]["table_name_user_regions"]} ur
                ON u.user_id = ur.user_id
            JOIN {self._credentials["CloudSQL"]["table_name_regions"]} r
                ON r.region_id = ur.region_id
            WHERE LOWER(u.user_email) = LOWER(%s);
        """
        region_names = await run_in_threadpool(partial(pg_run, self._credentials, self._pg_pool, query, params=(user_id,), fetch=True))
        region_names = [region_name["region_name"].lower() for region_name in region_names]
        if region_names:
            return region_names

        query = f"""
            SELECT
                DISTINCT(r.region_name) AS region_name
            FROM {self._credentials["CloudSQL"]["table_name_users"]} u
            JOIN {self._credentials["CloudSQL"]["table_name_organization_regions"]} org
                ON u.organization_id = org.organization_id
            JOIN {self._credentials["CloudSQL"]["table_name_regions"]} r
                ON r.region_id = org.region_id
            WHERE LOWER(u.user_email) = LOWER(%s);
        """
        region_names = await run_in_threadpool(partial(pg_run, self._credentials, self._pg_pool, query, params=(user_id,), fetch=True))
        region_names = [region_name["region_name"].lower() for region_name in region_names]
        return region_names

    async def load_organization_users(self, user_id):
        query = f"""
            SELECT u2.user_email AS user_email
            FROM {self._credentials["CloudSQL"]["table_name_users"]} u1
            JOIN {self._credentials["CloudSQL"]["table_name_users"]} u2
                ON u2.organization_id = u1.organization_id
            WHERE LOWER(u1.user_email) = LOWER(%s)
            ORDER BY u2.user_email;
        """
        visible_users = await run_in_threadpool(partial(pg_run, self._credentials, self._pg_pool, query, params=(user_id,), fetch=True))
        visible_users = [visible_user["user_email"].lower() for visible_user in visible_users]
        return visible_users

    async def is_admin(self, user_id):
        query = f"""
            SELECT
            r.role_name AS role_name
            FROM {self._credentials["CloudSQL"]["table_name_users"]} u
            JOIN {self._credentials["CloudSQL"]["table_name_roles"]} r
                ON u.role_id = r.role_id
            WHERE LOWER(u.user_email) = LOWER(%s);
        """
        role_name = await run_in_threadpool(partial(pg_run, self._credentials, self._pg_pool, query, params=(user_id,), fetch=True))
        return Admin(role_name[0]["role_name"])
