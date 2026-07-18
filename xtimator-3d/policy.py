from functools import partial
from fastapi.concurrency import run_in_threadpool

from helper import pg_run


__all__ = ["AccessControlService"]

class Admin:

    def __init__(self, role_name):
        self._role_name = role_name

    def __bool__(self):
        return self._role_name.lower().find("admin") != -1

    def local(self):
        return self._role_name.lower() == "admin"

    def super(self):
        return self._role_name.lower() == "super admin"

class Scope:

    def __init__(self, scopes):
        self._scopes = [scope.lower() for scope in scopes]

    def read(self):
        return "read" in self._scopes

    def write(self):
        return "write" in self._scopes

    def update(self):
        return "update" in self._scopes

    def delete(self):
        return "delete" in self._scopes

class AccessControlService:

    def __init__(self, credentials, pg_pool):
        self._credentials = credentials
        self._pg_pool = pg_pool

    async def _is_owner(self, project_id, plan_id, page_number, user_id):
        query = f"""
            SELECT user_id FROM {self._credentials["CloudSQL"]["table_name_models"]}
            WHERE LOWER(project_id) = LOWER(%s) AND LOWER(plan_id) = LOWER(%s) AND LOWER(page_number) = LOWER(%s);
        """
        user_owner = await run_in_threadpool(partial(pg_run, self._credentials, self._pg_pool, query, params=(project_id, plan_id, page_number,), fetch=True))
        return user_id.lower() == user_owner.lower()

    async def load_scope(self, user_id, project_id, plan_id, page_number):
        is_owner = await self._is_owner(project_id, plan_id, page_number, user_id)
        if is_owner:
            user_scopes = ["read", "write", "update", "delete"]
            return Scope(user_scopes)
        query = f"""
            SELECT
            p.permission_name as permission_name
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

            visible_regions AS (
                SELECT DISTINCT ur.region_id
                FROM {self._credentials["CloudSQL"]["table_name_user_regions"]} ur
                JOIN {self._credentials["CloudSQL"]["table_name_organization_regions"]} ogr
                    ON ogr.region_id = ur.region_id JOIN users u on u.user_id = ur.user_id
                WHERE LOWER(u.user_email) = LOWER(%s)
                    AND ogr.organization_id IN (
                        SELECT organization_id
                        FROM visible_organizations
                    )
            )

            SELECT DISTINCT
                u.user_email as user_email
            FROM {self._credentials["CloudSQL"]["table_name_users"]} u
            JOIN {self._credentials["CloudSQL"]["table_name_user_regions"]} ur
                ON ur.user_id = u.user_id
            WHERE ur.region_id IN (
                SELECT region_id
                FROM visible_regions
            )
            ORDER BY u.user_email;
        """
        visible_users = await run_in_threadpool(partial(pg_run, self._credentials, self._pg_pool, query, params=(user_id, user_id, user_id,), fetch=True))
        visible_users = [visible_user["user_email"].lower() for visible_user in visible_users]
        return visible_users

    async def load_organization_users(self, user_id):
        query = f"""
            SELECT u2.user_email as user_email
            FROM users u1
            JOIN users u2
                ON u2.organization_id = u1.organization_id
            WHERE u1.user_id = %s
            ORDER BY u2.user_email;
        """
        visible_users = await run_in_threadpool(partial(pg_run, self._credentials, self._pg_pool, query, params=(user_id,), fetch=True))
        visible_users = [visible_user["user_email"].lower() for visible_user in visible_users]
        return visible_users

    async def is_admin(self, user_id):
        query = f"""
            SELECT
            r.role_name as role_name
            FROM {self._credentials["CloudSQL"]["table_name_users"]} u
            JOIN {self._credentials["CloudSQL"]["table_name_roles"]} r
                ON u.role_id = r.role_id
            WHERE LOWER(u.user_email) = LOWER(%s);
        """
        role_name = await run_in_threadpool(partial(pg_run, self._credentials, self._pg_pool, query, params=(user_id,), fetch=True))
        return Admin(role_name[0]["role_name"])
