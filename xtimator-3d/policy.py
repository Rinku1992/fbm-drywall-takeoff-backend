from functools import partial
from fastapi.concurrency import run_in_threadpool

from helper import pg_run


class AccessControlService:

    def __init__(self, credentials, pg_pool):
        self._credentials = credentials
        self._pg_pool = pg_pool

    async def _load_scope(self, user_id):
        query = f"""
            SELECT
            p.name as permission_name
            FROM users u
            JOIN roles r
                ON u.role_id = r.role_id
            JOIN role_permissions rp
                ON r.role_id = rp.role_id
            JOIN permissions p
                ON p.permission_id = rp.permission_id
            WHERE LOWER(u.user_email) = LOWER(%s);
        """
        await run_in_threadpool(partial(pg_run, self._credentials, self._pg_pool, query, params=(user_id,), fetch=True))
        user_scopes = await run_in_threadpool(partial(pg_run, self._credentials, self._pg_pool, query, params=(user_id,), fetch=True))
        return user_scopes

    async def load_visible_users(self, user_id):
        query = f"""
            WITH visible_organizations AS (
                SELECT organization_id
                FROM {self._credentials["CloudSQL"]["table_name_users"]}
                WHERE LOWER(user_email) = LOWER(%s)

                UNION

                SELECT up.organization_id
                FRO {self._credentials["CloudSQL"]["table_name_user_partner_organizations"]} up JOIN users u ON up.user_id = u.user_id
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
                u.user_id,
                u.user_email,
                u.user_name,
                u.organization_id
            FROM {self._credentials["CloudSQL"]["table_name_users"]} u
            JOIN {self._credentials["CloudSQL"]["table_name_user_regions"]} ur
                ON ur.user_id = u.user_id
            WHERE ur.region_id IN (
                SELECT region_id
                FROM visible_regions
            )
            ORDER BY u.user_email;
        """
        visible_users = await run_in_threadpool(partial(pg_run, self._credentials, self._pg_pool, query, params=(user_id,), fetch=True))
        return visible_users
