from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from influxdb_client import InfluxDBClient
from influxdb_client.rest import ApiException

from cleardrive.core.config import (
    DEFAULT_INFLUX_BUCKET,
    DEFAULT_INFLUX_WHITELIST_FIELD,
    DEFAULT_INFLUX_WHITELIST_MEASUREMENT,
    DEFAULT_WHITELIST_CACHE_TTL_SECONDS,
    env_int,
    env_optional_str,
    env_str,
)

NormalizePlate = Callable[[str], str | None]


@dataclass
class _CacheEntry:
    plates: set[str]
    fetched_at: datetime


class InfluxWhitelistCache:
    """Fetches whitelisted plates from InfluxDB Cloud and caches them in memory."""

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        org: str | None = None,
        bucket: str | None = None,
        measurement: str | None = None,
        field: str | None = None,
        query: str | None = None,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        self.url = url or _resolve_influx_url()
        self.token = token or env_optional_str("INFLUX_TOKEN")
        self.org = org or env_optional_str("INFLUX_ORG")
        self.bucket = bucket or env_optional_str("INFLUX_BUCKET") or DEFAULT_INFLUX_BUCKET
        self.measurement = measurement or env_str(
            "INFLUX_WHITELIST_MEASUREMENT", DEFAULT_INFLUX_WHITELIST_MEASUREMENT
        )
        self.field = field or env_str("INFLUX_WHITELIST_FIELD", DEFAULT_INFLUX_WHITELIST_FIELD)
        self.query = query or env_optional_str("INFLUX_WHITELIST_QUERY")
        self.cache_ttl_seconds = (
            cache_ttl_seconds
            if cache_ttl_seconds is not None
            else env_int("WHITELIST_CACHE_TTL_SECONDS", DEFAULT_WHITELIST_CACHE_TTL_SECONDS)
        )
        self._cache: _CacheEntry | None = None
        self._client: InfluxDBClient | None = None

    def setup(self) -> None:
        if self._client is not None:
            return

        if not self.token:
            raise ValueError("INFLUX_TOKEN is required to fetch the whitelist from InfluxDB")

        if not self.url:
            raise ValueError(
                "INFLUX_URL is required. Copy your InfluxDB Cloud URL from the same page "
                "where you generated the API token."
            )

        discovery_client = InfluxDBClient(url=self.url, token=self.token)
        try:
            if self.org is None:
                self.org = self._discover_org(discovery_client)
            if self.bucket is None:
                self.bucket = self._discover_bucket(discovery_client)
        finally:
            discovery_client.close()

        self._client = InfluxDBClient(url=self.url, token=self.token, org=self.org)

    def teardown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def get_plates(self, normalize: NormalizePlate) -> set[str]:
        """Return cached whitelist plates, refreshing from InfluxDB when expired."""
        if self._cache is not None and not self._is_expired():
            return self._cache.plates

        self._refresh(normalize)
        return self._cache.plates if self._cache is not None else set()

    def _is_expired(self) -> bool:
        if self._cache is None:
            return True

        age = datetime.now(timezone.utc) - self._cache.fetched_at
        return age >= timedelta(seconds=self.cache_ttl_seconds)

    def _refresh(self, normalize: NormalizePlate) -> None:
        if self._client is None:
            self.setup()

        assert self._client is not None

        query = self._build_query()
        tables = self._client.query_api().query(query)

        plates: set[str] = set()
        for table in tables:
            for record in table.records:
                value = record.get_value()
                if not isinstance(value, str):
                    continue

                plate = normalize(value)
                if plate is not None:
                    plates.add(plate)

        self._cache = _CacheEntry(plates=plates, fetched_at=datetime.now(timezone.utc))

    def _discover_org(self, client: InfluxDBClient) -> str:
        try:
            orgs = client.organizations_api().find_organizations()
        except ApiException as exc:
            raise ValueError(
                "Could not discover InfluxDB organization from INFLUX_TOKEN. "
                "Check INFLUX_URL and token permissions."
            ) from exc

        if not orgs:
            raise ValueError("No InfluxDB organization found for INFLUX_TOKEN.")

        if len(orgs) == 1:
            return orgs[0].id

        org_names = ", ".join(org.name for org in orgs)
        raise ValueError(
            "Multiple InfluxDB organizations are available. Set INFLUX_ORG to one of: "
            f"{org_names}"
        )

    def _discover_bucket(self, client: InfluxDBClient) -> str:
        try:
            buckets = client.buckets_api().find_buckets()
        except ApiException as exc:
            raise ValueError(
                "Could not discover InfluxDB bucket from INFLUX_TOKEN. "
                "Set INFLUX_BUCKET explicitly if needed."
            ) from exc

        readable = [
            bucket.name
            for bucket in buckets.buckets or []
            if bucket.name and not bucket.name.startswith("_")
        ]

        if not readable:
            raise ValueError("No readable InfluxDB buckets found for INFLUX_TOKEN.")

        if len(readable) == 1:
            return readable[0]

        preferred = next(
            (name for name in readable if name.lower() == DEFAULT_INFLUX_BUCKET.lower()),
            None,
        )
        if preferred is not None:
            return preferred

        preferred = next((name for name in readable if name == self.measurement), None)
        if preferred is not None:
            return preferred

        bucket_names = ", ".join(readable)
        raise ValueError(
            "Multiple InfluxDB buckets are available. Set INFLUX_BUCKET to one of: "
            f"{bucket_names}"
        )

    def _build_query(self) -> str:
        if self.query:
            return self.query

        assert self.bucket is not None

        return (
            f'from(bucket: "{self.bucket}")\n'
            f'  |> range(start: -90d)\n'
            f'  |> filter(fn: (r) => r._measurement == "{self.measurement}")\n'
            f'  |> filter(fn: (r) => r._field == "{self.field}")\n'
            f'  |> keep(columns: ["_value"])\n'
            f'  |> distinct(column: "_value")'
        )


def _resolve_influx_url() -> str | None:
    for key in ("INFLUX_URL", "INFLUXDB_URL", "INFLUXDB_HOST"):
        value = env_optional_str(key)
        if value is not None:
            return value
    return None
