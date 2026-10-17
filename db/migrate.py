import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url


def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is not set")

    url = make_url(database_url)
    if url.drivername not in {"postgresql", "postgresql+psycopg2"}:
        raise RuntimeError("DATABASE_URL must use PostgreSQL")

    schema_path = Path(__file__).with_name("schema.sql")
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    schema_sql = schema_path.read_text(encoding="utf-8-sig")
    engine = create_engine(url, pool_pre_ping=True)

    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(schema_sql)

            result = connection.execute(
                text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_name IN ('vectors', 'shards', 'query_log')
                    ORDER BY table_name
                """)
            )
            tables = [row[0] for row in result]

            expected = {"vectors", "shards", "query_log"}
            missing = expected - set(tables)
            if missing:
                raise RuntimeError(
                    f"Schema migration completed without expected tables: {sorted(missing)}"
                )

        print("Schema migration completed successfully.")
        print("Verified tables: " + ", ".join(tables))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
