# GeekyClouds RDS PostgreSQL Notes

## Configuration

- AWS Region: `ap-south-2`
- DB instance identifier: `geekyclouds-pgvector`
- Engine: PostgreSQL 16.13
- Instance class: `db.t3.micro`
- Storage: 20 GiB gp3, encrypted
- Availability: Single-AZ
- Database: `geekyclouds`
- Master username: `geekyclouds`
- pgvector extension: 0.8.1
- Deletion protection: enabled
- Backup retention: 1 day

## Network access

- VPC: `vpc-098447ee2dc88373b`
- DB subnet group: `geekyclouds-pgvector-subnets`
- Security group: `sg-0a565025608abde10`
- PostgreSQL port: `5432`
- Inbound access is restricted to the developer's public IPv4 address at setup time.

If the developer's public IP changes, update the security-group rule to the new trusted IP. Do not allow `0.0.0.0/0` for PostgreSQL.

## Credentials

RDS manages the master password through AWS Secrets Manager. Retrieve it only when needed, using an authorized AWS identity. Never commit credentials or paste the password into logs, source files, or documentation.

## Schema

The schema is defined in `db/schema.sql`. It creates:

- `vectors`
- `shards`
- `query_log`

The `vectors.embedding` and `shards.centroid` columns use `vector(384)` to match the embedding service.

## Apply or verify the schema

Set `DATABASE_URL` in the environment to the RDS PostgreSQL connection string, then run:

```powershell
py -m pip install -r .\ingestion\requirements.txt
py .\db\migrate.py
```

Use a URL-encoded password if it contains reserved URL characters. Keep the connection string out of source control.

## Verified on Day 5

- RDS instance reached `available`.
- TCP connectivity to port 5432 succeeded from the developer machine.
- PostgreSQL 16.13 connection succeeded with SSL required.
- pgvector extension 0.8.1 was enabled.
- `db/schema.sql` ran successfully against RDS.
- The `vectors`, `shards`, and `query_log` tables were present.
- The embedding columns were verified as `vector(384)`.

## Cost and cleanup

RDS incurs ongoing instance and storage charges. Check AWS Billing for actual costs. Deletion protection is enabled, so disable it deliberately before deleting the instance if the project is finished.
