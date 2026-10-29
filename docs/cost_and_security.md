# Cost Analysis and Security Review (Track B, Day 10)

Account 704475327729, region ap-south-2. Prepared 29 Oct 2026.

## 1. Actual spend from Cost Explorer (1-29 Oct 2026)

| Item | USD |
|---|---|
| Gross usage | 10.21 |
| Credits applied | -10.21 |
| **Net charged** | **0.00** |

Gross usage by service: RDS 8.63, VPC 1.41, Secrets Manager 0.15, ECR 0.02. September usage was about 0.00. Raw queries are saved in docs/ce-by-record-type.json and docs/ce-by-service-nocredit.json.

Deviations and caveats, stated plainly:
- **Tag filter not possible.** No resource carries the Project=GeekyClouds tag (resourcegroupstaggingapi returned an empty list) and the tag-filtered Cost Explorer query returned nothing. The playbook's tagging rule was not followed. The figures above are account-level, not tag-filtered. The account holds only this project's resources.
- **Fargate and ALB are not in Cost Explorer yet.** The ALB was created 29 Oct 07:08 UTC and the ECS services on 29 Oct (12:49-12:51 IST). This is consistent with Cost Explorer reporting lag (up to about 24 hours). Their costs in section 2 are estimates and should be re-checked in Cost Explorer after the lag.
- The VPC charge is probably public IPv4 address hours (assignPublicIp is ENABLED, there are no NAT gateways). That attribution is an inference.

## 2. Sustained cost estimate (resource-hours x on-demand, ap-south-2)

| Component | Basis | USD per hour |
|---|---|---|
| query-router task | 0.5 vCPU, 1 GB at 0.04256 per vCPU-h and 0.00466 per GB-h | 0.0259 |
| shard-server task (1 task) | 1 vCPU, 2 GB | 0.0519 |
| ALB | fixed hourly rate, LCU charges excluded | 0.0239 |
| RDS db.t3.micro, 20 GB gp3 | measured: 8.63 USD over about 12 days (about 0.72 per day) | 0.0300 |
| **Total (1 shard task)** | | **0.1317** |

Projected monthly (730 h): about 96 USD. Fargate and ALB rates were taken from the AWS Price List API. Excluded: ALB LCU charges, data transfer, CloudWatch logs, public IPv4 hours, Secrets Manager. The playbook also asks for a cross-check in the AWS Pricing Calculator, which was not done here, so these figures rest on the Price List API and Cost Explorer only.

## 3. Cost per query

Method: hourly stack cost divided by queries per hour at the measured throughput (Locust, ALB /search, zero failures in all runs).

| Scenario | Throughput | Queries per hour | Cost per query (USD) | Cost per 1M queries (USD) |
|---|---|---|---|---|
| 50 users, 1 shard task (p95 53 ms) | 41.46 req/s | 149,256 | 8.82E-07 | 0.88 |
| 100 users, 5 min run (p95 390 ms, 1 to 2 tasks) | 88.91 req/s | 320,076 | 4.11E-07 | 0.41 |

The second row prices a single shard task, although the shard briefly scaled to 2 tasks in that run, so it slightly understates the cost of that run.

These are costs at sustained saturation. Actual spend per query during the project is far higher because the stack was mostly idle: the three recorded Locust runs total 43,644 requests.

Managed target: Pinecone free serverless tier. Only a smoke test was run (index geekyclouds-ann-vector-smoke, 128-d, 5 vectors), at 0 USD. The full MANAGED benchmark and its cost-per-query belong to Track A.

## 4. Security review: least-privilege decisions

Done:
- **RDS security group** sg-0a565025608abde10 allows port 5432 only from the ECS task security groups sg-05bf4be167ed4ebec (shard) and sg-0fe0bebbc59e9a227 (router). The developer laptop IP rule was removed on 29 Oct after /healthz returned 200.
- **IAM:** PowerUserAccess detached from geekyclouds-dev. No user, group or role holds it.
- **S3** bucket geekyclouds-vs-704475327729 has all four Block Public Access settings enabled.
- **ECS task execution role** geekyclouds-ecs-task-exec-role has AmazonECSTaskExecutionRolePolicy plus a scoped inline policy for reading the RDS secret.
- **Secrets:** the database credential is an RDS-managed secret injected through the ECS task-definition secret RDS_SECRET, not hardcoded.
- **Network:** only the ALB security group accepts 0.0.0.0/0 (port 80). The router accepts traffic only from the ALB SG, and the shard only from the router SG.
- **RDS** has deletion protection on, 1-day automated backups, no Multi-AZ (cost choice).

Residual risks (not fixed):
- geekyclouds-dev still holds AdministratorAccess, AmazonEC2FullAccess, AmazonRDSFullAccess, AmazonS3FullAccess, AmazonSQSFullAccess and IAMReadOnlyAccess. These were kept so teardown and cost queries keep working. Detach them after submission.
- RDS is still publicly accessible (PubliclyAccessible = true). Access is limited by the security group, but private-only access would be stronger.
- ECS tasks run with public IPs and no NAT gateway, to avoid NAT cost. Inbound access is controlled by security groups.
- The ALB listens on HTTP port 80 only, with no TLS.
- No resources carry the Project=GeekyClouds tag (see section 1).

## 5. Teardown

scripts/teardown.sh is a dry run by default. `bash scripts/teardown.sh` lists every resource and echoes each delete command, in dependency order, without deleting anything. `bash scripts/teardown.sh --execute` deletes, after a typed confirmation. A final RDS snapshot is kept unless KEEP_SNAPSHOT=0. The dry run output is in docs/teardown_dryrun_output.txt and listed all project resources, including task definitions, the Cloud Map namespace, log groups, alarms, the SNS topic and the task execution role. Not covered by the script: Pinecone indexes (delete in the Pinecone console) and the geekyclouds-dev IAM user.