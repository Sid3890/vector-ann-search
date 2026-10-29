# Day 8 — Autoscaling and Monitoring

## ECS shard autoscaling

- ECS cluster: geekyclouds-cluster
- ECS service: shard-server
- Scaling metric: ECSServiceAverageCPUUtilization
- Target average CPU utilization: 60%
- Minimum task count: 1
- Maximum task count: 3
- Scale-out cooldown: 60 seconds
- Scale-in cooldown: 180 seconds

## Rationale

A 60% average CPU target leaves capacity headroom before sustained CPU utilization reaches saturation. The minimum of one task limits idle compute consumption, while the maximum of three provides bounded scale-out for this development deployment.

Cooldown periods help avoid overly frequent scaling adjustments. Actual scaling behavior depends on CloudWatch metric evaluation and workload characteristics.

## Monitoring

CloudWatch alarms:
- geekyclouds-alb-5xx: alerts on target-group HTTP 5xx errors.
- geekyclouds-rds-low-storage: alerts when RDS free storage drops below 2 GiB.

Notifications are configured through the SNS topic:
- geekyclouds-alerts

## Verification status

- Scalable target registration succeeded.
- CPU target-tracking policy creation succeeded.
- CloudWatch alarm actions are enabled.
- End-to-end vector search was verified on Day 7.
- Scale-out under load still needs to be tested.
- SNS email delivery should be confirmed.

## IAM

The ECS task execution role has:
- AmazonECSTaskExecutionRolePolicy
- GeekyCloudsReadRDSSecret

The inline policy resource scope must be reviewed before declaring IAM hardening complete.
