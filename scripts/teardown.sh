#!/usr/bin/env bash
# GeekyClouds teardown. DEFAULT = DRY RUN (nothing is deleted).
#   bash scripts/teardown.sh             dry run: lists what exists, echoes every command
#   bash scripts/teardown.sh --execute   real deletion (asks for a typed confirmation)
#   KEEP_SNAPSHOT=0 bash scripts/teardown.sh --execute   skip the final RDS snapshot
set -u
export MSYS_NO_PATHCONV=1
export AWS_PAGER=""
R="ap-south-2"
ACCOUNT="704475327729"
CLUSTER="geekyclouds-cluster"
LB_ARN="arn:aws:elasticloadbalancing:ap-south-2:704475327729:loadbalancer/app/geekyclouds-alb/b7a7d757deec2073"
TG_ARN="arn:aws:elasticloadbalancing:ap-south-2:704475327729:targetgroup/gc-router-tg/5764dc72344106e7"
CM_SERVICE="srv-tpq5ekvbsuiu7qk7"
CM_NAMESPACE="ns-s3w3q27lwpzu4rje"
DB="geekyclouds-pgvector"
BUCKET="geekyclouds-vs-704475327729"
QUEUE_URL="https://sqs.ap-south-2.amazonaws.com/704475327729/geekyclouds-ingest-queue"
TOPIC_ARN="arn:aws:sns:ap-south-2:704475327729:geekyclouds-alerts"
ROLE="geekyclouds-ecs-task-exec-role"
KEEP_SNAPSHOT="${KEEP_SNAPSHOT:-1}"
EXECUTE=0
[ "${1:-}" = "--execute" ] && EXECUTE=1

run() {
  if [ "$EXECUTE" -eq 1 ]; then echo ">> $*"; "$@" || echo "   (failed, continuing - check leftovers)"; else echo "[DRY-RUN] $*"; fi
}
step() { echo; echo "### $1"; }
lst() { local label="$1"; shift; local out; out=$("$@" 2>&1); if [ -z "$out" ] || [ "$out" = "None" ]; then echo "  $label: (none)"; else echo "  $label: $out"; fi; }

inventory() {
  echo "===== $1 ====="
  lst "ECS services" aws ecs list-services --cluster "$CLUSTER" --region "$R" --query "serviceArns" --output text
  lst "ECS cluster" aws ecs describe-clusters --clusters "$CLUSTER" --region "$R" --query "clusters[?status!='INACTIVE'].clusterName" --output text
  lst "Autoscaling target" aws application-autoscaling describe-scalable-targets --service-namespace ecs --region "$R" --query "ScalableTargets[].ResourceId" --output text
  lst "ALB" aws elbv2 describe-load-balancers --region "$R" --query "LoadBalancers[?starts_with(LoadBalancerName,'geekyclouds')].LoadBalancerArn" --output text
  lst "Target groups" aws elbv2 describe-target-groups --region "$R" --query "TargetGroups[?starts_with(TargetGroupName,'gc-')].TargetGroupArn" --output text
  lst "Cloud Map namespace" aws servicediscovery list-namespaces --region "$R" --query "Namespaces[?Name=='geekyclouds.local'].Id" --output text
  lst "RDS instance" aws rds describe-db-instances --region "$R" --query "DBInstances[?starts_with(DBInstanceIdentifier,'geekyclouds')].DBInstanceIdentifier" --output text
  lst "RDS snapshots" aws rds describe-db-snapshots --region "$R" --query "DBSnapshots[?starts_with(DBInstanceIdentifier,'geekyclouds')].DBSnapshotIdentifier" --output text
  lst "Security groups" aws ec2 describe-security-groups --region "$R" --filters "Name=group-name,Values=geekyclouds-*" --query "SecurityGroups[].GroupId" --output text
  lst "S3 bucket" aws s3api list-buckets --query "Buckets[?starts_with(Name,'geekyclouds')].Name" --output text
  lst "SQS queue" aws sqs list-queues --region "$R" --queue-name-prefix geekyclouds --query "QueueUrls" --output text
  lst "ECR repos" aws ecr describe-repositories --region "$R" --query "repositories[?starts_with(repositoryName,'geekyclouds')].repositoryName" --output text
  lst "SNS topic" aws sns list-topics --region "$R" --query "Topics[?contains(TopicArn,'geekyclouds')].TopicArn" --output text
  lst "CloudWatch alarms" aws cloudwatch describe-alarms --region "$R" --query "MetricAlarms[?contains(AlarmName,'geekyclouds')].AlarmName" --output text
  lst "Log groups" aws logs describe-log-groups --region "$R" --log-group-name-prefix /ecs/geekyclouds --query "logGroups[].logGroupName" --output text
  lst "IAM role" aws iam list-roles --query "Roles[?starts_with(RoleName,'geekyclouds')].RoleName" --output text
  lst "OpenSearch domains" aws opensearch list-domain-names --region "$R" --query "DomainNames[].DomainName" --output text
  lst "Secrets Manager" aws secretsmanager list-secrets --region "$R" --query "SecretList[].Name" --output text
}

ACTUAL=$(aws sts get-caller-identity --query Account --output text 2>&1)
if [ "$ACTUAL" != "$ACCOUNT" ]; then echo "Wrong account or no credentials ($ACTUAL). Aborting."; exit 1; fi
echo "Account $ACCOUNT, region $R. Mode: $([ $EXECUTE -eq 1 ] && echo EXECUTE || echo DRY-RUN)"
inventory "BEFORE (resources that exist now)"

if [ "$EXECUTE" -eq 1 ]; then
  echo; echo "THIS WILL DELETE ALL GEEKYCLOUDS RESOURCES IN $R."
  printf "Type DELETE GEEKYCLOUDS to continue: "; read -r ans
  [ "$ans" = "DELETE GEEKYCLOUDS" ] || { echo "Aborted. Nothing deleted."; exit 1; }
fi

step "1. Autoscaling policy and scalable target"
run aws application-autoscaling delete-scaling-policy --policy-name gc-cpu-scale --service-namespace ecs --resource-id "service/$CLUSTER/shard-server" --scalable-dimension ecs:service:DesiredCount --region "$R"
run aws application-autoscaling deregister-scalable-target --service-namespace ecs --resource-id "service/$CLUSTER/shard-server" --scalable-dimension ecs:service:DesiredCount --region "$R"

step "2. ECS: scale to 0, delete services, cluster, task definitions"
for s in query-router shard-server; do run aws ecs update-service --cluster "$CLUSTER" --service "$s" --desired-count 0 --region "$R" --query "service.serviceName" --output text; done
for s in query-router shard-server; do run aws ecs delete-service --cluster "$CLUSTER" --service "$s" --force --region "$R" --query "service.serviceName" --output text; done
run aws ecs wait services-inactive --cluster "$CLUSTER" --services query-router shard-server --region "$R"
run aws ecs delete-cluster --cluster "$CLUSTER" --region "$R" --query "cluster.clusterName" --output text
for td in embedding-service:1 query-router:1 shard-server:1 shard-server:2; do run aws ecs deregister-task-definition --task-definition "$td" --region "$R" --query "taskDefinition.family" --output text; done

step "3. ALB and target group"
run aws elbv2 delete-load-balancer --load-balancer-arn "$LB_ARN" --region "$R"
run aws elbv2 wait load-balancers-deleted --load-balancer-arns "$LB_ARN" --region "$R"
run sleep 20
run aws elbv2 delete-target-group --target-group-arn "$TG_ARN" --region "$R"

step "4. Cloud Map (geekyclouds.local / shard-1)"
run aws servicediscovery delete-service --id "$CM_SERVICE" --region "$R"
run aws servicediscovery delete-namespace --id "$CM_NAMESPACE" --region "$R"

step "5. RDS (deletion protection is ON, so turn it off first)"
run aws rds modify-db-instance --db-instance-identifier "$DB" --no-deletion-protection --apply-immediately --region "$R" --query "DBInstance.DBInstanceIdentifier" --output text
run aws rds wait db-instance-available --db-instance-identifier "$DB" --region "$R"
if [ "$KEEP_SNAPSHOT" = "1" ]; then
  run aws rds delete-db-instance --db-instance-identifier "$DB" --final-db-snapshot-identifier "geekyclouds-pgvector-final-$(date +%Y%m%d)" --region "$R" --query "DBInstance.DBInstanceIdentifier" --output text
else
  run aws rds delete-db-instance --db-instance-identifier "$DB" --skip-final-snapshot --region "$R" --query "DBInstance.DBInstanceIdentifier" --output text
fi
run aws rds wait db-instance-deleted --db-instance-identifier "$DB" --region "$R"

step "6. Security groups (referencing groups first)"
run sleep 30
for sg in sg-0a565025608abde10 sg-05bf4be167ed4ebec sg-0fe0bebbc59e9a227 sg-0c8b0547befd71666; do run aws ec2 delete-security-group --group-id "$sg" --region "$R"; done

step "7. S3 bucket (empties it first)"
run aws s3 rb "s3://$BUCKET" --force

step "8. SQS queue"
run aws sqs delete-queue --queue-url "$QUEUE_URL" --region "$R"

step "9. ECR repositories"
for repo in geekyclouds/query-router geekyclouds/embedding-service geekyclouds/shard-index-server; do run aws ecr delete-repository --repository-name "$repo" --force --region "$R" --query "repository.repositoryName" --output text; done

step "10. SNS topic, CloudWatch alarms, log groups"
run aws sns delete-topic --topic-arn "$TOPIC_ARN" --region "$R"
run aws cloudwatch delete-alarms --alarm-names geekyclouds-alb-5xx geekyclouds-rds-low-storage --region "$R"
for lg in /ecs/geekyclouds/embedding-service /ecs/geekyclouds/query-router /ecs/geekyclouds/shard-server; do run aws logs delete-log-group --log-group-name "$lg" --region "$R"; done

step "11. IAM role $ROLE (the geekyclouds-dev user is NOT deleted)"
for p in $(aws iam list-attached-role-policies --role-name "$ROLE" --query "AttachedPolicies[].PolicyArn" --output text 2>/dev/null); do run aws iam detach-role-policy --role-name "$ROLE" --policy-arn "$p"; done
for p in $(aws iam list-role-policies --role-name "$ROLE" --query "PolicyNames" --output text 2>/dev/null); do run aws iam delete-role-policy --role-name "$ROLE" --policy-name "$p"; done
run aws iam delete-role --role-name "$ROLE"

step "12. Not handled by this script"
echo "  - RDS-managed secret rds!db-... is removed automatically with the instance."
echo "  - Pinecone (not AWS): delete indexes geekyclouds-ann-vector-smoke and geekyclouds-ann-benchmark in the Pinecone console."
echo "  - Final RDS snapshot (if kept) still bills storage until you delete it."
echo "  - IAM user geekyclouds-dev and its policies: detach/delete manually after submission."

echo
if [ "$EXECUTE" -eq 1 ]; then inventory "AFTER (anything listed here is a leftover)"; else echo "Dry run complete. Nothing was deleted."; fi