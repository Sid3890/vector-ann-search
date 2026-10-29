# Day 9 Load Test and Autoscaling Evidence (Track B)

Target: http://geekyclouds-alb-1509011703.ap-south-2.elb.amazonaws.com/search
Service: shard-server (ECS Fargate, 1 vCPU / 2 GiB), scalable target min 1 / max 3, policy gc-cpu-scale.

## Runs
| Run | Users | Duration | Requests | Failures | p95 | Report |
|---|---|---|---|---|---|---|
| Playbook run | ramp to 50 | 3 min | 7,429 | 0 | 53 ms | loadtest/report.html |
| Stress run | 100 | 2 min | 9,639 | 0 | 550 ms | loadtest/day9-load-100.html |
| Autoscaling run (temporary 15% target) | 100 | 5 min | 26,576 | 0 | 390 ms (p99 1,100 ms) | loadtest/day9-autoscaling-v2.html |

## Autoscaling evidence
- Production policy: ECSServiceAverageCPUUtilization target 60%, scale-out cooldown 60 s, scale-in cooldown 180 s, min 1 / max 3.
- At 50 and 100 users, average shard CPU peaked at about 10% and about 19-20% respectively, below the 60% target. No scale-out occurred at 60%.
- To test the mechanism, the target was temporarily lowered to 15%. With CPU at about 19% for 4 consecutive minutes (16:26-16:29 IST, 29 Oct 2026), AlarmHigh fired and Application Auto Scaling recorded "Setting desired count to 2" at 16:31:48 IST (loadtest/day9-v2-scaling-activities.json, loadtest/day9-v2-cpu.txt).
- The policy was then restored to 60% and verified (target 60, scale-in enabled, cooldowns 60/180).

## Limitations
- Scale-out was demonstrated only with the temporary 15% target, not at the production 60% target.
- The new task was requested after the load test ended, so the latency in these reports was measured on a single task. These runs do not show latency improving after scale-out.
- The router sends to one fixed Cloud Map DNS name (shard-1.geekyclouds.local), with no explicit fan-out across replicas, so extra replicas are not guaranteed an even share of traffic.

## Scale-in (observed)
- Application Auto Scaling recorded "Setting desired count to 1" at 16:34:01 IST, about 2 minutes after the scale-out, once the load had ended. ECS then stopped the extra task and reported a steady state at 1 running / 0 pending.
- Full sequence: 1 -> 2 at 16:31:48 (Successful), 2 -> 1 at 16:34:01 (Successful).
- The activity output does not record which alarm or policy setting triggered scale-in, so no claim is made about it.

## Day 9 conclusion
- Load test: complete. 50-user ramp (3 min) saved as loadtest/report.html, plus 100-user stress runs. Zero failures in all runs.
- Autoscaling: scale-out and scale-in both demonstrated, using a temporary 15% CPU target. The production 60% target was never reached by this workload (CPU peak about 19-20% at 100 users).
- Not demonstrated: latency improving after scale-out, because the new task was requested after the load ended.
- Managed target: Pinecone route chosen, API key and smoke test passed (index geekyclouds-ann-vector-smoke, 128-d). The full MANAGED benchmark is Track A's Day 9 deliverable.
- Production policy restored and verified: target 60, scale-in enabled, cooldowns 60/180.
