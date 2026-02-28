# Operational Guidelines

## Security Do's
- Rotate JWT secrets and store them in a secret manager.
- Put API behind TLS and gateway-level authn/authz.
- Add account lockout, throttling, and CAPTCHA for auth endpoints.
- Record audit logs for login attempts and admin operations.

## Security Don'ts
- Don’t keep default JWT secret values in production.
- Don’t expose development auth implementation to internet traffic.
- Don’t log raw passwords, tokens, or secret headers.

## Performance Do's
- Benchmark planner output against real hardware before committing topology.
- Monitor NVMe throughput and p99 latency under realistic traffic.
- Use pinned memory and asynchronous prefetch in runtime implementation.
- Separate control-plane API and inference runtime autoscaling groups.

## Performance Don'ts
- Don’t co-locate noisy background jobs with latency-critical inference workers.
- Don’t size GPU count using only peak VRAM without transfer-rate constraints.
- Don’t scale batch size blindly; validate tail latency and token throughput.

## Reliability Do's
- Add structured logs + metrics + traces in every API and worker.
- Use retries with jitter for transient storage/network faults.
- Build idempotent job submission and deterministic run identifiers.
- Add health probes for storage subsystem and model-loading workers.

## Reliability Don'ts
- Don’t rely on process-local memory store for user/session state.
- Don’t deploy without backup/restore path for metadata databases.
- Don’t skip chaos or failure-injection tests for storage degradation.
