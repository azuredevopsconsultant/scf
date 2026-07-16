#!/usr/bin/env bash
# scripts/setup_secret_scopes.sh
#
# Creates Databricks secret scopes and populates secrets for each environment.
# Run once per workspace after initial provisioning.
# Authentication: databricks CLI must already be logged in (profile or env var).
#
# Usage:
#   ./scripts/setup_secret_scopes.sh dev
#   ./scripts/setup_secret_scopes.sh preprod
#   ./scripts/setup_secret_scopes.sh prod
#
# GitHub Variables to configure (Settings → Environments → Variables):
#   Per environment (dev / preprod / prod):
#     DATABRICKS_HOST_DEV        https://dbc-xxxx.cloud.databricks.com
#     DATABRICKS_HOST_PREPROD    https://dbc-xxxx.cloud.databricks.com
#     DATABRICKS_HOST_PROD       https://dbc-xxxx.cloud.databricks.com
#     DATABRICKS_CLIENT_ID_DEV        <service-principal-client-id>
#     DATABRICKS_CLIENT_ID_PREPROD    <service-principal-client-id>
#     DATABRICKS_CLIENT_ID_PROD       <service-principal-client-id>
#   Shared (repo-level):
#     (none — no secrets stored in GitHub for this stack)

set -euo pipefail

ENV="${1:-dev}"

case "$ENV" in
  dev)
    SCOPE="scf-cohort-dev"
    PROFILE="${DATABRICKS_PROFILE:-dbc-a2b9471c-0cac}"
    ;;
  preprod)
    SCOPE="scf-cohort-preprod"
    PROFILE="${DATABRICKS_PROFILE:-dbc-preprod}"
    ;;
  prod)
    SCOPE="scf-cohort-prod"
    PROFILE="${DATABRICKS_PROFILE:-dbc-prod}"
    ;;
  *)
    echo "Usage: $0 <dev|preprod|prod>"
    exit 1
    ;;
esac

echo "Setting up secret scope '$SCOPE' for environment '$ENV' ..."

# ── Create scope (idempotent: ignore error if already exists) ─────────────
databricks secrets create-scope "$SCOPE" --profile "$PROFILE" 2>/dev/null || \
  echo "Scope '$SCOPE' already exists — continuing."

# ── SMTP credentials (read from terminal — never stored in script) ────────
echo ""
echo "Enter SMTP credentials for scope '$SCOPE':"
read -rsp "  smtp-user     : " SMTP_USER;     echo
read -rsp "  smtp-password : " SMTP_PASS;     echo

databricks secrets put-secret "$SCOPE" smtp-user     \
  --string-value "$SMTP_USER"     --profile "$PROFILE"
databricks secrets put-secret "$SCOPE" smtp-password \
  --string-value "$SMTP_PASS"     --profile "$PROFILE"

# ── Teams webhook (optional) ──────────────────────────────────────────────
echo ""
read -rsp "  webhook-url (Teams/Slack, leave blank to skip): " WEBHOOK; echo
if [[ -n "$WEBHOOK" ]]; then
  databricks secrets put-secret "$SCOPE" webhook-url \
    --string-value "$WEBHOOK" --profile "$PROFILE"
fi

echo ""
echo "Done. Secrets written to scope '$SCOPE':"
databricks secrets list-secrets "$SCOPE" --profile "$PROFILE"
