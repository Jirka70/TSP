# Ansible deployment

This directory contains the server-side deployment automation for the TSP Docker image.

## Files

- `inventory.example.ini` shows the expected host group. Copy it to `inventory.ini` for local runs.
- `group_vars/staging.yml` stores shared staging variables such as `/opt/tsp` paths.
- `playbooks/provision.yml` prepares the host with Docker, users, groups, and persistent directories.
- `playbooks/deploy.yml` pulls an immutable GHCR image tag, writes `/opt/tsp/bin/tsp-run`, and runs a smoke test.
- `deploy-container.yml` is a compatibility entrypoint that imports `playbooks/deploy.yml`.

## Local usage

```bash
cp ansible/inventory.example.ini ansible/inventory.ini
ansible-playbook ansible/playbooks/provision.yml
TSP_IMAGE=ghcr.io/<owner>/tsp-eeg-classification:sha-<commit> ansible-playbook ansible/playbooks/deploy.yml
```

The GitHub Actions deploy workflow creates `inventory.ini` at runtime from repository secrets.

## GitHub environment secrets

The `staging` environment must define:

- `NUADA_SSH_HOST`
- `NUADA_SSH_PRIVATE_KEY`
- `NUADA_SSH_USER`
- `NUADA_KNOWN_HOSTS`

Optional:

- `NUADA_SSH_PORT`
