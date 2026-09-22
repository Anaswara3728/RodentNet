# RodentNet

Privacy-preserving federated learning for CIFAR10 using Flower, FedPer, MPC secure
aggregation, and threshold-controlled Dilithium / ML-DSA-65 identity verification.

RodentNet is organized as a monorepo:

```text
backend/   FastAPI control plane, Flower strategy/client code, crypto and MPC primitives
frontend/  React + Vite operations console
scripts/   Local runner helpers
```

## What Is Implemented

- **FedPer for CIFAR10**: shared convolutional representation and projection layers are
  federated, while each hospital keeps its `personal_head` local.
- **Custom Flower strategy**: `MPCFedPerStrategy` aggregates sample-weighted, masked shared
  tensors and emits audit metrics for accepted/rejected clients.
- **MPC secure aggregation**: finite-field fixed-point aggregation with pairwise masks that
  cancel only at the cohort sum. Shamir secret sharing is included for threshold mask/key
  recovery workflows.
- **Threshold Dilithium identity**: hospital signing keys are split with Shamir sharing.
  A threshold subset reconstructs the signing key for a Flower handshake transcript, and
  the server verifies the transcript before admitting updates.
- **PQC provider adapter**: the backend uses `pqcrypto.sign.ml_dsa_65` when available,
  falls back to `pqcrypto.sign.dilithium3` for older package names, and includes a
  deterministic development provider for machines without PQC bindings.
- **FastAPI middleware**: enroll clients, issue and verify handshake challenges, expose
  status/audit events, and launch local Flower simulations.
- **React console**: client identity generation, threshold challenge signing, verification,
  simulation start, status, and audit event views.

The development fallback is intentionally not a production signature scheme. Install the
`pqc` extra and set `RODENTNET_ALLOW_DEV_PQC_FALLBACK=false` for deployments that must fail
closed when ML-DSA-65 bindings are unavailable.

## Backend

```bash
cd /Users/gokulalex/Code/EPIC/RodentNet
python -m venv backend/.venv
source backend/.venv/bin/activate
pip install -e "backend[dev,pqc]"
python -m pytest backend/tests
python -m uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/status
```

## Frontend

```bash
cd /Users/gokulalex/Code/EPIC/RodentNet
npm install --prefix frontend
VITE_API_BASE=http://localhost:8000 npm --prefix frontend run dev
```

Open `http://localhost:5173`.

## Local Handshake Flow

```bash
curl -s http://localhost:8000/api/status

curl -s -X POST http://localhost:8000/api/dev/identity \
  -H 'content-type: application/json' \
  -d '{"client_id":"hospital-1","display_name":"Hospital One","threshold":2,"shares":3}'
```

Use the returned key shares with `/api/handshake/challenge`, `/api/dev/sign`, and
`/api/handshake/verify`, or use the React console to run the same flow.

## Simulation

The simulation endpoint starts a Flower simulation in the background. Use `fake_data=true`
for fast local smoke tests without downloading CIFAR10.

```bash
curl -s -X POST http://localhost:8000/api/simulation/start \
  -H 'content-type: application/json' \
  -d '{"rounds":1,"clients":3,"fake_data":true}'
```

For real CIFAR10, set `fake_data=false`; torchvision will download the dataset under
`backend/data/cifar10`.

## Key Files

- `backend/app/crypto/shamir.py` - Shamir secret and tensor sharing utilities.
- `backend/app/crypto/mpc.py` - pairwise masked secure aggregation protocol.
- `backend/app/crypto/threshold_dilithium.py` - ML-DSA-65 provider adapter and threshold
  handshake identity registry.
- `backend/app/fl/strategy.py` - custom Flower `MPCFedPerStrategy`.
- `backend/app/fl/client.py` - FedPer Flower NumPy client with MPC masking.
- `backend/app/api/routes.py` - FastAPI enrollment, handshake, status, and simulation APIs.
- `frontend/src/App.jsx` - React operations console.
