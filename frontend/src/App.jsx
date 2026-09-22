import {
  Activity,
  CheckCircle2,
  KeyRound,
  Network,
  Play,
  RefreshCw,
  Server,
  ShieldCheck,
  UserPlus,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { API_BASE, api } from "./api.js";

const defaultIdentity = {
  client_id: "hospital-1",
  display_name: "Saint Rowan Hospital",
  threshold: 2,
  shares: 3,
};

const defaultSimulation = {
  rounds: 1,
  clients: 3,
  fake_data: true,
};

function StatusPill({ state }) {
  const normalized = state || "unknown";
  return <span className={`status-pill status-${normalized}`}>{normalized}</span>;
}

function Field({ label, value, onChange, type = "text", min, max }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type={type} min={min} max={max} value={value} onChange={onChange} />
    </label>
  );
}

function App() {
  const [status, setStatus] = useState(null);
  const [identityForm, setIdentityForm] = useState(defaultIdentity);
  const [simulationForm, setSimulationForm] = useState(defaultSimulation);
  const [identity, setIdentity] = useState(null);
  const [transcript, setTranscript] = useState(null);
  const [verifyResult, setVerifyResult] = useState(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const registeredCount = status?.registered_clients?.length || 0;
  const thresholdLabel = useMemo(() => {
    if (!identity) return "none";
    return `${identity.threshold} of ${identity.shares.length}`;
  }, [identity]);

  async function refreshStatus() {
    setError("");
    try {
      const payload = await api.status();
      setStatus(payload);
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    refreshStatus();
    const timer = window.setInterval(refreshStatus, 5000);
    return () => window.clearInterval(timer);
  }, []);

  async function runAction(name, action) {
    setBusy(name);
    setError("");
    try {
      await action();
      await refreshStatus();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  function updateIdentity(field, value) {
    setIdentityForm((current) => ({ ...current, [field]: value }));
  }

  function updateSimulation(field, value) {
    setSimulationForm((current) => ({ ...current, [field]: value }));
  }

  async function createIdentity() {
    const payload = {
      ...identityForm,
      threshold: Number(identityForm.threshold),
      shares: Number(identityForm.shares),
    };
    const created = await api.createIdentity(payload);
    setIdentity(created);
    setTranscript(null);
    setVerifyResult(null);
  }

  async function issueChallenge() {
    const targetClient = identity?.client_id || identityForm.client_id;
    const created = await api.challenge({
      client_id: targetClient,
      round_id: `round-${status?.simulation?.rounds || 0}`,
      model_digest: "fedper-shared-bootstrap",
    });
    setTranscript(created.transcript);
    setVerifyResult(null);
  }

  async function signAndVerify() {
    if (!identity || !transcript) return;
    const signed = await api.sign({
      transcript,
      shares: identity.shares.slice(0, identity.threshold),
      threshold: identity.threshold,
    });
    const verified = await api.verify({
      transcript,
      signature_b64: signed.signature_b64,
    });
    setVerifyResult(verified);
  }

  async function startSimulation() {
    await api.startSimulation({
      rounds: Number(simulationForm.rounds),
      clients: Number(simulationForm.clients),
      fake_data: Boolean(simulationForm.fake_data),
    });
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">RodentNet</p>
          <h1>Privacy FL Operations Console</h1>
        </div>
        <button
          className="icon-button"
          title="Refresh status"
          onClick={() => runAction("refresh", refreshStatus)}
          disabled={busy === "refresh"}
        >
          <RefreshCw size={18} />
          <span>Refresh</span>
        </button>
      </header>

      {error && (
        <section className="alert">
          <XCircle size={18} />
          <span>{error}</span>
        </section>
      )}

      <section className="metric-strip">
        <div className="metric">
          <Server size={20} />
          <span>API</span>
          <strong>{API_BASE}</strong>
        </div>
        <div className="metric">
          <ShieldCheck size={20} />
          <span>PQC Provider</span>
          <strong>{status?.pqc_provider || "loading"}</strong>
        </div>
        <div className="metric">
          <Network size={20} />
          <span>Clients</span>
          <strong>{registeredCount}</strong>
        </div>
        <div className="metric">
          <Activity size={20} />
          <span>Simulation</span>
          <StatusPill state={status?.simulation?.state} />
        </div>
      </section>

      <section className="workspace-grid">
        <article className="panel">
          <div className="panel-heading">
            <UserPlus size={20} />
            <h2>Client Identity</h2>
          </div>
          <div className="form-grid">
            <Field
              label="Client ID"
              value={identityForm.client_id}
              onChange={(event) => updateIdentity("client_id", event.target.value)}
            />
            <Field
              label="Display Name"
              value={identityForm.display_name}
              onChange={(event) => updateIdentity("display_name", event.target.value)}
            />
            <Field
              label="Threshold"
              type="number"
              min="2"
              max="10"
              value={identityForm.threshold}
              onChange={(event) => updateIdentity("threshold", event.target.value)}
            />
            <Field
              label="Shares"
              type="number"
              min="2"
              max="12"
              value={identityForm.shares}
              onChange={(event) => updateIdentity("shares", event.target.value)}
            />
          </div>
          <button
            className="primary-action"
            onClick={() => runAction("identity", createIdentity)}
            disabled={busy === "identity"}
          >
            <KeyRound size={18} />
            <span>Generate Threshold Identity</span>
          </button>
          <dl className="detail-list">
            <div>
              <dt>Threshold</dt>
              <dd>{thresholdLabel}</dd>
            </div>
            <div>
              <dt>Public Key</dt>
              <dd className="truncate">{identity?.public_key_b64 || "none"}</dd>
            </div>
          </dl>
        </article>

        <article className="panel">
          <div className="panel-heading">
            <ShieldCheck size={20} />
            <h2>Handshake</h2>
          </div>
          <div className="button-row">
            <button onClick={() => runAction("challenge", issueChallenge)} disabled={busy === "challenge"}>
              <RefreshCw size={18} />
              <span>Issue Challenge</span>
            </button>
            <button
              onClick={() => runAction("verify", signAndVerify)}
              disabled={busy === "verify" || !identity || !transcript}
            >
              <CheckCircle2 size={18} />
              <span>Sign And Verify</span>
            </button>
          </div>
          <dl className="detail-list">
            <div>
              <dt>Nonce</dt>
              <dd className="truncate">{transcript?.nonce || "none"}</dd>
            </div>
            <div>
              <dt>Round</dt>
              <dd>{transcript?.round_id || "none"}</dd>
            </div>
            <div>
              <dt>Decision</dt>
              <dd>
                {verifyResult ? (
                  <span className={verifyResult.accepted ? "decision-ok" : "decision-bad"}>
                    {verifyResult.accepted ? "accepted" : "rejected"}
                  </span>
                ) : (
                  "pending"
                )}
              </dd>
            </div>
          </dl>
        </article>

        <article className="panel panel-wide">
          <div className="panel-heading">
            <Network size={20} />
            <h2>Secure Aggregation Simulation</h2>
          </div>
          <div className="simulation-layout">
            <div className="form-grid compact">
              <Field
                label="Rounds"
                type="number"
                min="1"
                max="50"
                value={simulationForm.rounds}
                onChange={(event) => updateSimulation("rounds", event.target.value)}
              />
              <Field
                label="Clients"
                type="number"
                min="2"
                max="20"
                value={simulationForm.clients}
                onChange={(event) => updateSimulation("clients", event.target.value)}
              />
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={simulationForm.fake_data}
                  onChange={(event) => updateSimulation("fake_data", event.target.checked)}
                />
                <span>FakeData</span>
              </label>
            </div>
            <button
              className="primary-action"
              onClick={() => runAction("simulation", startSimulation)}
              disabled={busy === "simulation" || status?.simulation?.state === "running"}
            >
              <Play size={18} />
              <span>Start Simulation</span>
            </button>
          </div>
          <div className="run-state">
            <StatusPill state={status?.simulation?.state} />
            <span>{status?.simulation?.message || "loading"}</span>
          </div>
        </article>
      </section>

      <section className="audit-section">
        <div className="section-heading">
          <Activity size={20} />
          <h2>Audit Events</h2>
        </div>
        <div className="audit-table">
          {(status?.audit || []).map((event) => (
            <div className="audit-row" key={`${event.created_at}-${event.event}`}>
              <span>{event.event}</span>
              <strong>{event.detail}</strong>
              <time>{new Date(event.created_at * 1000).toLocaleTimeString()}</time>
            </div>
          ))}
          {status?.audit?.length === 0 && <div className="empty-row">No events yet.</div>}
        </div>
      </section>
    </main>
  );
}

export default App;
