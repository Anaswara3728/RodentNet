const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      // Keep the HTTP status text.
    }
    throw new Error(detail);
  }
  return response.json();
}

export const api = {
  status: () => request("/api/status"),
  createIdentity: (payload) =>
    request("/api/dev/identity", { method: "POST", body: JSON.stringify(payload) }),
  challenge: (payload) =>
    request("/api/handshake/challenge", { method: "POST", body: JSON.stringify(payload) }),
  sign: (payload) => request("/api/dev/sign", { method: "POST", body: JSON.stringify(payload) }),
  verify: (payload) =>
    request("/api/handshake/verify", { method: "POST", body: JSON.stringify(payload) }),
  startSimulation: (payload) =>
    request("/api/simulation/start", { method: "POST", body: JSON.stringify(payload) }),
};

export { API_BASE };
