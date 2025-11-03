(() => {
  const STORAGE_KEY = "scrizaAuth";

  function parseJwt(token) {
    try {
      const base64Url = token.split(".")[1];
      const base64 = base64Url.replace(/-/g, "+").replace(/_/g, "/");
      const jsonPayload = decodeURIComponent(
        atob(base64)
          .split("")
          .map((c) => "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2))
          .join("")
      );
      return JSON.parse(jsonPayload);
    } catch (err) {
      console.error("Failed to parse JWT", err);
      return null;
    }
  }

  function setAuth(token) {
    const payload = parseJwt(token);
    if (!payload) throw new Error("Invalid token");
    const data = {
      token,
      payload,
      storedAt: Date.now(),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    return data;
  }

  function getAuth() {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch {
      localStorage.removeItem(STORAGE_KEY);
      return null;
    }
  }

  function clearAuth() {
    localStorage.removeItem(STORAGE_KEY);
  }

  async function authFetch(url, options = {}) {
    const auth = getAuth();
    if (!auth || !auth.token) {
      // Soft-redirect for better UX
      try { clearAuth(); } catch {}
      if (!/\/login$/.test(window.location.pathname)) {
        window.location.href = "/login";
      }
      throw new Error("Not authenticated");
    }
    const headers = new Headers(options.headers || {});
    headers.set("Authorization", `Bearer ${auth.token}`);
    if (options.body && !(options.body instanceof FormData)) {
      headers.set("Content-Type", "application/json");
    }
    const response = await fetch(url, { ...options, headers });
    if (!response.ok) {
      // Auto-handle auth failures
      if (response.status === 401 || response.status === 403) {
        try { clearAuth(); } catch {}
        const dest = (response.status === 403) ? "/admin/login" : "/login";
        if (!dest.endsWith(window.location.pathname)) {
          window.location.href = dest;
        }
      }
      let detail = await response.text();
      try {
        const parsed = JSON.parse(detail);
        detail = parsed.detail || parsed.message || JSON.stringify(parsed);
      } catch {
        // keep detail as text
      }
      throw new Error(detail || `Request failed: ${response.status}`);
    }
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      return response.json();
    }
    return response.text();
  }

  function requireRole(required) {
    const auth = getAuth();
    if (!auth) return false;
    const role = auth?.payload?.role;
    if (!required) return true;
    if (Array.isArray(required)) {
      return required.includes(role);
    }
    return role === required;
  }

  window.scrizaAuth = {
    setAuth,
    getAuth,
    clearAuth,
    authFetch,
    requireRole,
  };
})();
