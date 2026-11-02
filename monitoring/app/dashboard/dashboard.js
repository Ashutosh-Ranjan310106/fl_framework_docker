// Flower Federated Learning Dashboard JS

let ws = null;
let reconnectTimer = null;
let pingInterval = null;
let globalMetricsChart = null;
let clientLossChart = null;

let allEvents = [];
let currentExperiment = null;

// Color palette
const COLORS = {
    blue: "#38bdf8",
    indigo: "#818cf8",
    green: "#34d399",
    orange: "#fb923c",
    red: "#f87171",
    purple: "#c084fc",
    muted: "#64748b",
};

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", () => {
    initCharts();
    setupEventListeners();
    fetchInitialState();
    connectWebSocket();
});

// 1. CHART INITIALIZATION
function initCharts() {
    // Global Metrics Chart
    const ctxGlobal = document.getElementById("globalMetricsChart").getContext("2d");
    globalMetricsChart = new Chart(ctxGlobal, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Global Accuracy (%)",
                    data: [],
                    borderColor: COLORS.green,
                    backgroundColor: "rgba(52, 211, 153, 0.1)",
                    borderWidth: 2,
                    yAxisID: "y-accuracy",
                    tension: 0.2,
                    fill: true,
                },
                {
                    label: "Global Loss",
                    data: [],
                    borderColor: COLORS.orange,
                    backgroundColor: "transparent",
                    borderWidth: 2,
                    borderDash: [4, 4],
                    yAxisID: "y-loss",
                    tension: 0.2,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: "#27354a" },
                    ticks: { color: "#94a3b8" },
                    title: { display: true, text: "FL Round", color: "#64748b" },
                },
                "y-accuracy": {
                    type: "linear",
                    position: "left",
                    min: 0,
                    max: 100,
                    grid: { color: "#27354a" },
                    ticks: { color: COLORS.green, callback: (v) => `${v}%` },
                    title: { display: true, text: "Accuracy (%)", color: COLORS.green },
                },
                "y-loss": {
                    type: "linear",
                    position: "right",
                    grid: { drawOnChartArea: false },
                    ticks: { color: COLORS.orange },
                    title: { display: true, text: "Loss", color: COLORS.orange },
                },
            },
            plugins: {
                legend: { labels: { color: "#f1f5f9" } },
            },
        },
    });

    // Client Loss Chart
    const ctxClient = document.getElementById("clientLossChart").getContext("2d");
    clientLossChart = new Chart(ctxClient, {
        type: "line",
        data: {
            labels: [],
            datasets: [],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: "#27354a" },
                    ticks: { color: "#94a3b8" },
                    title: { display: true, text: "FL Round", color: "#64748b" },
                },
                y: {
                    grid: { color: "#27354a" },
                    ticks: { color: "#94a3b8" },
                    title: { display: true, text: "Training Loss", color: "#64748b" },
                },
            },
            plugins: {
                legend: { labels: { color: "#f1f5f9" } },
            },
        },
    });
}

// 2. WEBSOCKET CONNECTION
function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;

    updateWsIndicator("connecting");

    try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            updateWsIndicator("connected");
            clearInterval(reconnectTimer);
            reconnectTimer = null;

            // Heartbeat
            pingInterval = setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send("ping");
                }
            }, 10000);
        };

        ws.onmessage = (event) => {
            if (event.data === "pong") return;
            try {
                const message = JSON.parse(event.data);
                handleWebSocketMessage(message);
            } catch (err) {
                console.error("Error parsing WebSocket message:", err);
            }
        };

        ws.onclose = () => {
            updateWsIndicator("disconnected");
            clearInterval(pingInterval);
            if (!reconnectTimer) {
                reconnectTimer = setTimeout(connectWebSocket, 3000);
            }
        };

        ws.onerror = (err) => {
            console.warn("WebSocket encountered error:", err);
            ws.close();
        };
    } catch (e) {
        updateWsIndicator("disconnected");
        reconnectTimer = setTimeout(connectWebSocket, 3000);
    }
}

function updateWsIndicator(status) {
    const dot = document.getElementById("ws-status-dot");
    const text = document.getElementById("ws-status-text");

    if (status === "connected") {
        dot.className = "dot connected";
        text.innerText = "Live Telemetry Connected";
    } else {
        dot.className = "dot disconnected";
        text.innerText = status === "connecting" ? "Connecting..." : "Disconnected (Reconnecting...)";
    }
}

// 3. WEBSOCKET MESSAGE DISPATCHER
function handleWebSocketMessage(msg) {
    if (msg.type === "event") {
        const ev = msg.event;
        allEvents.unshift(ev);
        renderEvent(ev);
        handleLifecycleEvent(ev);
    } else if (msg.type === "system_resources") {
        updateSystemResourceBars(msg.data);
    }
}

function handleLifecycleEvent(ev) {
    const type = ev.event_type;
    const data = ev.data || {};

    if (type === "experiment_started") {
        setExperimentStatus("RUNNING");
        if (data.num_rounds) {
            updateRoundProgress(0, data.num_rounds);
        }
    } else if (type === "experiment_completed") {
        setExperimentStatus("COMPLETED");
    } else if (type === "experiment_failed") {
        setExperimentStatus("FAILED");
    } else if (type === "round_started") {
        updateRoundProgress(ev.round, data.total_rounds || 3);
        fetchRounds();
    } else if (type === "model_distribution_started" || type === "client_updates_started" || type === "aggregation_started" || type === "aggregation_finished" || type === "evaluation_finished" || type === "round_completed") {
        fetchRounds();
        fetchClients();
        if (type === "evaluation_finished") {
            appendGlobalMetric(ev.round, data.accuracy, data.loss);
        }
    } else if (type.startsWith("client_")) {
        fetchClients();
        if (type === "client_training_completed" && ev.client_id) {
            appendClientMetric(ev.client_id, ev.round, data.loss);
        }
    }
}

function updateSystemResourceBars(data) {
    const cpuVal = Math.round(data.host_cpu_percent || 0);
    const ramVal = Math.round(data.host_ram_percent || 0);

    const cpuBar = document.getElementById("host-cpu-bar");
    const ramBar = document.getElementById("host-ram-bar");
    const cpuText = document.getElementById("host-cpu-val");
    const ramText = document.getElementById("host-ram-val");

    if (cpuBar) cpuBar.style.width = `${Math.min(cpuVal, 100)}%`;
    if (ramBar) ramBar.style.width = `${Math.min(ramVal, 100)}%`;
    if (cpuText) cpuText.innerText = `${cpuVal}%`;
    if (ramText) ramText.innerText = `${ramVal}% (${Math.round(data.host_ram_used_mb || 0)} MB)`;
}

// 4. REST DATA FETCHING
async function fetchInitialState() {
    try {
        const [expRes, roundsRes, clientsRes, metricsRes, eventsRes] = await Promise.all([
            fetch("/api/experiment/current"),
            fetch("/api/rounds"),
            fetch("/api/clients"),
            fetch("/api/metrics/global"),
            fetch("/api/events?limit=100"),
        ]);

        if (expRes.ok) {
            currentExperiment = await expRes.json();
            renderExperiment(currentExperiment);
        }
        if (roundsRes.ok) {
            const rounds = await roundsRes.json();
            renderRounds(rounds);
        }
        if (clientsRes.ok) {
            const clients = await clientsRes.json();
            renderClientsTable(clients);
        }
        if (metricsRes.ok) {
            const metrics = await metricsRes.json();
            populateGlobalMetrics(metrics);
        }
        if (eventsRes.ok) {
            allEvents = await eventsRes.json();
            renderEventFeed(allEvents);
        }
    } catch (err) {
        console.error("Error fetching initial state:", err);
    }
}

async function fetchRounds() {
    try {
        const res = await fetch("/api/rounds");
        if (res.ok) {
            const rounds = await res.json();
            renderRounds(rounds);
        }
    } catch (e) {
        console.debug("Failed to refresh rounds:", e);
    }
}

async function fetchClients() {
    try {
        const res = await fetch("/api/clients");
        if (res.ok) {
            const clients = await res.json();
            renderClientsTable(clients);
        }
    } catch (e) {
        console.debug("Failed to refresh clients:", e);
    }
}

// 5. DOM RENDERING HELPERS
function renderExperiment(exp) {
    if (!exp) return;
    document.getElementById("exp-id-display").innerText = exp.id || "cifar10_baseline_001";
    setExperimentStatus(exp.status || "IDLE");
    if (exp.strategy) {
        document.getElementById("stat-strategy-name").innerText = `Strategy: ${exp.strategy.toUpperCase()}`;
    }
}

function setExperimentStatus(status) {
    const pill = document.getElementById("exp-status-pill");
    const val = document.getElementById("stat-exp-status");
    const st = (status || "IDLE").toUpperCase();

    pill.innerText = st;
    val.innerText = st;

    pill.className = `pill ${st.toLowerCase()}`;
}

function updateRoundProgress(current, total) {
    const totalR = total || 3;
    const curR = current || 0;
    document.getElementById("stat-current-round").innerText = `${curR} / ${totalR}`;
    const pct = Math.min(Math.round((curR / totalR) * 100), 100);
    document.getElementById("round-progress-bar").style.width = `${pct}%`;
}

function renderRounds(rounds) {
    const container = document.getElementById("rounds-lifecycle-container");
    if (!rounds || rounds.length === 0) {
        container.innerHTML = '<div class="empty-state">No federated learning rounds started yet.</div>';
        return;
    }

    container.innerHTML = rounds
        .map((r) => {
            const stages = r.stages || {};
            const dur = r.duration_seconds ? `(${r.duration_seconds.toFixed(1)}s)` : "";
            return `
            <div class="round-row">
                <div class="round-header-line">
                    <span class="round-title">Round ${r.round_number} ${dur}</span>
                    <span class="status-badge ${r.status.toLowerCase().replace(" ", "-")}">${r.status}</span>
                </div>
                <div class="stages-tracker">
                    <div class="stage-step">
                        <span class="stage-name">1. Distribution</span>
                        <span class="stage-status ${formatStageClass(stages.model_distribution)}">${stages.model_distribution || "WAITING"}</span>
                    </div>
                    <div class="stage-step">
                        <span class="stage-name">2. Client Training</span>
                        <span class="stage-status ${formatStageClass(stages.client_training)}">${stages.client_training || "WAITING"}</span>
                    </div>
                    <div class="stage-step">
                        <span class="stage-name">3. Update Collection</span>
                        <span class="stage-status ${formatStageClass(stages.update_collection)}">${stages.update_collection || "WAITING"}</span>
                    </div>
                    <div class="stage-step">
                        <span class="stage-name">4. Aggregation</span>
                        <span class="stage-status ${formatStageClass(stages.aggregation)}">${stages.aggregation || "WAITING"}</span>
                    </div>
                    <div class="stage-step">
                        <span class="stage-name">5. Global Eval</span>
                        <span class="stage-status ${formatStageClass(stages.global_evaluation)}">${stages.global_evaluation || "WAITING"}</span>
                    </div>
                </div>
            </div>`;
        })
        .join("");
}

function formatStageClass(status) {
    if (!status) return "waiting";
    const s = status.toLowerCase().replace(" ", "-");
    return s;
}

function renderClientsTable(clients) {
    const tbody = document.getElementById("clients-table-body");
    if (!clients || clients.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" class="empty-state">No clients registered yet.</td></tr>';
        return;
    }

    let activeCount = 0;
    let completedCount = 0;

    tbody.innerHTML = clients
        .map((c) => {
            const st = (c.status || "Registered").toLowerCase().replace(" ", "-");
            if (st === "training" || st === "sending-update") activeCount++;
            if (st === "waiting" || st === "completed") completedCount++;

            const lossStr = c.loss !== null && c.loss !== undefined ? c.loss.toFixed(4) : "--";
            const accStr = c.accuracy !== null && c.accuracy !== undefined ? `${(c.accuracy * 100).toFixed(1)}%` : "--";
            const cpuStr = c.cpu !== null && c.cpu !== undefined ? `${Math.round(c.cpu)}%` : "--";
            const ramStr = c.ram !== null && c.ram !== undefined ? `${Math.round(c.ram)} MB` : "--";

            return `
            <tr>
                <td><strong>${c.client_id}</strong></td>
                <td><span class="role-badge">${c.role || "honest"}</span></td>
                <td><span class="status-badge ${st}">${c.status}</span></td>
                <td>${c.current_round || 0}</td>
                <td>${c.samples ? c.samples.toLocaleString() : "--"}</td>
                <td>${lossStr}</td>
                <td>${accStr}</td>
                <td>${cpuStr}</td>
                <td>${ramStr}</td>
                <td><button class="btn-sm" onclick="showClientModal('${c.client_id}')">View</button></td>
            </tr>`;
        })
        .join("");

    document.getElementById("stat-clients-count").innerText = `${clients.length} Configured`;
    document.getElementById("stat-active-clients").innerText = activeCount;
    document.getElementById("stat-completed-clients").innerText = completedCount;
}

// 6. EVENT FEED
function renderEvent(ev) {
    const container = document.getElementById("event-feed-container");
    if (container.querySelector(".empty-state")) {
        container.innerHTML = "";
    }

    if (!matchesFilter(ev)) return;

    const div = document.createElement("div");
    div.className = `event-item ${(ev.severity || "info").toLowerCase()}`;
    div.innerHTML = `
        <span class="event-time">${formatTime(ev.timestamp)}</span>
        <span class="event-badge ${(ev.severity || "info").toLowerCase()}">${ev.severity || "INFO"}</span>
        <span class="event-msg">${ev.message || ev.event_type}</span>
    `;

    container.insertBefore(div, container.firstChild);
}

function renderEventFeed(events) {
    const container = document.getElementById("event-feed-container");
    const filtered = events.filter(matchesFilter);

    if (filtered.length === 0) {
        container.innerHTML = '<div class="empty-state">No matching events.</div>';
        return;
    }

    container.innerHTML = filtered
        .map(
            (ev) => `
        <div class="event-item ${(ev.severity || "info").toLowerCase()}">
            <span class="event-time">${formatTime(ev.timestamp)}</span>
            <span class="event-badge ${(ev.severity || "info").toLowerCase()}">${ev.severity || "INFO"}</span>
            <span class="event-msg">${ev.message || ev.event_type}</span>
        </div>`
        )
        .join("");
}

function matchesFilter(ev) {
    const sev = document.getElementById("filter-severity").value;
    const cid = document.getElementById("filter-client").value;

    if (sev && ev.severity !== sev) return false;
    if (cid && ev.client_id !== cid) return false;
    return true;
}

function formatTime(isoStr) {
    if (!isoStr) return "";
    const d = new Date(isoStr);
    return d.toTimeString().split(" ")[0];
}

// 7. CHART UPDATES
function populateGlobalMetrics(metrics) {
    if (!metrics || metrics.length === 0) return;

    const labels = [];
    const accData = [];
    const lossData = [];

    metrics.forEach((m) => {
        labels.push(`R${m.round}`);
        accData.push(Number((m.accuracy * 100).toFixed(2)));
        lossData.push(Number(m.loss.toFixed(4)));
    });

    globalMetricsChart.data.labels = labels;
    globalMetricsChart.data.datasets[0].data = accData;
    globalMetricsChart.data.datasets[1].data = lossData;
    globalMetricsChart.update();

    const latest = metrics[metrics.length - 1];
    document.getElementById("stat-global-accuracy").innerText = `${(latest.accuracy * 100).toFixed(1)}%`;
    document.getElementById("stat-global-loss").innerText = latest.loss.toFixed(4);
}

function appendGlobalMetric(roundNum, accuracy, loss) {
    const label = `R${roundNum}`;
    const accPct = Number((accuracy * 100).toFixed(2));
    const lossVal = Number(loss.toFixed(4));

    const idx = globalMetricsChart.data.labels.indexOf(label);
    if (idx >= 0) {
        globalMetricsChart.data.datasets[0].data[idx] = accPct;
        globalMetricsChart.data.datasets[1].data[idx] = lossVal;
    } else {
        globalMetricsChart.data.labels.push(label);
        globalMetricsChart.data.datasets[0].data.push(accPct);
        globalMetricsChart.data.datasets[1].data.push(lossVal);
    }
    globalMetricsChart.update();

    document.getElementById("stat-global-accuracy").innerText = `${accPct.toFixed(1)}%`;
    document.getElementById("stat-global-loss").innerText = lossVal.toFixed(4);
}

function appendClientMetric(clientId, roundNum, loss) {
    const label = `R${roundNum}`;
    if (!clientLossChart.data.labels.includes(label)) {
        clientLossChart.data.labels.push(label);
    }

    let dataset = clientLossChart.data.datasets.find((d) => d.label === clientId);
    if (!dataset) {
        const clientColors = [COLORS.blue, COLORS.purple, COLORS.orange, COLORS.green];
        const color = clientColors[clientLossChart.data.datasets.length % clientColors.length];
        dataset = {
            label: clientId,
            data: [],
            borderColor: color,
            tension: 0.2,
            fill: false,
        };
        clientLossChart.data.datasets.push(dataset);
    }

    const roundIndex = clientLossChart.data.labels.indexOf(label);
    dataset.data[roundIndex] = Number(loss.toFixed(4));
    clientLossChart.update();
}

// 8. CLIENT MODAL
async function showClientModal(clientId) {
    const modal = document.getElementById("client-modal");
    const title = document.getElementById("modal-client-title");
    const body = document.getElementById("modal-client-body");

    title.innerText = `Client Details: ${clientId}`;
    body.innerHTML = '<div class="empty-state">Loading client telemetry...</div>';
    modal.classList.remove("hidden");

    try {
        const res = await fetch(`/api/clients/${clientId}`);
        if (!res.ok) throw new Error("Failed to load client details");
        const data = await res.json();

        body.innerHTML = `
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
                <div><strong>Role:</strong> ${data.role || "honest"}</div>
                <div><strong>Status:</strong> <span class="status-badge ${data.status.toLowerCase().replace(" ", "-")}">${data.status}</span></div>
                <div><strong>Current Round:</strong> ${data.current_round}</div>
                <div><strong>Partition Samples:</strong> ${data.samples ? data.samples.toLocaleString() : "--"}</div>
                <div><strong>Latest Loss:</strong> ${data.loss !== null ? data.loss.toFixed(4) : "--"}</div>
                <div><strong>Latest Accuracy:</strong> ${data.accuracy !== null ? (data.accuracy * 100).toFixed(1) + "%" : "--"}</div>
                <div><strong>CPU Usage:</strong> ${data.cpu !== null ? data.cpu + "%" : "--"}</div>
                <div><strong>RAM Footprint:</strong> ${data.ram !== null ? data.ram.toFixed(1) + " MB" : "--"}</div>
            </div>
            <h4>Recent Training Metrics History</h4>
            <div class="table-responsive" style="margin-top: 8px;">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Round</th>
                            <th>Loss</th>
                            <th>Accuracy</th>
                            <th>Duration</th>
                            <th>CPU</th>
                            <th>RAM</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${
                            data.metrics_history && data.metrics_history.length > 0
                                ? data.metrics_history
                                      .map(
                                          (m) => `
                            <tr>
                                <td>Round ${m.round}</td>
                                <td>${m.loss.toFixed(4)}</td>
                                <td>${(m.accuracy * 100).toFixed(1)}%</td>
                                <td>${m.duration_seconds.toFixed(2)}s</td>
                                <td>${m.cpu ? m.cpu.toFixed(1) + "%" : "--"}</td>
                                <td>${m.ram ? m.ram.toFixed(1) + " MB" : "--"}</td>
                            </tr>`
                                      )
                                      .join("")
                                : '<tr><td colspan="6" class="empty-state">No training metrics recorded yet.</td></tr>'
                        }
                    </tbody>
                </table>
            </div>
        `;
    } catch (e) {
        body.innerHTML = `<div class="empty-state" style="color: var(--accent-red);">Error: ${e.message}</div>`;
    }
}

// 9. EVENT LISTENERS
function setupEventListeners() {
    document.getElementById("filter-severity").addEventListener("change", () => renderEventFeed(allEvents));
    document.getElementById("filter-client").addEventListener("change", () => renderEventFeed(allEvents));
    document.getElementById("clear-events-btn").addEventListener("click", () => {
        document.getElementById("filter-severity").value = "";
        document.getElementById("filter-client").value = "";
        renderEventFeed(allEvents);
    });

    document.getElementById("modal-close-btn").addEventListener("click", () => {
        document.getElementById("client-modal").classList.add("hidden");
    });

    document.getElementById("client-modal").addEventListener("click", (e) => {
        if (e.target.id === "client-modal") {
            document.getElementById("client-modal").classList.add("hidden");
        }
    });
}
