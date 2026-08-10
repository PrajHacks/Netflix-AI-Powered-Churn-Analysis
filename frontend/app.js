const API_BASE_URL = "";
const TABLE_PAGE_SIZE = 25;
const CUSTOMER_FETCH_PAGE_SIZE = 100;

const state = {
  page: 1,
  pageSize: TABLE_PAGE_SIZE,
  summary: null,
  segments: [],
  allCustomers: [],
  filteredCustomers: [],
  activeSegment: null,
  charts: {},
};

const chartPalette = {
  low: "#36d39b",
  medium: "#ffbf3d",
  high: "#ff6b7f",
  neutral: "#7bb0ff",
};

const app = {};

document.addEventListener("DOMContentLoaded", initializeApp);

async function initializeApp() {
  cacheElements();
  bindEvents();
  syncSliderOutputs();

  const backendOnline = await pingBackend();
  if (backendOnline) {
    await loadDashboard();
  }
}

function cacheElements() {
  app.apiStatus = document.getElementById("apiStatus");
  app.notice = document.getElementById("appNotice");
  app.kpiTotalCustomers = document.getElementById("kpiTotalCustomers");
  app.kpiChurnRate = document.getElementById("kpiChurnRate");
  app.kpiChurnRateSubtext = document.getElementById("kpiChurnRateSubtext");
  app.kpiAtRiskCustomers = document.getElementById("kpiAtRiskCustomers");
  app.kpiAtRiskCustomersSubtext = document.getElementById("kpiAtRiskCustomersSubtext");
  app.kpiAtRiskRevenue = document.getElementById("kpiAtRiskRevenue");
  app.segmentCards = document.getElementById("segmentCards");
  app.segmentFilterLabel = document.getElementById("segmentFilterLabel");
  app.clearSegmentFilterBtn = document.getElementById("clearSegmentFilterBtn");
  app.customerTableBody = document.getElementById("customerTableBody");
  app.pageIndicator = document.getElementById("pageIndicator");
  app.prevPageBtn = document.getElementById("prevPageBtn");
  app.nextPageBtn = document.getElementById("nextPageBtn");
  app.predictForm = document.getElementById("predictForm");
  app.predictResultPanel = document.getElementById("predictResultPanel");
  app.predictRiskBadge = document.getElementById("predictRiskBadge");
  app.predictProbability = document.getElementById("predictProbability");
  app.predictBandText = document.getElementById("predictBandText");
  app.predictProgress = document.getElementById("predictProgress");
  app.predictReasonsList = document.getElementById("predictReasonsList");
  app.customerModalOverlay = document.getElementById("customerModalOverlay");
  app.customerModalContent = document.getElementById("customerModalContent");
  app.closeCustomerModalBtn = document.getElementById("closeCustomerModalBtn");
  app.satisfactionSlider = document.getElementById("customer_satisfaction_score");
  app.satisfactionValue = document.getElementById("customerSatisfactionValue");
  app.engagementSlider = document.getElementById("engagement_rate");
  app.engagementValue = document.getElementById("engagementRateValue");
}

function bindEvents() {
  app.prevPageBtn.addEventListener("click", () => changePage(-1));
  app.nextPageBtn.addEventListener("click", () => changePage(1));
  app.segmentCards.addEventListener("click", onSegmentCardClick);
  app.clearSegmentFilterBtn.addEventListener("click", () => setActiveSegment(null));
  app.customerTableBody.addEventListener("click", onCustomerRowClick);
  app.customerTableBody.addEventListener("keydown", onCustomerRowKeydown);
  app.predictForm.addEventListener("submit", handlePredictSubmit);
  app.predictForm.addEventListener("reset", () => {
    window.setTimeout(() => {
      syncSliderOutputs();
      clearPredictResult();
    }, 0);
  });
  app.customerModalOverlay.addEventListener("click", onModalOverlayClick);
  app.closeCustomerModalBtn.addEventListener("click", closeCustomerModal);
  document.addEventListener("keydown", onGlobalKeydown);
  app.satisfactionSlider.addEventListener("input", syncSliderOutputs);
  app.engagementSlider.addEventListener("input", syncSliderOutputs);
}

function syncSliderOutputs() {
  app.satisfactionValue.textContent = app.satisfactionSlider.value;
  app.engagementValue.textContent = app.engagementSlider.value;
}

async function pingBackend() {
  try {
    await fetchJSON("/health");
    setApiStatus("Backend online", "success");
    return true;
  } catch (error) {
    setApiStatus("Backend offline", "danger");
    showNotice(
      "The backend is not reachable right now. Start the FastAPI service and refresh the page.",
      "error"
    );
    return false;
  }
}

async function loadDashboard() {
  try {
    const [summary, segmentsPayload, customers] = await Promise.all([
      fetchJSON("/summary"),
      fetchJSON("/segments"),
      fetchAllCustomers(),
    ]);

    state.summary = summary;
    state.segments = Array.isArray(segmentsPayload.segments) ? segmentsPayload.segments : [];
    state.allCustomers = Array.isArray(customers) ? customers.slice() : [];
    state.filteredCustomers = state.allCustomers.slice();
    state.activeSegment = null;
    state.page = 1;
    renderSummary(summary);
    renderCharts(summary);
    renderSegments(state.segments);
    renderCustomers();
    hideNotice();
  } catch (error) {
    showNotice(error.message || "Could not load dashboard data from the backend.", "error");
  }
}

async function changePage(delta) {
  const totalPages = Math.max(1, Math.ceil(getVisibleCustomers().length / state.pageSize));
  const nextPage = state.page + delta;
  if (nextPage < 1 || nextPage > totalPages) {
    return;
  }

  state.page = nextPage;
  renderCustomers();
}

async function fetchCustomers(page) {
  const query = new URLSearchParams({
    page: String(page),
    page_size: String(CUSTOMER_FETCH_PAGE_SIZE),
    sort_by: "churn_probability",
    order: "desc",
  });

  return fetchJSON(`/customers?${query.toString()}`);
}

async function fetchAllCustomers() {
  const firstPage = await fetchCustomers(1);
  const totalPages = Number(firstPage.total_pages || 1);
  const pages = [firstPage];

  if (totalPages > 1) {
    const remaining = await Promise.all(
      Array.from({ length: totalPages - 1 }, (_, index) => fetchCustomers(index + 2))
    );
    pages.push(...remaining);
  }

  return pages
    .flatMap((page) => page.items || [])
    .sort((a, b) => Number(b.churn_probability) - Number(a.churn_probability));
}

async function fetchJSON(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      Accept: "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const contentType = response.headers.get("content-type") || "";
  let payload = null;

  if (contentType.includes("application/json")) {
    payload = await response.json();
  } else {
    payload = await response.text();
  }

  if (!response.ok) {
    const detail = payload && typeof payload === "object" ? payload.detail : payload;
    throw new Error(detail || `Request failed with status ${response.status}.`);
  }

  return payload;
}

function setApiStatus(text, variant) {
  app.apiStatus.textContent = text;
  app.apiStatus.className = `status-pill status-pill--${variant}`;
}

function showNotice(message) {
  app.notice.textContent = message;
  app.notice.classList.remove("notice--hidden");
}

function hideNotice() {
  app.notice.classList.add("notice--hidden");
  app.notice.textContent = "";
}

function formatCurrency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

function formatPercent(value, digits = 1) {
  return `${Number(value).toFixed(digits)}%`;
}

function getRiskBand(probability) {
  const value = Number(probability);
  if (value < 0.3) {
    return "low";
  }
  if (value <= 0.6) {
    return "medium";
  }
  return "high";
}

function getRiskLabel(probability) {
  const band = getRiskBand(probability);
  if (band === "low") return "Low risk";
  if (band === "medium") return "Moderate risk";
  return "High risk";
}

function applyRiskState(element, probability) {
  const band = getRiskBand(probability);
  element.dataset.risk = band;
  return band;
}

function riskColor(probability) {
  const band = getRiskBand(probability);
  return chartPalette[band] || chartPalette.neutral;
}

function renderSummary(summary) {
  const totalCustomers = Number(summary.total_customers || 0);
  const churnRate = Number(summary.overall_churn_rate_pct || 0);
  const atRiskCount = Number(summary.at_risk_customer_count || 0);
  const atRiskRevenue = Number(summary.estimated_at_risk_revenue || 0);
  const atRiskRate = totalCustomers ? (atRiskCount / totalCustomers) * 100 : 0;

  app.kpiTotalCustomers.textContent = totalCustomers.toLocaleString();
  app.kpiChurnRate.textContent = formatPercent(churnRate, 1);
  app.kpiAtRiskCustomers.textContent = atRiskCount.toLocaleString();
  app.kpiAtRiskRevenue.textContent = formatCurrency(atRiskRevenue);

  app.kpiChurnRateSubtext.textContent = `${getRiskLabel(churnRate / 100)} for the full base`;
  app.kpiAtRiskCustomersSubtext.textContent = `${atRiskRate.toFixed(1)}% of customers are above the risk threshold`;

  applyRiskState(app.kpiChurnRate.parentElement, churnRate / 100);
  applyRiskState(app.kpiAtRiskCustomers.parentElement, atRiskRate / 100);
  applyRiskState(app.kpiAtRiskRevenue.parentElement, atRiskRate / 100);
}

function renderCharts(summary) {
  const breakdowns = summary.breakdowns || {};
  buildOrUpdateChart(
    "segmentChart",
    breakdowns.segment_name || [],
    { label: "Churn rate %", sorted: true, preserveOrder: false }
  );
  buildOrUpdateChart(
    "planChart",
    breakdowns.subscription_plan || [],
    { label: "Churn rate %", sorted: false, preserveOrder: true }
  );
  buildOrUpdateChart(
    "regionChart",
    breakdowns.region || [],
    { label: "Churn rate %", sorted: true, preserveOrder: false }
  );
}

function buildOrUpdateChart(canvasId, items, options = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) {
    return;
  }

  const source = Array.isArray(items) ? [...items] : [];
  if (options.sorted) {
    source.sort((a, b) => Number(b.churn_rate_pct) - Number(a.churn_rate_pct));
  }

  const labels = source.map((item) => item.category);
  const values = source.map((item) => Number(item.churn_rate_pct));
  const colors = values.map((value) => riskColor(value / 100));

  const config = {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: options.label || "Churn rate %",
          data: values,
          backgroundColor: colors,
          borderColor: colors,
          borderWidth: 1,
          borderRadius: 12,
          maxBarThickness: 54,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          callbacks: {
            label(context) {
              return ` ${context.formattedValue}% churn`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: {
            display: false,
          },
          ticks: {
            color: "#c3cde7",
            font: {
              family: "Inter",
            },
          },
        },
        y: {
          beginAtZero: true,
          grid: {
            color: "rgba(255,255,255,0.08)",
          },
          ticks: {
            color: "#c3cde7",
            callback(value) {
              return `${value}%`;
            },
          },
        },
      },
    },
  };

  if (state.charts[canvasId]) {
    state.charts[canvasId].data = config.data;
    state.charts[canvasId].options = config.options;
    state.charts[canvasId].update();
    return;
  }

  state.charts[canvasId] = new Chart(canvas.getContext("2d"), config);
}

function renderSegments(segments) {
  if (!app.segmentCards) {
    return;
  }

  if (!segments.length) {
    app.segmentCards.innerHTML = `
      <div class="table-placeholder segment-placeholder">
        No segment summary data is available.
      </div>
    `;
    app.segmentFilterLabel.textContent = "Showing all segments";
    app.clearSegmentFilterBtn.disabled = true;
    return;
  }

  app.segmentCards.innerHTML = segments
    .map((item) => {
      const avgChurn = Number(item.avg_churn_probability || 0);
      const customerCount = Number(item.customer_count ?? item.size ?? 0);
      const riskBand = getRiskBand(avgChurn);
      const isActive = state.activeSegment === item.segment_name;
      return `
        <button
          type="button"
          class="segment-card"
          data-segment="${escapeHtml(item.segment_name)}"
          data-risk="${riskBand}"
          data-active="${isActive ? "true" : "false"}"
          aria-pressed="${isActive ? "true" : "false"}"
        >
          <div class="segment-card__header">
            <div>
              <div class="segment-card__title">${escapeHtml(item.segment_name)}</div>
              <div class="segment-card__meta">${customerCount.toLocaleString()} customers</div>
            </div>
            <span class="risk-badge risk-badge--${riskBand}">${formatPercent(avgChurn * 100, 1)}</span>
          </div>
          <div class="segment-card__stats">
            <div class="segment-stat">
              <span>% of total</span>
              <strong>${formatPercent(Number(item.pct_of_total || 0), 1)}</strong>
            </div>
            <div class="segment-stat">
              <span>Avg satisfaction</span>
              <strong>${Number(item.avg_satisfaction_score || 0).toFixed(1)}/10</strong>
            </div>
            <div class="segment-stat">
              <span>% delayed payments</span>
              <strong>${formatPercent(Number(item.pct_delayed_payments || 0), 1)}</strong>
            </div>
          </div>
        </button>
      `;
    })
    .join("");

  updateSegmentFilterControls();
}

function setActiveSegment(segmentName) {
  state.activeSegment = segmentName || null;
  state.page = 1;
  state.filteredCustomers = state.activeSegment
    ? state.allCustomers.filter((customer) => customer.segment_name === state.activeSegment)
    : state.allCustomers.slice();

  renderSegments(state.segments);
  renderCustomers();
}

function getVisibleCustomers() {
  if (Array.isArray(state.filteredCustomers) && state.filteredCustomers.length >= 0) {
    return state.filteredCustomers;
  }
  return state.allCustomers;
}

function updateSegmentFilterControls() {
  if (!app.segmentFilterLabel || !app.clearSegmentFilterBtn) {
    return;
  }

  if (state.activeSegment) {
    app.segmentFilterLabel.textContent = `Filtered by: ${state.activeSegment}`;
    app.clearSegmentFilterBtn.disabled = false;
  } else {
    app.segmentFilterLabel.textContent = "Showing all segments";
    app.clearSegmentFilterBtn.disabled = true;
  }
}

function onSegmentCardClick(event) {
  const button = event.target.closest("button[data-segment]");
  if (!button) {
    return;
  }
  const segmentName = button.dataset.segment;
  setActiveSegment(segmentName);
}

function renderCustomers() {
  const items = getVisibleCustomers();
  const totalItems = items.length;
  const totalPages = Math.max(1, Math.ceil(totalItems / state.pageSize));

  if (state.page > totalPages) {
    state.page = totalPages;
  }

  app.pageIndicator.textContent = state.activeSegment
    ? `Page ${state.page} of ${totalPages} • ${totalItems.toLocaleString()} customers in ${state.activeSegment}`
    : `Page ${state.page} of ${totalPages} • ${totalItems.toLocaleString()} customers`;
  app.prevPageBtn.disabled = state.page <= 1 || totalItems === 0;
  app.nextPageBtn.disabled = state.page >= totalPages || totalItems === 0;

  if (!items.length) {
    app.customerTableBody.innerHTML = `
      <tr>
        <td colspan="5" class="table-placeholder">
          ${state.activeSegment ? `No customers found in ${escapeHtml(state.activeSegment)}.` : "No customers available."}
        </td>
      </tr>
    `;
    return;
  }

  const start = (state.page - 1) * state.pageSize;
  const pageItems = items.slice(start, start + state.pageSize);

  if (!pageItems.length) {
    app.customerTableBody.innerHTML = `
      <tr>
        <td colspan="5" class="table-placeholder">No customers available for this page.</td>
      </tr>
    `;
    return;
  }

  app.customerTableBody.innerHTML = pageItems
    .map((item) => {
      const riskBand = getRiskBand(item.churn_probability);
      return `
        <tr tabindex="0" data-customer-id="${escapeHtml(item.customer_id)}" data-risk="${riskBand}">
          <td><strong>${escapeHtml(item.customer_id)}</strong></td>
          <td><span class="risk-badge risk-badge--${riskBand}">${formatPercent(item.churn_probability * 100, 1)}</span></td>
          <td>${escapeHtml(item.segment_name || "--")}</td>
          <td>${escapeHtml(item.subscription_plan || "--")}</td>
          <td>${escapeHtml(item.region || "--")}</td>
        </tr>
      `;
    })
    .join("");
}

function onCustomerRowClick(event) {
  const row = event.target.closest("tr[data-customer-id]");
  if (!row) {
    return;
  }
  openCustomerModal(row.dataset.customerId);
}

function onCustomerRowKeydown(event) {
  if (event.key !== "Enter" && event.key !== " ") {
    return;
  }
  const row = event.target.closest("tr[data-customer-id]");
  if (!row) {
    return;
  }
  event.preventDefault();
  openCustomerModal(row.dataset.customerId);
}

async function openCustomerModal(customerId) {
  openModal();
  app.customerModalContent.innerHTML = `<p class="modal-loading">Loading customer ${escapeHtml(customerId)}...</p>`;

  try {
    const detail = await fetchJSON(`/customers/${encodeURIComponent(customerId)}`);
    renderCustomerModal(detail);
  } catch (error) {
    app.customerModalContent.innerHTML = `
      <div class="modal-loading">
        <p>Could not load customer details.</p>
        <p>${escapeHtml(error.message || "Unknown error.")}</p>
      </div>
    `;
  }
}

function renderCustomerModal(detail) {
  const decoded = detail.decoded || {};
  const rawFeatures = detail.raw_features || {};
  const shapReasons = Array.isArray(detail.shap_reasons) && detail.shap_reasons.length
    ? detail.shap_reasons
    : [detail.reason_1, detail.reason_2, detail.reason_3]
        .filter(Boolean)
        .map((reason, index) => ({
          label: `Reason ${index + 1}`,
          text: reason,
          shap_value: index === 0 ? detail.reason_1_strength || 0 : index === 1 ? detail.reason_2_strength || 0 : detail.reason_3_strength || 0,
          magnitude: Math.abs(index === 0 ? detail.reason_1_strength || 0 : index === 1 ? detail.reason_2_strength || 0 : detail.reason_3_strength || 0),
        }));

  const maxMagnitude = Math.max(
    ...shapReasons.map((reason) => Math.abs(Number(reason.magnitude || reason.shap_value || 0))),
    1
  );

  const reasonCards = shapReasons
    .slice(0, 3)
    .map((reason, index) => {
      const shapValue = Number(reason.shap_value || 0);
      const magnitude = Math.abs(Number(reason.magnitude || shapValue));
      const width = Math.max(8, (magnitude / maxMagnitude) * 100);
      const direction = shapValue >= 0 ? "positive" : "negative";
      const directionLabel = shapValue >= 0 ? "Increased risk" : "Reduced risk";
      return `
        <article class="reason-card reason-card--${direction}">
          <div class="reason-card__label">Reason ${index + 1}</div>
          <div class="reason-card__text">${escapeHtml(reason.text || reason.reason || "")}</div>
          <div class="reason-bar">
            <progress class="reason-progress" max="100" value="${width.toFixed(0)}" data-direction="${direction}"></progress>
            <span class="reason-bar__value">${directionLabel} ${width.toFixed(0)}%</span>
          </div>
        </article>
      `;
    })
    .join("");

  const summaryCards = [
    ["Churn probability", formatPercent(detail.churn_probability * 100, 1)],
    ["Segment", detail.segment_name || "--"],
    ["Plan", decoded.subscription_plan || "--"],
    ["Payment history", decoded.payment_history || "--"],
  ]
    .map(
      ([label, value]) => `
        <div class="modal-summary-card">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
        </div>
      `
    )
    .join("");

  const metaBadges = [
    decoded.region ? `<span class="risk-badge risk-badge--neutral">${escapeHtml(decoded.region)}</span>` : "",
    decoded.device_used_most_often ? `<span class="risk-badge risk-badge--neutral">${escapeHtml(decoded.device_used_most_often)}</span>` : "",
    decoded.genre_preference ? `<span class="risk-badge risk-badge--neutral">${escapeHtml(decoded.genre_preference)}</span>` : "",
  ].join("");

  const rawFeatureOrder = [
    "subscription_length_months",
    "customer_satisfaction_score",
    "daily_watch_time_hours",
    "engagement_rate",
    "device_used_most_often",
    "genre_preference",
    "region",
    "payment_history",
    "subscription_plan",
    "support_queries_logged",
    "age",
    "monthly_income_usd",
    "promotional_offers_used",
    "number_of_profiles_created",
  ];

  const rawFeatureGrid = rawFeatureOrder
    .filter((key) => Object.prototype.hasOwnProperty.call(rawFeatures, key))
    .map(
      (key) => `
        <div class="raw-field">
          <span>${escapeHtml(formatFeatureLabel(key))}</span>
          <strong>${escapeHtml(formatFeatureValue(key, rawFeatures[key]))}</strong>
        </div>
      `
    )
    .join("");

  app.customerModalContent.innerHTML = `
    <div class="modal-header">
      <div>
        <p class="section-eyebrow">Customer drill-down</p>
        <h2 id="customerModalTitle">${escapeHtml(detail.customer_id)}</h2>
        <div class="modal-meta">${metaBadges}</div>
      </div>
      <span class="risk-badge risk-badge--${getRiskBand(detail.churn_probability)}">
        ${getRiskLabel(detail.churn_probability)}
      </span>
    </div>

    <section class="modal-summary-grid">
      ${summaryCards}
    </section>

    <section class="modal-section">
      <h3>Top SHAP reasons</h3>
      <div class="reason-cards">
        ${reasonCards}
      </div>
    </section>

    <section class="modal-section">
      <h3>Raw customer features</h3>
      <div class="modal-raw-grid">
        ${rawFeatureGrid}
      </div>
    </section>

    <p class="modal-footer">
      This detail view combines the stored customer record, the saved model output, and the SHAP explanation generated by the backend.
    </p>
  `;
}

function formatFeatureLabel(key) {
  if (key === "subscription_plan") return "Subscription plan";
  if (key === "monthly_income_usd") return "Monthly income (USD)";
  if (key === "daily_watch_time_hours") return "Daily watch time (hours)";
  if (key === "customer_satisfaction_score") return "Customer satisfaction score";
  if (key === "engagement_rate") return "Engagement rate";
  if (key === "support_queries_logged") return "Support queries logged";
  if (key === "promotional_offers_used") return "Promotional offers used";
  if (key === "number_of_profiles_created") return "Number of profiles created";
  if (key === "payment_history_delayed") return "Payment history - delayed";
  if (key === "payment_history_on_time") return "Payment history - on-time";

  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatFeatureValue(key, value) {
  if (key === "subscription_plan") {
    const planMap = { 1: "Basic", 2: "Standard", 3: "Premium" };
    return planMap[String(value)] || String(value);
  }

  if (key === "monthly_income_usd") {
    return formatCurrency(Number(value));
  }

  if (key === "daily_watch_time_hours") {
    return `${Number(value).toFixed(2)} hours`;
  }

  if (key === "churn_status") {
    return Number(value) === 1 ? "Yes" : "No";
  }

  if (key.startsWith("payment_history_") || key.startsWith("device_used_most_often_") || key.startsWith("genre_preference_") || key.startsWith("region_")) {
    return Number(value) === 1 ? "Yes" : "No";
  }

  if (typeof value === "number") {
    if (Number.isInteger(value)) {
      return value.toLocaleString();
    }
    return value.toFixed(2);
  }

  return String(value);
}

function openModal() {
  app.customerModalOverlay.classList.remove("modal-overlay--hidden");
  app.customerModalOverlay.setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");
}

function closeCustomerModal() {
  app.customerModalOverlay.classList.add("modal-overlay--hidden");
  app.customerModalOverlay.setAttribute("aria-hidden", "true");
  document.body.classList.remove("modal-open");
}

function onModalOverlayClick(event) {
  if (event.target === app.customerModalOverlay) {
    closeCustomerModal();
  }
}

function onGlobalKeydown(event) {
  if (event.key === "Escape" && !app.customerModalOverlay.classList.contains("modal-overlay--hidden")) {
    closeCustomerModal();
  }
}

async function handlePredictSubmit(event) {
  event.preventDefault();
  const formData = new FormData(app.predictForm);
  const payload = formDataToPayload(formData);

  try {
    const result = await fetchJSON("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    renderPredictResult(result);
  } catch (error) {
    showNotice(error.message || "Prediction failed. Check the backend logs and try again.");
  }
}

function formDataToPayload(formData) {
  return {
    subscription_length_months: Number(formData.get("subscription_length_months")),
    customer_satisfaction_score: Number(formData.get("customer_satisfaction_score")),
    daily_watch_time_hours: Number(formData.get("daily_watch_time_hours")),
    engagement_rate: Number(formData.get("engagement_rate")),
    device_used_most_often: formData.get("device_used_most_often"),
    genre_preference: formData.get("genre_preference"),
    region: formData.get("region"),
    payment_history: formData.get("payment_history"),
    subscription_plan: formData.get("subscription_plan"),
    support_queries_logged: Number(formData.get("support_queries_logged")),
    age: Number(formData.get("age")),
    monthly_income_usd: Number(formData.get("monthly_income_usd")),
    promotional_offers_used: Number(formData.get("promotional_offers_used")),
    number_of_profiles_created: Number(formData.get("number_of_profiles_created")),
  };
}

function renderPredictResult(result) {
  const riskBand = getRiskBand(result.churn_probability);
  const probabilityPct = result.churn_probability * 100;

  app.predictResultPanel.dataset.risk = riskBand;
  app.predictRiskBadge.className = `risk-badge risk-badge--${riskBand}`;
  app.predictRiskBadge.textContent = `${getRiskLabel(result.churn_probability)} risk`;
  app.predictProbability.textContent = formatPercent(probabilityPct, 1);
  app.predictBandText.textContent = riskBand === "high"
    ? "This customer shows a strong likelihood of churn."
    : riskBand === "medium"
      ? "This customer sits in the watchlist range."
      : "This customer currently looks relatively stable.";
  app.predictProgress.value = probabilityPct;

  const reasons = [result.reason_1, result.reason_2, result.reason_3].filter(Boolean);
  app.predictReasonsList.innerHTML = reasons
    .map((reason) => `<li>${escapeHtml(reason)}</li>`)
    .join("");
}

function clearPredictResult() {
  app.predictResultPanel.dataset.risk = "neutral";
  app.predictRiskBadge.className = "risk-badge risk-badge--neutral";
  app.predictRiskBadge.textContent = "Awaiting input";
  app.predictProbability.textContent = "--";
  app.predictBandText.textContent = "Submit the form to generate a prediction.";
  app.predictProgress.value = 0;
  app.predictReasonsList.innerHTML = "<li>Reasons will appear here after prediction.</li>";
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
