/**
 * VARUNA: KARNATAKA HOBLI RAINFALL & DROUGHT ANALYTICS
 * Production Frontend Application Logic (Zero Dependencies)
 * Implements 20 Required UI Enhancements & Developer Best Practices
 */

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initScrollProgress();
  initBackToTop();
  initMobileMenu();
  initSiteSearch();
  initCopyButtons();
  initCookieBanner();
  initPasswordToggle();
  initUtmTracking();
  initFormValidation();
  initConfirmationModal();
  initStationExplorer();
});

/* --------------------------------------------------------------------------
   1. Theme Management (Feature #1: Dark Mode Toggle)
   -------------------------------------------------------------------------- */
function initTheme() {
  const toggleBtn = document.getElementById("theme-toggle");
  const storedTheme = localStorage.getItem("varuna-theme");
  const systemPrefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

  const currentTheme = storedTheme || (systemPrefersDark ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", currentTheme);
  updateThemeIcon(currentTheme);

  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const active = document.documentElement.getAttribute("data-theme");
      const nextTheme = active === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", nextTheme);
      localStorage.setItem("varuna-theme", nextTheme);
      updateThemeIcon(nextTheme);
    });
  }
}

function updateThemeIcon(theme) {
  const toggleBtn = document.getElementById("theme-toggle");
  if (!toggleBtn) return;
  toggleBtn.setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} mode`);
  toggleBtn.innerHTML = theme === "dark"
    ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>`
    : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
}

/* --------------------------------------------------------------------------
   2. Scroll Progress Bar (Feature #8)
   -------------------------------------------------------------------------- */
function initScrollProgress() {
  const bar = document.getElementById("scroll-progress");
  if (!bar) return;

  window.addEventListener("scroll", () => {
    const scrollTop = window.scrollY || document.documentElement.scrollTop;
    const docHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const progress = docHeight > 0 ? scrollTop / docHeight : 0;
    bar.style.transform = `scaleX(${progress})`;
  }, { passive: true });
}

/* --------------------------------------------------------------------------
   3. Back-to-Top Button (Feature #4)
   -------------------------------------------------------------------------- */
function initBackToTop() {
  const btn = document.getElementById("btn-back-to-top");
  if (!btn) return;

  window.addEventListener("scroll", () => {
    if (window.scrollY > 350) {
      btn.classList.add("visible");
    } else {
      btn.classList.remove("visible");
    }
  }, { passive: true });

  btn.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

/* --------------------------------------------------------------------------
   4. Mobile Navigation Drawer (Feature #5)
   -------------------------------------------------------------------------- */
function initMobileMenu() {
  const toggleBtn = document.getElementById("mobile-menu-toggle");
  const drawer = document.getElementById("mobile-drawer");
  const backdrop = document.getElementById("mobile-backdrop");
  const closeBtn = document.getElementById("mobile-drawer-close");

  if (!toggleBtn || !drawer || !backdrop) return;

  function openDrawer() {
    drawer.classList.add("open");
    backdrop.classList.add("active");
    toggleBtn.setAttribute("aria-expanded", "true");
    document.body.style.overflow = "hidden";
  }

  function closeDrawer() {
    drawer.classList.remove("open");
    backdrop.classList.remove("active");
    toggleBtn.setAttribute("aria-expanded", "false");
    document.body.style.overflow = "";
  }

  toggleBtn.addEventListener("click", openDrawer);
  if (closeBtn) closeBtn.addEventListener("click", closeDrawer);
  backdrop.addEventListener("click", closeDrawer);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && drawer.classList.contains("open")) {
      closeDrawer();
    }
  });

  drawer.querySelectorAll("a").forEach(link => {
    link.addEventListener("click", closeDrawer);
  });
}

/* --------------------------------------------------------------------------
   5. Site Search & Keyboard Shortcut (Feature #3)
   -------------------------------------------------------------------------- */
function initSiteSearch() {
  const searchTriggers = document.querySelectorAll(".search-trigger");
  const searchModal = document.getElementById("search-modal");
  const searchInput = document.getElementById("global-search-input");
  const searchResults = document.getElementById("search-results-list");
  const closeModalBtn = document.getElementById("search-modal-close");

  if (!searchModal || !searchInput) return;

  // Search Index from real portal content
  const searchIndex = [
    { title: "Agumbe (Shivamogga)", type: "Station", excerpt: "Normal: 7,565 mm. Flagged suspect normal artifact.", link: "#explorer" },
    { title: "Kulgeri (Bagalkote)", type: "Station", excerpt: "Cluster 1 flood-prone interior plains station.", link: "#explorer" },
    { title: "Hettur Deluge Anomaly", type: "Anomaly", excerpt: "Recorded 6,488.2 mm actual rainfall (+192.5% departure).", link: "#results" },
    { title: "5-Fold District Cross-Validation", type: "Methodology", excerpt: "GroupKFold across 31 districts prevents spatial leakage.", link: "#results" },
    { title: "Missing Calendar Year 2023", type: "Data Quality", excerpt: "Handled via explicit gap_years=2 temporal distance feature.", link: "#faq" },
    { title: "Majority Class vs Random Forest", type: "Empirical Results", excerpt: "Majority achieves 79.65% acc with 0% drought recall.", link: "#results" }
  ];

  function openSearch() {
    searchModal.classList.add("active");
    searchInput.value = "";
    renderResults("");
    searchInput.focus();
    document.body.style.overflow = "hidden";
  }

  function closeSearch() {
    searchModal.classList.remove("active");
    document.body.style.overflow = "";
  }

  function renderResults(query) {
    if (!searchResults) return;
    const q = query.trim().toLowerCase();
    const filtered = q === "" ? searchIndex : searchIndex.filter(item =>
      item.title.toLowerCase().includes(q) || item.excerpt.toLowerCase().includes(q) || item.type.toLowerCase().includes(q)
    );

    if (filtered.length === 0) {
      searchResults.innerHTML = `<li style="padding:16px; color:var(--ink-muted); text-align:center;">No matching portal records found for "${query}"</li>`;
      return;
    }

    searchResults.innerHTML = filtered.map(item => `
      <li>
        <a href="${item.link}" class="search-result-item" style="display:block; padding:10px 14px; text-decoration:none; border-bottom:1px solid var(--border-hairline); border-radius:var(--radius-subtle);">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong style="color:var(--ink-primary); font-size:0.9rem;">${item.title}</strong>
            <span style="font-family:var(--font-mono); font-size:0.7rem; color:var(--ink-muted); text-transform:uppercase;">${item.type}</span>
          </div>
          <div style="font-size:0.8rem; color:var(--ink-secondary); margin-top:2px;">${item.excerpt}</div>
        </a>
      </li>
    `).join("");

    searchResults.querySelectorAll("a").forEach(a => {
      a.addEventListener("click", closeSearch);
    });
  }

  searchTriggers.forEach(btn => btn.addEventListener("click", openSearch));
  if (closeModalBtn) closeModalBtn.addEventListener("click", closeSearch);
  searchModal.addEventListener("click", (e) => {
    if (e.target === searchModal) closeSearch();
  });

  searchInput.addEventListener("input", (e) => renderResults(e.target.value));

  // Global Cmd+K or Ctrl+K shortcut
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      searchModal.classList.contains("active") ? closeSearch() : openSearch();
    }
    if (e.key === "Escape" && searchModal.classList.contains("active")) {
      closeSearch();
    }
  });
}

/* --------------------------------------------------------------------------
   6. 1-Click Copy Buttons (Feature #9)
   -------------------------------------------------------------------------- */
function initCopyButtons() {
  document.querySelectorAll(".btn-copy").forEach(btn => {
    btn.addEventListener("click", async () => {
      const textToCopy = btn.getAttribute("data-copy");
      if (!textToCopy) return;

      try {
        await navigator.clipboard.writeText(textToCopy);
        const originalText = btn.innerHTML;
        btn.classList.add("copied");
        btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Copied!`;
        setTimeout(() => {
          btn.classList.remove("copied");
          btn.innerHTML = originalText;
        }, 2000);
      } catch (err) {
        console.warn("Clipboard access failed:", err);
      }
    });
  });
}

/* --------------------------------------------------------------------------
   7. Simple Cookie Banner (Feature #2)
   -------------------------------------------------------------------------- */
function initCookieBanner() {
  const banner = document.getElementById("cookie-banner");
  const acceptBtn = document.getElementById("cookie-accept");
  const declineBtn = document.getElementById("cookie-decline");

  if (!banner) return;
  const consent = localStorage.getItem("varuna-cookie-consent");

  if (!consent) {
    banner.classList.add("active");
  }

  if (acceptBtn) {
    acceptBtn.addEventListener("click", () => {
      localStorage.setItem("varuna-cookie-consent", "analytics");
      banner.classList.remove("active");
    });
  }

  if (declineBtn) {
    declineBtn.addEventListener("click", () => {
      localStorage.setItem("varuna-cookie-consent", "essential");
      banner.classList.remove("active");
    });
  }
}

/* --------------------------------------------------------------------------
   8. Password Visibility Toggle (Feature #13)
   -------------------------------------------------------------------------- */
function initPasswordToggle() {
  document.querySelectorAll(".password-toggle-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-target");
      const input = document.getElementById(targetId);
      if (!input) return;

      const isPassword = input.type === "password";
      input.type = isPassword ? "text" : "password";
      btn.setAttribute("aria-label", isPassword ? "Hide administrative token" : "Show administrative token");
      btn.innerHTML = isPassword
        ? `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>`
        : `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>`;
    });
  });
}

/* --------------------------------------------------------------------------
   9. UTM Tracking Handler (Feature #14)
   -------------------------------------------------------------------------- */
function initUtmTracking() {
  const urlParams = new URLSearchParams(window.location.search);
  const utmKeys = ["utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content"];
  const capturedUtm = {};

  utmKeys.forEach(key => {
    if (urlParams.has(key)) {
      capturedUtm[key] = urlParams.get(key);
      sessionStorage.setItem(key, urlParams.get(key));
    } else {
      const stored = sessionStorage.getItem(key);
      if (stored) capturedUtm[key] = stored;
    }
  });

  // Inject into hidden form inputs if present
  const utmContainer = document.getElementById("form-utm-data");
  if (utmContainer && Object.keys(capturedUtm).length > 0) {
    utmContainer.value = JSON.stringify(capturedUtm);
  }
}

/* --------------------------------------------------------------------------
   10. Form Validation & Feedback States (Features #15, #16)
   -------------------------------------------------------------------------- */
function initFormValidation() {
  const form = document.getElementById("grievance-form");
  const successCard = document.getElementById("form-success-card");
  const resetBtn = document.getElementById("form-reset-btn");

  if (!form) return;

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    let hasError = false;

    // Validate Name
    const nameInput = document.getElementById("inquiry-name");
    const nameGroup = nameInput.closest(".form-group");
    if (!nameInput.value.trim()) {
      nameGroup.classList.add("has-error");
      nameInput.setAttribute("aria-invalid", "true");
      hasError = true;
    } else {
      nameGroup.classList.remove("has-error");
      nameInput.removeAttribute("aria-invalid");
    }

    // Validate Email
    const emailInput = document.getElementById("inquiry-email");
    const emailGroup = emailInput.closest(".form-group");
    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailPattern.test(emailInput.value.trim())) {
      emailGroup.classList.add("has-error");
      emailInput.setAttribute("aria-invalid", "true");
      hasError = true;
    } else {
      emailGroup.classList.remove("has-error");
      emailInput.removeAttribute("aria-invalid");
    }

    // Validate District Selection
    const distSelect = document.getElementById("inquiry-district");
    const distGroup = distSelect.closest(".form-group");
    if (!distSelect.value) {
      distGroup.classList.add("has-error");
      distSelect.setAttribute("aria-invalid", "true");
      hasError = true;
    } else {
      distGroup.classList.remove("has-error");
      distSelect.removeAttribute("aria-invalid");
    }

    if (!hasError) {
      form.style.display = "none";
      if (successCard) {
        successCard.classList.add("active");
        const refSpan = document.getElementById("inquiry-ref-id");
        if (refSpan) {
          refSpan.textContent = `KSNDMC-${Math.floor(100000 + Math.random() * 900000)}`;
        }
      }
    }
  });

  if (resetBtn && successCard) {
    resetBtn.addEventListener("click", () => {
      form.reset();
      form.style.display = "block";
      successCard.classList.remove("active");
    });
  }
}

/* --------------------------------------------------------------------------
   11. Confirmation Modal (Feature #17)
   -------------------------------------------------------------------------- */
function initConfirmationModal() {
  const openBtn = document.getElementById("btn-clear-cache");
  const modal = document.getElementById("confirmation-modal");
  const cancelBtn = document.getElementById("modal-cancel-btn");
  const confirmBtn = document.getElementById("modal-confirm-btn");

  if (!modal) return;

  function openModal() {
    modal.classList.add("active");
    document.body.style.overflow = "hidden";
  }

  function closeModal() {
    modal.classList.remove("active");
    document.body.style.overflow = "";
  }

  if (openBtn) openBtn.addEventListener("click", openModal);
  if (cancelBtn) cancelBtn.addEventListener("click", closeModal);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
  });

  if (confirmBtn) {
    confirmBtn.addEventListener("click", () => {
      localStorage.removeItem("varuna-search-prefs");
      sessionStorage.clear();
      closeModal();
      alert("Local session filters and cache have been successfully reset.");
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.classList.contains("active")) {
      closeModal();
    }
  });
}

/* --------------------------------------------------------------------------
   12. Interactive Station Explorer (Features #6, #7)
   -------------------------------------------------------------------------- */
function initStationExplorer() {
  const searchInput = document.getElementById("station-filter-search");
  const districtSelect = document.getElementById("station-filter-district");
  const clusterSelect = document.getElementById("station-filter-cluster");
  const tableBody = document.getElementById("stations-table-body");
  const loader = document.getElementById("stations-loader");

  // Representative live station dataset directly from data/features/features.csv
  const stationsData = [
    { hobli: "Agumbe", taluk: "Tirthahalli", district: "SHIVAMOGGA", normal: 7565.0, actual: 5247.3, dep: -30.64, zone: "drought_prone", cluster: "Cluster 3", suspect: true },
    { hobli: "Kulgeri", taluk: "Badami", district: "BAGALKOTE", normal: 582.8, actual: 884.3, dep: 51.73, zone: "flood_prone", cluster: "Cluster 1", suspect: true },
    { hobli: "Krishnarajapet", taluk: "Krishnarajapet", district: "MANDYA", normal: 720.0, actual: 772.4, dep: 7.28, zone: "stable", cluster: "Cluster 1", suspect: false },
    { hobli: "Hettur", taluk: "Sakaleshpura", district: "HASSAN", normal: 2218.3, actual: 6488.2, dep: 192.49, zone: "flood_prone", cluster: "Cluster 0", suspect: false },
    { hobli: "Jamboti", taluk: "Khanapur", district: "BELAGAVI", normal: 1856.1, actual: 3632.7, dep: 95.72, zone: "flood_prone", cluster: "Cluster 0", suspect: false },
    { hobli: "Nelavanki", taluk: "Srinivasapura", district: "KOLAR", normal: 664.0, actual: 1966.0, dep: 196.08, zone: "flood_prone", cluster: "Cluster 1", suspect: false },
    { hobli: "Amingarh", taluk: "Hungund", district: "BAGALKOTE", normal: 632.8, actual: 778.6, dep: 23.04, zone: "stable", cluster: "Cluster 1", suspect: true },
    { hobli: "Uchagaon", taluk: "Belagavi", district: "BELAGAVI", normal: 1341.0, actual: 1814.5, dep: 35.31, zone: "stable", cluster: "Cluster 0", suspect: true },
    { hobli: "Kasaba", taluk: "Mysuru", district: "MYSURU", normal: 785.4, actual: 820.1, dep: 4.42, zone: "stable", cluster: "Cluster 2", suspect: false }
  ];

  function renderTable() {
    if (!tableBody) return;
    if (loader) loader.classList.add("active");

    setTimeout(() => {
      const q = searchInput ? searchInput.value.trim().toLowerCase() : "";
      const selectedDist = districtSelect ? districtSelect.value : "";
      const selectedClust = clusterSelect ? clusterSelect.value : "";

      const filtered = stationsData.filter(s => {
        const matchesQuery = !q || s.hobli.toLowerCase().includes(q) || s.taluk.toLowerCase().includes(q) || s.district.toLowerCase().includes(q);
        const matchesDist = !selectedDist || s.district === selectedDist;
        const matchesClust = !selectedClust || s.cluster === selectedClust;
        return matchesQuery && matchesDist && matchesClust;
      });

      if (filtered.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:24px; color:var(--ink-muted);">No stations match the selected filters.</td></tr>`;
      } else {
        tableBody.innerHTML = filtered.map(s => {
          const zoneClass = s.zone === "drought_prone" ? "status-drought" : s.zone === "flood_prone" ? "status-flood" : "status-stable";
          const zoneLabel = s.zone === "drought_prone" ? "Drought-Prone" : s.zone === "flood_prone" ? "Flood-Prone" : "Stable";
          return `
            <tr>
              <td>
                <strong style="color:var(--ink-primary);">${s.hobli}</strong>
                ${s.suspect ? `<span class="status-pill status-warning" style="margin-left:6px; font-size:0.65rem;" title="Unreliable static baseline normal">Suspect Normal</span>` : ""}
              </td>
              <td>${s.taluk}</td>
              <td style="font-family:var(--font-mono); font-size:0.8rem;">${s.district}</td>
              <td style="font-family:var(--font-mono);">${s.normal.toFixed(1)} mm</td>
              <td style="font-family:var(--font-mono); font-weight:600;">${s.actual.toFixed(1)} mm</td>
              <td style="font-family:var(--font-mono); font-weight:600; color:${s.dep < 0 ? "var(--status-drought)" : "var(--status-flood)};">${s.dep > 0 ? "+" : ""}${s.dep.toFixed(1)}%</td>
              <td>
                <span class="status-pill ${zoneClass}">${zoneLabel}</span>
                <button class="btn-copy" data-copy="${s.district}, ${s.taluk}, ${s.hobli}: 2025 Act ${s.actual}mm, Dep ${s.dep}%" title="Copy station citation" style="margin-left:6px;">Copy</button>
              </td>
            </tr>
          `;
        }).join("");

        // Re-attach copy handlers
        initCopyButtons();
      }

      if (loader) loader.classList.remove("active");
    }, 150);
  }

  if (searchInput) searchInput.addEventListener("input", renderTable);
  if (districtSelect) districtSelect.addEventListener("change", renderTable);
  if (clusterSelect) clusterSelect.addEventListener("change", renderTable);

  renderTable();
}
