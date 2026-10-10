/**
 * Caption Lens — Main UI Interactions
 */

document.addEventListener("DOMContentLoaded", () => {
  // 1. Mobile Navigation Drawer Toggle
  const mobileToggle = document.getElementById("mobileNavToggle");
  const mobileDrawer = document.getElementById("mobileDrawer");

  if (mobileToggle && mobileDrawer) {
    mobileToggle.addEventListener("click", () => {
      mobileDrawer.classList.toggle("open");
      const icon = mobileToggle.querySelector("i");
      if (icon) {
        icon.className = mobileDrawer.classList.contains("open") ? "bi bi-x-lg" : "bi bi-list";
      }
    });

    // Close mobile drawer when clicking any link inside it
    mobileDrawer.querySelectorAll("a").forEach(link => {
      link.addEventListener("click", () => {
        mobileDrawer.classList.remove("open");
        const icon = mobileToggle.querySelector("i");
        if (icon) icon.className = "bi bi-list";
      });
    });
  }

  // 2. Smooth Scrolling Offset for Anchor Links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener("click", function(e) {
      const targetId = this.getAttribute("href");
      if (targetId === "#") return;
      const targetEl = document.querySelector(targetId);
      if (targetEl) {
        e.preventDefault();
        targetEl.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });
});
