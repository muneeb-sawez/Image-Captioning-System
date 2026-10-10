/* metrics count up when visible (dash if left at 0) */
document.querySelectorAll(".metric .n").forEach(n => {
  new IntersectionObserver((es, o) => es.forEach(x => {
    if (!x.isIntersecting) return; o.unobserve(n);
    const v = parseFloat(n.dataset.value); if (!(v > 0)) return;
    const t0 = performance.now();
    (function f(t){ const k = Math.min((t - t0) / 1200, 1); n.textContent = (v * k).toFixed(2); if (k < 1) requestAnimationFrame(f); })(t0);
  }), {threshold:.6}).observe(n);
});

