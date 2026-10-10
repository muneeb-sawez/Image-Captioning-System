/* reveal the pipeline once, in order */
new IntersectionObserver((e, o) => e.forEach(x => { if (x.isIntersecting){ x.target.classList.add("on"); o.unobserve(x.target); } }), {threshold:.3}).observe($("#pipe"));

