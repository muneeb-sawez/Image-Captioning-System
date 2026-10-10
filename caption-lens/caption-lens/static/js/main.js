const $ = s => document.querySelector(s);
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
/* typewriter (cancels any earlier run on the same element) */
function type(el, text){
  const id = el._id = (el._id || 0) + 1;
  el.classList.remove("idle"); el.textContent = "";
  return new Promise(res => {
    if (reduced){ el.textContent = text; return res(); }
    let i = 0;
    (function step(){
      if (el._id !== id) return res();
      el.textContent = text.slice(0, ++i);
      i < text.length ? setTimeout(step, 26) : res();
    })();
  });
}

