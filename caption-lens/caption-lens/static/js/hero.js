/* hero: loops through sample captions */
(async function hero(){
  const lines = ["a dog is running in a grassy park", "a sunset over the hills", "two people walking along a beach"];
  const el = $("#heroCap");
  for (let i = 0; ; i = (i + 1) % lines.length){
    await type(el, lines[i]);
    await new Promise(r => setTimeout(r, 2200));
  }
})();

